//! A minimal single-core GDB server backed by the MCD [`Core`] API.
//!
//! This exposes core 0 of the connected system over the GDB Remote Serial
//! Protocol so that a TriCore GDB (e.g. HighTec `tricore-gdb`) can attach to a
//! running target, inspect and modify registers and memory, set hardware
//! breakpoints and single step.
//!
//! The server is intentionally single-threaded: the MCD [`Core`] wraps a raw
//! pointer (and is therefore `!Send`) and the owning [`System`] tears down the
//! MCD library on drop, so the [`Core`] is held for the whole session on a
//! single thread, which matches `gdbstub`'s blocking event loop model.

use std::net::{TcpListener, TcpStream};
use std::num::NonZeroUsize;

use anyhow::Context;
use gdbstub::arch::{Arch, RegId, Registers};
use gdbstub::common::Signal;
use gdbstub::conn::ConnectionExt;
use gdbstub::stub::run_blocking::{self, WaitForStopReasonError};
use gdbstub::stub::{GdbStub, SingleThreadStopReason};
use gdbstub::target::ext::base::singlethread::{
    SingleThreadBase, SingleThreadResume, SingleThreadResumeOps, SingleThreadSingleStep,
    SingleThreadSingleStepOps,
};
use gdbstub::target::ext::base::BaseOps;
use gdbstub::target::ext::breakpoints::{Breakpoints, BreakpointsOps, HwBreakpoint, HwBreakpointOps};
use gdbstub::target::{Target, TargetError, TargetResult};

use rust_mcd::breakpoint::TriggerType;
use rust_mcd::core::{Core, CoreState};
use rust_mcd::system::System;

/// Number of registers exposed to GDB: D0-D15, A0-A15, PC, PSW, PCXI.
const REG_COUNT: usize = 35;
/// Number of bytes in the serialized register file (all registers are 32-bit).
const REG_BYTES: usize = REG_COUNT * 4;

/// Target description handed to GDB so that it agrees on the register layout.
///
/// The register order here must match [`TricoreRegs`] and the read/write order
/// in [`TricoreTarget::read_all`] / [`TricoreTarget::write_all`].
const TARGET_XML: &str = r#"<?xml version="1.0"?>
<!DOCTYPE target SYSTEM "gdb-target.dtd">
<target version="1.0">
  <architecture>tricore</architecture>
  <feature name="org.gnu.gdb.tricore.core">
    <reg name="d0" bitsize="32" type="uint32"/>
    <reg name="d1" bitsize="32" type="uint32"/>
    <reg name="d2" bitsize="32" type="uint32"/>
    <reg name="d3" bitsize="32" type="uint32"/>
    <reg name="d4" bitsize="32" type="uint32"/>
    <reg name="d5" bitsize="32" type="uint32"/>
    <reg name="d6" bitsize="32" type="uint32"/>
    <reg name="d7" bitsize="32" type="uint32"/>
    <reg name="d8" bitsize="32" type="uint32"/>
    <reg name="d9" bitsize="32" type="uint32"/>
    <reg name="d10" bitsize="32" type="uint32"/>
    <reg name="d11" bitsize="32" type="uint32"/>
    <reg name="d12" bitsize="32" type="uint32"/>
    <reg name="d13" bitsize="32" type="uint32"/>
    <reg name="d14" bitsize="32" type="uint32"/>
    <reg name="d15" bitsize="32" type="uint32"/>
    <reg name="a0" bitsize="32" type="data_ptr"/>
    <reg name="a1" bitsize="32" type="data_ptr"/>
    <reg name="a2" bitsize="32" type="data_ptr"/>
    <reg name="a3" bitsize="32" type="data_ptr"/>
    <reg name="a4" bitsize="32" type="data_ptr"/>
    <reg name="a5" bitsize="32" type="data_ptr"/>
    <reg name="a6" bitsize="32" type="data_ptr"/>
    <reg name="a7" bitsize="32" type="data_ptr"/>
    <reg name="a8" bitsize="32" type="data_ptr"/>
    <reg name="a9" bitsize="32" type="data_ptr"/>
    <reg name="a10" bitsize="32" type="data_ptr"/>
    <reg name="a11" bitsize="32" type="data_ptr"/>
    <reg name="a12" bitsize="32" type="data_ptr"/>
    <reg name="a13" bitsize="32" type="data_ptr"/>
    <reg name="a14" bitsize="32" type="data_ptr"/>
    <reg name="a15" bitsize="32" type="data_ptr"/>
    <reg name="pc" bitsize="32" type="code_ptr"/>
    <reg name="psw" bitsize="32" type="uint32"/>
    <reg name="pcxi" bitsize="32" type="uint32"/>
  </feature>
</target>"#;

/// The TriCore register file as seen by GDB.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TricoreRegs {
    /// Data registers D0-D15.
    pub d: [u32; 16],
    /// Address registers A0-A15 (A10 is the stack pointer, A11 the return address).
    pub a: [u32; 16],
    /// Program counter.
    pub pc: u32,
    /// Program status word.
    pub psw: u32,
    /// Previous context information.
    pub pcxi: u32,
}

impl Registers for TricoreRegs {
    type ProgramCounter = u32;

    fn pc(&self) -> Self::ProgramCounter {
        self.pc
    }

    fn gdb_serialize(&self, mut write_byte: impl FnMut(Option<u8>)) {
        let mut push = |value: u32| {
            for byte in value.to_le_bytes() {
                write_byte(Some(byte));
            }
        };
        for reg in self.d {
            push(reg);
        }
        for reg in self.a {
            push(reg);
        }
        push(self.pc);
        push(self.psw);
        push(self.pcxi);
    }

    fn gdb_deserialize(&mut self, bytes: &[u8]) -> Result<(), ()> {
        if bytes.len() < REG_BYTES {
            return Err(());
        }
        let word = |index: usize| -> u32 {
            let offset = index * 4;
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        };
        for i in 0..16 {
            self.d[i] = word(i);
        }
        for i in 0..16 {
            self.a[i] = word(16 + i);
        }
        self.pc = word(32);
        self.psw = word(33);
        self.pcxi = word(34);
        Ok(())
    }
}

/// Identifies a single register by its index in the register file.
///
/// The index is currently unused because single-register access (`p`/`P`
/// packets) is not implemented; GDB falls back to whole-register-file transfers.
#[derive(Debug, Clone, Copy)]
pub struct TricoreRegId(#[allow(dead_code)] usize);

impl RegId for TricoreRegId {
    fn from_raw_id(id: usize) -> Option<(Self, Option<NonZeroUsize>)> {
        if id < REG_COUNT {
            Some((TricoreRegId(id), Some(NonZeroUsize::new(4).unwrap())))
        } else {
            None
        }
    }
}

/// The TriCore architecture description for `gdbstub`.
pub enum Tricore {}

impl Arch for Tricore {
    type Usize = u32;
    type Registers = TricoreRegs;
    type BreakpointKind = usize;
    type RegId = TricoreRegId;

    fn target_description_xml() -> Option<&'static str> {
        Some(TARGET_XML)
    }
}

/// Whether the last resume request was a continue or a single step. Used to
/// report the correct stop reason once the core halts.
#[derive(Clone, Copy)]
enum ResumeKind {
    Continue,
    Step,
}

/// A `gdbstub` target that drives a single MCD [`Core`].
///
/// `'a` ties the [`Core`] (and the breakpoints created on it) to the [`System`]
/// it was opened from, which must outlive the target.
pub struct TricoreTarget<'a> {
    core: Core<'a>,
    /// Installed hardware breakpoints as `(address, trigger_id)` pairs.
    breakpoints: Vec<(u32, u32)>,
    resume_kind: ResumeKind,
}

/// Converts an MCD/anyhow error into a fatal `gdbstub` target error.
fn fatal(error: anyhow::Error) -> TargetError<anyhow::Error> {
    TargetError::Fatal(error)
}

impl<'a> TricoreTarget<'a> {
    fn new(core: Core<'a>) -> Self {
        TricoreTarget {
            core,
            breakpoints: Vec::new(),
            resume_kind: ResumeKind::Continue,
        }
    }

    /// Reads the whole register file from the core, by register name.
    fn read_all(&self) -> anyhow::Result<TricoreRegs> {
        let groups = self.core.register_groups()?;
        let group = groups.get_group(0)?;
        let read = |name: &str| -> anyhow::Result<u32> {
            group
                .register(name)
                .with_context(|| format!("Core has no register {name}"))?
                .read()
        };

        let mut regs = TricoreRegs::default();
        for i in 0..16 {
            regs.d[i] = read(&format!("D{i}"))?;
        }
        for i in 0..16 {
            regs.a[i] = read(&format!("A{i}"))?;
        }
        regs.pc = read("PC")?;
        regs.psw = read("PSW")?;
        regs.pcxi = read("PCXI")?;
        Ok(regs)
    }

    /// Writes the whole register file to the core, by register name.
    fn write_all(&self, regs: &TricoreRegs) -> anyhow::Result<()> {
        let groups = self.core.register_groups()?;
        let group = groups.get_group(0)?;
        let write = |name: &str, value: u32| -> anyhow::Result<()> {
            group
                .register(name)
                .with_context(|| format!("Core has no register {name}"))?
                .write(value)
        };

        for i in 0..16 {
            write(&format!("D{i}"), regs.d[i])?;
        }
        for i in 0..16 {
            write(&format!("A{i}"), regs.a[i])?;
        }
        write("PC", regs.pc)?;
        write("PSW", regs.psw)?;
        write("PCXI", regs.pcxi)?;
        Ok(())
    }

    /// Polls the core once and returns a stop reason if it is no longer running.
    fn poll_stop(&mut self) -> Option<SingleThreadStopReason<u32>> {
        match self.core.query_state() {
            Ok(info) if info.state != CoreState::Running => Some(match self.resume_kind {
                ResumeKind::Step => SingleThreadStopReason::DoneStep,
                ResumeKind::Continue => SingleThreadStopReason::HwBreak(()),
            }),
            Ok(_) => None,
            Err(error) => {
                // Transient query errors can happen around resets; keep polling
                // rather than tearing down the whole session.
                log::warn!("Could not query core state: {error}");
                None
            }
        }
    }
}

impl Target for TricoreTarget<'_> {
    type Arch = Tricore;
    type Error = anyhow::Error;

    fn base_ops(&mut self) -> BaseOps<'_, Self::Arch, Self::Error> {
        BaseOps::SingleThread(self)
    }

    #[inline(always)]
    fn support_breakpoints(&mut self) -> Option<BreakpointsOps<'_, Self>> {
        Some(self)
    }
}

impl SingleThreadBase for TricoreTarget<'_> {
    fn read_registers(&mut self, regs: &mut TricoreRegs) -> TargetResult<(), Self> {
        *regs = self.read_all().map_err(fatal)?;
        Ok(())
    }

    fn write_registers(&mut self, regs: &TricoreRegs) -> TargetResult<(), Self> {
        self.write_all(regs).map_err(fatal)?;
        Ok(())
    }

    fn read_addrs(&mut self, start_addr: u32, data: &mut [u8]) -> TargetResult<usize, Self> {
        // Reads are chunked to stay within the MCD payload size limit.
        const CHUNK: usize = 0x400;
        let mut offset = 0;
        while offset < data.len() {
            let len = core::cmp::min(CHUNK, data.len() - offset);
            let bytes = self
                .core
                .read_bytes(start_addr as u64 + offset as u64, len)
                .map_err(fatal)?;
            data[offset..offset + len].copy_from_slice(&bytes);
            offset += len;
        }
        Ok(data.len())
    }

    fn write_addrs(&mut self, start_addr: u32, data: &[u8]) -> TargetResult<(), Self> {
        const CHUNK: usize = 0x400;
        let mut offset = 0;
        while offset < data.len() {
            let len = core::cmp::min(CHUNK, data.len() - offset);
            self.core
                .write(
                    start_addr as u64 + offset as u64,
                    data[offset..offset + len].to_vec(),
                )
                .map_err(fatal)?;
            offset += len;
        }
        Ok(())
    }

    #[inline(always)]
    fn support_resume(&mut self) -> Option<SingleThreadResumeOps<'_, Self>> {
        Some(self)
    }
}

impl SingleThreadResume for TricoreTarget<'_> {
    fn resume(&mut self, _signal: Option<Signal>) -> Result<(), Self::Error> {
        self.resume_kind = ResumeKind::Continue;
        self.core.run()
    }

    #[inline(always)]
    fn support_single_step(&mut self) -> Option<SingleThreadSingleStepOps<'_, Self>> {
        Some(self)
    }
}

impl SingleThreadSingleStep for TricoreTarget<'_> {
    fn step(&mut self, _signal: Option<Signal>) -> Result<(), Self::Error> {
        self.resume_kind = ResumeKind::Step;
        self.core.step()
    }
}

impl Breakpoints for TricoreTarget<'_> {
    #[inline(always)]
    fn support_hw_breakpoint(&mut self) -> Option<HwBreakpointOps<'_, Self>> {
        Some(self)
    }
}

impl HwBreakpoint for TricoreTarget<'_> {
    fn add_hw_breakpoint(
        &mut self,
        addr: u32,
        _kind: usize,
    ) -> TargetResult<bool, Self> {
        let trigger = self
            .core
            .create_breakpoint(TriggerType::IP, addr as u64, 2)
            .map_err(fatal)?;
        let id = trigger.id();
        // Drop the borrowed handle; the trigger is tracked by id from here on so
        // that the target does not become self-referential.
        drop(trigger);
        self.core.download_triggers();
        self.breakpoints.push((addr, id));
        Ok(true)
    }

    fn remove_hw_breakpoint(
        &mut self,
        addr: u32,
        _kind: usize,
    ) -> TargetResult<bool, Self> {
        let Some(pos) = self.breakpoints.iter().position(|(a, _)| *a == addr) else {
            return Ok(false);
        };
        let (_, id) = self.breakpoints.remove(pos);
        self.core.remove_breakpoint(id).map_err(fatal)?;
        self.core.download_triggers();
        Ok(true)
    }
}

/// The blocking event loop that polls the core and watches for GDB interrupts.
struct TricoreEventLoop<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> run_blocking::BlockingEventLoop for TricoreEventLoop<'a> {
    type Target = TricoreTarget<'a>;
    type Connection = TcpStream;
    type StopReason = SingleThreadStopReason<u32>;

    fn wait_for_stop_reason(
        target: &mut TricoreTarget<'a>,
        conn: &mut TcpStream,
    ) -> Result<
        run_blocking::Event<SingleThreadStopReason<u32>>,
        WaitForStopReasonError<anyhow::Error, std::io::Error>,
    > {
        loop {
            // Has GDB sent something (e.g. a Ctrl-C interrupt) while running?
            if conn
                .peek()
                .map_err(WaitForStopReasonError::Connection)?
                .is_some()
            {
                let byte = conn.read().map_err(WaitForStopReasonError::Connection)?;
                return Ok(run_blocking::Event::IncomingData(byte));
            }

            if let Some(stop_reason) = target.poll_stop() {
                return Ok(run_blocking::Event::TargetStopped(stop_reason));
            }

            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    fn on_interrupt(
        target: &mut TricoreTarget<'a>,
    ) -> Result<Option<SingleThreadStopReason<u32>>, anyhow::Error> {
        target.core.stop()?;
        Ok(Some(SingleThreadStopReason::Signal(Signal::SIGINT)))
    }
}

/// Opens core 0 of the given system and serves a single GDB session on the
/// given TCP port. Returns once the GDB client disconnects.
pub fn serve_gdb(system: System, port: u16) -> anyhow::Result<()> {
    let core = system
        .get_core(0)
        .context("Cannot open core 0 for GDB session")?;
    let mut target = TricoreTarget::new(core);

    let listener = TcpListener::bind(("0.0.0.0", port))
        .with_context(|| format!("Cannot bind GDB server to port {port}"))?;
    println!("GDB server listening on port {port}, waiting for a connection...");

    let (stream, peer) = listener
        .accept()
        .context("Failed to accept GDB connection")?;
    stream.set_nodelay(true).ok();
    log::info!("GDB client connected from {peer}");

    let gdb = GdbStub::new(stream);
    match gdb.run_blocking::<TricoreEventLoop>(&mut target) {
        Ok(reason) => log::info!("GDB session ended: {reason:?}"),
        Err(error) => return Err(anyhow::anyhow!("GDB session error: {error}")),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdbstub::arch::Registers;

    #[test]
    fn register_file_round_trips() {
        let mut regs = TricoreRegs::default();
        for i in 0..16 {
            regs.d[i] = 0xD000_0000 + i as u32;
            regs.a[i] = 0xA000_0000 + i as u32;
        }
        regs.pc = 0x8000_0000;
        regs.psw = 0x0000_0800;
        regs.pcxi = 0x0001_2345;

        let mut buffer = Vec::new();
        regs.gdb_serialize(|byte| buffer.push(byte.unwrap()));
        assert_eq!(buffer.len(), REG_BYTES);

        let mut decoded = TricoreRegs::default();
        decoded.gdb_deserialize(&buffer).unwrap();
        assert_eq!(regs, decoded);
    }
}
