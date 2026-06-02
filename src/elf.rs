//! Hosts utilities to work with elf files.

use anyhow::Context;
use elf::abi::PT_LOAD;
use elf::endian::AnyEndian;
use elf::ElfBytes;
use ihex::Record;

/// Interprets the given data as an ELF file and returns it in Intel HEX format.
pub fn elf_to_hex(data: &[u8]) -> anyhow::Result<String> {
    if cfg!(feature = "in_docker") {
        return std::fs::read_to_string("C:\\output.hex")
            .context("Docker: Cannot read resulting hex file");
    }

    let elf_file =
        ElfBytes::<AnyEndian>::minimal_parse(data).context("Cannot parse ELF file")?;

    let mut records: Vec<Record> = Vec::new();
    let mut current_upper: Option<u16> = None;

    if let Some(segments) = elf_file.segments() {
        for phdr in segments {
            if phdr.p_type != PT_LOAD || phdr.p_filesz == 0 {
                continue;
            }

            let seg_data = elf_file
                .segment_data(&phdr)
                .context("Cannot read ELF segment data")?;

            let mut offset = 0usize;
            while offset < seg_data.len() {
                let addr = phdr.p_paddr as u32 + offset as u32;
                let upper = (addr >> 16) as u16;
                let lower = (addr & 0xFFFF) as u16;

                if current_upper != Some(upper) {
                    records.push(Record::ExtendedLinearAddress(upper));
                    current_upper = Some(upper);
                }

                // Cap chunk at 16 bytes and never cross a 64 KB boundary.
                let remaining_in_64k = (0x10000u32 - (addr & 0xFFFF)) as usize;
                let chunk_size = remaining_in_64k.min(16).min(seg_data.len() - offset);

                records.push(Record::Data {
                    offset: lower,
                    value: seg_data[offset..offset + chunk_size].to_vec(),
                });

                offset += chunk_size;
            }
        }
    }

    records.push(Record::EndOfFile);

    ihex::create_object_file_representation(&records)
        .context("Cannot create Intel HEX representation")
}
