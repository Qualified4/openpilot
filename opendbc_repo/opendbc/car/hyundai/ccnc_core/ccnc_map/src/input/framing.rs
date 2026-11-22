//! Stream one framed cereal Event at a time from an uncompressed rlog or event cache.

use std::io::{self, BufRead};

const MAX_SEGMENTS: usize = 511;
const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

pub fn next(reader: &mut impl BufRead, frame: &mut Vec<u8>) -> io::Result<bool> {
    if reader.fill_buf()?.is_empty() {
        return Ok(false);
    }
    frame.resize(4, 0);
    reader.read_exact(frame)?;
    let segments = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize + 1;
    let bad_frame = || io::Error::new(io::ErrorKind::InvalidData, "invalid Cap'n Proto frame");
    if segments > MAX_SEGMENTS {
        return Err(bad_frame());
    }
    let header = (4 + segments * 4).next_multiple_of(8);
    frame.resize(header, 0);
    reader.read_exact(&mut frame[4..])?;
    let bytes = frame[4..4 + segments * 4]
        .chunks_exact(4)
        .try_fold(header, |total, word| {
            let words = u32::from_le_bytes(word.try_into().unwrap()) as usize;
            total.checked_add(words.checked_mul(8)?)
        })
        .filter(|&bytes| bytes <= MAX_FRAME_BYTES)
        .ok_or_else(bad_frame)?;
    frame.resize(bytes, 0);
    reader.read_exact(&mut frame[header..])?;
    Ok(true)
}
