//! A bounds-checked Cap'n Proto reader for the few cereal fields the recogniser reads, without a compiled schema.
//!
//! Same approach as `ccnc_core/src/live_tracks.rs`: the framing pycapnp / msgq write (segment table, then segments),
//! struct and list pointers, far pointers. Anything this does not read exactly (capabilities, other list forms) gives
//! None. The field positions live in `messages.rs` and are checked against `fixtures/schema_offsets.json`.

const MAX_SEGMENTS: usize = 511;

pub struct Message<'a> {
    segments: Vec<&'a [u8]>,
}

#[derive(Clone, Copy)]
pub struct Struct<'a> {
    pub data: &'a [u8],
    segment: usize,
    pointers: usize,
    pointer_count: usize,
}

impl Struct<'_> {
    fn empty() -> Self {
        Struct { data: &[], segment: 0, pointers: 0, pointer_count: 0 }
    }

    pub fn u16(&self, offset: usize, default: u16) -> u16 {
        u16::from_le_bytes(data_bytes(self.data, offset * 2)) ^ default
    }

    pub fn u32(&self, offset: usize, default: u32) -> u32 {
        u32::from_le_bytes(data_bytes(self.data, offset * 4)) ^ default
    }

    pub fn u64(&self, offset: usize, default: u64) -> u64 {
        u64::from_le_bytes(data_bytes(self.data, offset * 8)) ^ default
    }

    pub fn f32(&self, offset: usize) -> f32 {
        f32::from_bits(self.u32(offset, 0))
    }

    pub fn bool(&self, bit: usize, default: bool) -> bool {
        (self.data.get(bit / 8).copied().unwrap_or(0) >> (bit % 8) & 1 != 0) ^ default
    }
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at.checked_add(4)?)?.try_into().ok()?))
}

/// Fields past the end of the data section (an older schema) read as their defaults.
fn data_bytes<const N: usize>(data: &[u8], at: usize) -> [u8; N] {
    data.get(at..at + N).and_then(|b| b.try_into().ok()).unwrap_or([0; N])
}

/// The byte length of the first framed message in `bytes` (an rlog is such messages back to back).
pub fn framed_len(bytes: &[u8]) -> Option<usize> {
    let count = (u32_at(bytes, 0)? as usize).checked_add(1)?;
    if count > MAX_SEGMENTS {
        return None;
    }
    let mut end = (4 + 4 * count).div_ceil(8) * 8;
    for i in 0..count {
        end = end.checked_add((u32_at(bytes, 4 + 4 * i)? as usize).checked_mul(8)?)?;
    }
    (end <= bytes.len()).then_some(end)
}

impl<'a> Message<'a> {
    /// Exactly one framed message (msgq delivers one per recv).
    pub fn new(bytes: &'a [u8]) -> Option<Self> {
        let count = (u32_at(bytes, 0)? as usize).checked_add(1)?;
        if count > MAX_SEGMENTS {
            return None;
        }
        let header = (4 + 4 * count).div_ceil(8) * 8;
        let mut segments = Vec::with_capacity(count);
        let mut start = header;
        for i in 0..count {
            let words = u32_at(bytes, 4 + 4 * i)? as usize;
            let end = start.checked_add(words.checked_mul(8)?)?;
            segments.push(bytes.get(start..end)?);
            start = end;
        }
        (start == bytes.len()).then_some(Message { segments })
    }

    fn word(&self, segment: usize, index: usize) -> Option<u64> {
        let at = index.checked_mul(8)?;
        let bytes = self.segments.get(segment)?.get(at..at.checked_add(8)?)?;
        Some(u64::from_le_bytes(bytes.try_into().ok()?))
    }

    fn words(&self, segment: usize) -> usize {
        self.segments[segment].len() / 8
    }

    fn resolve(&self, segment: usize, index: usize) -> Option<Option<(usize, usize, u64)>> {
        let pointer = self.word(segment, index)?;
        if pointer == 0 {
            return Some(None);
        }
        let near = |segment: usize, index: usize, pointer: u64| -> Option<(usize, usize, u64)> {
            let offset = (pointer as u32 as i32 >> 2) as i64;
            let start = usize::try_from(index as i64 + 1 + offset).ok()?;
            Some((segment, start, pointer))
        };
        match pointer & 3 {
            0 | 1 => near(segment, index, pointer).map(Some),
            2 => {
                let target = (pointer >> 32) as usize;
                let pad = ((pointer as u32) >> 3) as usize;
                if pointer & 4 == 0 {
                    let landing = self.word(target, pad)?;
                    if landing & 3 > 1 || landing == 0 {
                        return None;
                    }
                    near(target, pad, landing).map(Some)
                } else {
                    let far = self.word(target, pad)?;
                    let tag = self.word(target, pad.checked_add(1)?)?;
                    if far & 7 != 2 || tag & 3 > 1 {
                        return None;
                    }
                    let segment = (far >> 32) as usize;
                    if segment >= self.segments.len() {
                        return None;
                    }
                    Some(Some((segment, ((far as u32) >> 3) as usize, tag)))
                }
            }
            _ => None,
        }
    }

    fn read_struct(&self, segment: usize, index: usize) -> Option<Struct<'a>> {
        let Some((segment, start, tag)) = self.resolve(segment, index)? else {
            return Some(Struct::empty());
        };
        if tag & 3 != 0 {
            return None;
        }
        let data_words = ((tag >> 32) & 0xffff) as usize;
        let pointer_count = (tag >> 48) as usize;
        let data_end = start.checked_add(data_words)?;
        if data_end.checked_add(pointer_count)? > self.words(segment) {
            return None;
        }
        Some(Struct { data: &self.segments[segment][start * 8..data_end * 8], segment, pointers: data_end, pointer_count })
    }

    pub fn root(&self) -> Option<Struct<'a>> {
        self.read_struct(0, 0)
    }

    pub fn struct_at(&self, s: &Struct<'a>, index: usize) -> Option<Struct<'a>> {
        if index >= s.pointer_count {
            return Some(Struct::empty());
        }
        self.read_struct(s.segment, s.pointers + index)
    }

    /// A struct list in the inline composite form.
    pub fn struct_list(&self, s: &Struct<'a>, index: usize) -> Option<Vec<Struct<'a>>> {
        if index >= s.pointer_count {
            return Some(Vec::new());
        }
        let Some((segment, start, tag)) = self.resolve(s.segment, s.pointers + index)? else {
            return Some(Vec::new());
        };
        if tag & 3 != 1 || (tag >> 32) & 7 != 7 {
            return None;
        }
        let words = (tag >> 35) as usize;
        let element_tag = self.word(segment, start)?;
        if start.checked_add(1)?.checked_add(words)? > self.words(segment) || element_tag & 3 != 0 {
            return None;
        }
        let count = ((element_tag as u32) >> 2) as usize;
        let data_words = ((element_tag >> 32) & 0xffff) as usize;
        let pointer_count = (element_tag >> 48) as usize;
        let step = data_words + pointer_count;
        if count.checked_mul(step)? > words || (step == 0 && count > 0) {
            return None;
        }
        let seg = self.segments[segment];
        Some((0..count).map(|i| {
            let at = start + 1 + i * step;
            Struct { data: &seg[at * 8..(at + data_words) * 8], segment, pointers: at + data_words, pointer_count }
        }).collect())
    }

    /// A `List(Float32)` in the four-byte element form.
    pub fn f32_list(&self, s: &Struct<'a>, index: usize) -> Option<Vec<f32>> {
        if index >= s.pointer_count {
            return Some(Vec::new());
        }
        let Some((segment, start, tag)) = self.resolve(s.segment, s.pointers + index)? else {
            return Some(Vec::new());
        };
        if tag & 3 != 1 || (tag >> 32) & 7 != 4 {
            return None;
        }
        let count = (tag >> 35) as usize;
        if start.checked_add((count * 4).div_ceil(8))? > self.words(segment) {
            return None;
        }
        let bytes = &self.segments[segment][start * 8..start * 8 + count * 4];
        Some(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }
}
