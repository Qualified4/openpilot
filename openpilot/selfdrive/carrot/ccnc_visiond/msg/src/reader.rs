//! A bounds-checked reader for the parts of a Cap'n Proto message this crate needs: structs, struct lists and f32 lists.
//! Anything else (a capability, another list form, bytes that do not frame a message) reads as None.

/// The segment count capnp's FlatArrayMessageReader takes (it rejects 512 or more).
const MAX_SEGMENTS: usize = 511;

pub struct Message<'a> {
    segments: Vec<&'a [u8]>,
}

/// A struct's data section and where its pointers are.
pub struct Struct<'a> {
    data: &'a [u8],
    segment: usize,
    pointers: usize,
    pointer_count: usize,
}

impl Struct<'_> {
    /// A null pointer reads as a struct with every field at its default.
    fn empty() -> Self {
        Struct { data: &[], segment: 0, pointers: 0, pointer_count: 0 }
    }

    /// Fields past the end of the data section (an older schema) read as zero.
    fn bytes<const N: usize>(&self, at: usize) -> [u8; N] {
        self.data.get(at..at + N).and_then(|b| b.try_into().ok()).unwrap_or([0; N])
    }

    pub fn u8_at(&self, at: usize) -> u8 {
        self.bytes::<1>(at)[0]
    }

    pub fn u16_at(&self, at: usize) -> u16 {
        u16::from_le_bytes(self.bytes(at))
    }

    pub fn u32_at(&self, at: usize) -> u32 {
        u32::from_le_bytes(self.bytes(at))
    }

    pub fn u64_at(&self, at: usize) -> u64 {
        u64::from_le_bytes(self.bytes(at))
    }

    pub fn f32_at(&self, at: usize) -> f32 {
        f32::from_le_bytes(self.bytes(at))
    }

    pub fn bit(&self, bit: usize) -> bool {
        self.u8_at(bit / 8) & (1 << (bit % 8)) != 0
    }
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at.checked_add(4)?)?.try_into().ok()?))
}

impl<'a> Message<'a> {
    /// The framing messaging sends: segment count - 1, each segment's size in words, padding to a word, the segments.
    pub fn new(bytes: &'a [u8]) -> Option<Self> {
        let count = (u32_at(bytes, 0)? as usize).checked_add(1)?;
        if count > MAX_SEGMENTS {
            return None;
        }
        let mut start = (4 + 4 * count).div_ceil(8) * 8;
        let mut segments = Vec::with_capacity(count);
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

    /// The pointer at (segment, index), past far pointers. None inside for null, else the segment and word where
    /// the object starts and the struct or list pointer word that describes it.
    fn resolve(&self, segment: usize, index: usize) -> Option<Option<(usize, usize, u64)>> {
        let pointer = self.word(segment, index)?;
        if pointer == 0 {
            return Some(None);
        }
        let near = |segment: usize, index: usize, pointer: u64| -> Option<(usize, usize, u64)> {
            // Signed 30-bit offset in words from the end of the pointer.
            let offset = i64::from(pointer as u32 as i32 >> 2);
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
                    // Two words: a far pointer to the object's start, then the tag that describes it.
                    let far = self.word(target, pad)?;
                    let tag = self.word(target, pad.checked_add(1)?)?;
                    let segment = (far >> 32) as usize;
                    if far & 7 != 2 || tag & 3 > 1 || segment >= self.segments.len() {
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

    /// Pointer `index` of a struct. One past its pointer section reads as null.
    pub fn struct_pointer(&self, parent: &Struct<'a>, index: usize) -> Option<Struct<'a>> {
        if index >= parent.pointer_count {
            return Some(Struct::empty());
        }
        self.read_struct(parent.segment, parent.pointers + index)
    }

    /// A List(Float32). Another element size gives None.
    pub fn f32_list(&self, parent: &Struct<'a>, index: usize) -> Option<Vec<f32>> {
        if index >= parent.pointer_count {
            return Some(Vec::new());
        }
        let Some((segment, start, tag)) = self.resolve(parent.segment, parent.pointers + index)? else {
            return Some(Vec::new());
        };
        if tag & 3 != 1 || (tag >> 32) & 7 != 4 {
            return None;
        }
        let count = (tag >> 35) as usize;
        let bytes = self.segments[segment].get(start.checked_mul(8)?..start.checked_mul(8)?.checked_add(count.checked_mul(4)?)?)?;
        Some(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }

    /// A list of structs in the inline composite form. Other list forms give None.
    pub fn struct_list(&self, parent: &Struct<'a>, index: usize) -> Option<Vec<Struct<'a>>> {
        if index >= parent.pointer_count {
            return Some(Vec::new());
        }
        let Some((segment, start, tag)) = self.resolve(parent.segment, parent.pointers + index)? else {
            return Some(Vec::new());
        };
        if tag & 3 != 1 || (tag >> 32) & 7 != 7 {
            return None;
        }
        let words = (tag >> 35) as usize;
        let element_tag = self.word(segment, start)?;
        let first = start.checked_add(1)?;
        if first.checked_add(words)? > self.words(segment) || element_tag & 3 != 0 {
            return None;
        }
        let count = ((element_tag as u32) >> 2) as usize;
        let data_words = ((element_tag >> 32) & 0xffff) as usize;
        let pointer_count = (element_tag >> 48) as usize;
        let size = data_words + pointer_count;
        if count.checked_mul(size)? > words {
            return None;
        }
        Some(
            (0..count)
                .map(|i| {
                    let at = first + i * size;
                    Struct {
                        data: &self.segments[segment][at * 8..(at + data_words) * 8],
                        segment,
                        pointers: at + data_words,
                        pointer_count,
                    }
                })
                .collect(),
        )
    }
}
