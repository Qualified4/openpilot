//! CcncMap as a cereal Event (customReserved3). A BSD-only slot can have a zero vid.

use crate::input::live_tracks::{data_bytes, Message, EVENT_DISCRIMINANT_OFFSET};
use crate::slots::Slot;

const EVENT_WORDS: usize = 2;
const MAP_WORDS: usize = 3;
const SLOT_WORDS: usize = 3;
const SLOT_COUNT: usize = 5;
const EVENT_VALID_BIT: usize = 80;
const MAP_VALID_BIT: usize = 104;
const KIND_CONFIRMED_BIT: usize = 112;
const SLOT_PRESENT_BIT: usize = 113;
const KIND_WANTED_BIT: usize = 114;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapOut {
    pub radar_log_mono_time: u64,
    pub generation: u32,
    pub status: u8,
    pub ego_lane_coord: f32,
    pub slots: [Option<Slot>; SLOT_COUNT],
    pub valid: bool,
}

#[derive(Debug)]
pub struct Schema {
    pub event_discriminant: u16,
    pub event_pointer: usize,
    pub slots_pointer: usize,
    // Offsets in the field's own units, as reported by Cap'n Proto.
    pub map_fields: [usize; 5],
    pub slot_fields: [usize; 11],
    pub data_words: [usize; 3],
    pub pointer_words: [usize; 3],
}

pub const fn schema() -> Schema {
    Schema {
        event_discriminant: 108,
        event_pointer: 0,
        slots_pointer: 0,
        map_fields: [0, 2, 12, 4, MAP_VALID_BIT],
        slot_fields: [0, 1, 2, 12, 4, 5, 13, KIND_CONFIRMED_BIT, SLOT_PRESENT_BIT, 15, KIND_WANTED_BIT],
        data_words: [EVENT_WORDS, MAP_WORDS, SLOT_WORDS],
        pointer_words: [1, 1, 0],
    }
}

fn struct_tag(data: usize, pointers: usize) -> u64 {
    ((data as u64) << 32) | ((pointers as u64) << 48)
}

fn put<const N: usize>(bytes: &mut [u8], offset: usize, value: [u8; N]) {
    bytes[offset * N..(offset + 1) * N].copy_from_slice(&value);
}

/// The default Event timestamp is the input radar time. Live publication can supply its own clock.
pub fn encode(map: &MapOut) -> Vec<u8> {
    encode_at(map, map.radar_log_mono_time)
}

pub fn encode_at(map: &MapOut, log_mono_time: u64) -> Vec<u8> {
    // Single segment: root pointer, Event, map, then the inline-composite slot list.
    const HEADER_WORDS: usize = 2;
    const MAP_START: usize = HEADER_WORDS + EVENT_WORDS + 1;
    const LIST_START: usize = MAP_START + MAP_WORDS + 1;
    const WORDS: usize = LIST_START + 1 + SLOT_COUNT * SLOT_WORDS;
    let s = schema();
    let mut bytes = vec![0; WORDS * 8];
    put(&mut bytes, 1, ((WORDS - 1) as u32).to_le_bytes());
    put(&mut bytes, 1, struct_tag(EVENT_WORDS, 1).to_le_bytes());

    let event = &mut bytes[HEADER_WORDS * 8..MAP_START * 8];
    put(event, 0, log_mono_time.to_le_bytes());
    put(
        event,
        EVENT_DISCRIMINANT_OFFSET,
        s.event_discriminant.to_le_bytes(),
    );
    event[EVENT_VALID_BIT / 8] = u8::from(!map.valid) << (EVENT_VALID_BIT % 8);
    put(event, EVENT_WORDS, struct_tag(MAP_WORDS, 1).to_le_bytes());

    let [stamp, generation, status, ego, valid] = s.map_fields;
    let payload = &mut bytes[MAP_START * 8..LIST_START * 8];
    put(payload, stamp, map.radar_log_mono_time.to_le_bytes());
    put(payload, generation, map.generation.to_le_bytes());
    payload[status] = map.status;
    put(payload, ego, map.ego_lane_coord.to_le_bytes());
    payload[valid / 8] |= u8::from(map.valid) << (valid % 8);
    let list_pointer = 1u64 | (7 << 32) | ((SLOT_COUNT * SLOT_WORDS) as u64) << 35;
    put(payload, MAP_WORDS, list_pointer.to_le_bytes());

    let list_tag = struct_tag(SLOT_WORDS, 0) | ((SLOT_COUNT as u64) << 2);
    put(&mut bytes, LIST_START, list_tag.to_le_bytes());
    let [vid, distance, lateral, detect, lane, speed, kind, confirmed, present, vision, wanted] = s.slot_fields;
    for (i, slot) in map.slots.iter().enumerate() {
        let Some(slot) = slot else { continue };
        let start = (LIST_START + 1 + i * SLOT_WORDS) * 8;
        let dst = &mut bytes[start..start + SLOT_WORDS * 8];
        put(dst, vid, slot.vid.to_le_bytes());
        put(dst, distance, slot.distance.to_le_bytes());
        put(dst, lateral, slot.lateral.to_le_bytes());
        dst[detect] = slot.detect;
        put(dst, lane, slot.lane_coord_rel.to_le_bytes());
        put(dst, speed, slot.v_rel.to_le_bytes());
        dst[kind] = slot.kind;
        dst[confirmed / 8] |= u8::from(slot.kind_confirmed) << (confirmed % 8);
        dst[present / 8] |= 1 << (present % 8);
        dst[vision] = slot.vision;
        dst[wanted / 8] |= u8::from(slot.kind_wanted) << (wanted % 8);
    }
    bytes
}

fn bit(data: &[u8], offset: usize) -> bool {
    data.get(offset / 8)
        .is_some_and(|byte| byte & (1 << (offset % 8)) != 0)
}

/// Decode the framed Event, rejecting another union member or a malformed slot list.
pub fn decode(bytes: &[u8]) -> Option<MapOut> {
    let s = schema();
    let message = Message::new(bytes)?;
    let event = message.read_struct(0, 0)?;
    if u16::from_le_bytes(data_bytes(event.data, EVENT_DISCRIMINANT_OFFSET * 2))
        != s.event_discriminant
    {
        return None;
    }
    let map = message.struct_pointer(&event, s.event_pointer)?;
    let fields = message.struct_elements(&map, s.slots_pointer)?;
    if fields.len() != SLOT_COUNT {
        return None;
    }
    let [vid, distance, lateral, detect, lane, speed, kind, confirmed, present, vision, wanted] = s.slot_fields;
    let slots = std::array::from_fn(|i| {
        let data = fields[i].data;
        let f32_at = |offset| f32::from_le_bytes(data_bytes(data, offset * 4));
        let vid = u32::from_le_bytes(data_bytes(data, vid * 4));
        bit(data, present).then(|| Slot {
            vid,
            distance: f32_at(distance),
            lateral: f32_at(lateral),
            detect: data_bytes::<1>(data, detect)[0],
            lane_coord_rel: f32_at(lane),
            v_rel: f32_at(speed),
            kind: data_bytes::<1>(data, kind)[0],
            kind_confirmed: bit(data, confirmed),
            vision: data_bytes::<1>(data, vision)[0],
            kind_wanted: bit(data, wanted),
        })
    });
    let [stamp, generation, status, ego, valid] = s.map_fields;
    Some(MapOut {
        radar_log_mono_time: u64::from_le_bytes(data_bytes(map.data, stamp * 8)),
        generation: u32::from_le_bytes(data_bytes(map.data, generation * 4)),
        status: data_bytes::<1>(map.data, status)[0],
        ego_lane_coord: f32::from_le_bytes(data_bytes(map.data, ego * 4)),
        slots,
        valid: !bit(event.data, EVENT_VALID_BIT) && bit(map.data, valid),
    })
}
