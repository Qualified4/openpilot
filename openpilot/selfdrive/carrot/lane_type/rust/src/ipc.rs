//! msgq (cereal) subscribe / publish and the VisionIPC road stream through `ipc/shim.cc` (feature `ipc`).
//!
//! The only FFI in the crate: msgq's shared-memory queues and VisionIPC's buffer passing are openpilot's own C++
//! transport, so they are linked from msgq_repo rather than re-implemented.

use std::ffi::{c_char, c_int, c_void, CString};

#[repr(C)]
struct SnFrame {
    data: *const u8,
    len: usize,
    width: usize,
    height: usize,
    stride: usize,
    uv_offset: usize,
    frame_id: u32,
    timestamp_eof: u64,
    buf: *mut c_void,
}

extern "C" {
    fn sn_sub_new(service: *const c_char, conflate: c_int) -> *mut c_void;
    fn sn_sub_poll(s: *mut c_void, timeout_ms: c_int) -> c_int;
    fn sn_sub_recv(s: *mut c_void, len: *mut usize) -> *const u8;
    fn sn_sub_free(s: *mut c_void);
    fn sn_pub_new(service: *const c_char) -> *mut c_void;
    fn sn_pub_send(p: *mut c_void, data: *const u8, len: usize) -> c_int;
    fn sn_pub_free(p: *mut c_void);
    fn sn_vipc_new(name: *const c_char, stream: c_int, conflate: c_int) -> *mut c_void;
    fn sn_vipc_connect(c: *mut c_void, blocking: c_int) -> c_int;
    fn sn_vipc_connected(c: *mut c_void) -> c_int;
    fn sn_vipc_recv(c: *mut c_void, timeout_ms: c_int, out: *mut SnFrame) -> c_int;
    fn sn_vipc_buf_frame_id(buf: *mut c_void) -> u64;
    fn sn_vipc_free(c: *mut c_void);
}

pub struct Sub(*mut c_void);

impl Sub {
    pub fn new(service: &str, conflate: bool) -> Option<Self> {
        let name = CString::new(service).ok()?;
        let p = unsafe { sn_sub_new(name.as_ptr(), conflate as c_int) };
        (!p.is_null()).then_some(Sub(p))
    }

    pub fn poll(&self, timeout_ms: i32) -> bool {
        unsafe { sn_sub_poll(self.0, timeout_ms) != 0 }
    }

    /// Every queued message (copied out), without waiting.
    pub fn drain(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        loop {
            let mut len = 0;
            let p = unsafe { sn_sub_recv(self.0, &mut len) };
            if p.is_null() {
                return out;
            }
            out.push(unsafe { std::slice::from_raw_parts(p, len) }.to_vec());
        }
    }
}

impl Drop for Sub {
    fn drop(&mut self) {
        unsafe { sn_sub_free(self.0) }
    }
}

pub struct Pub(*mut c_void);

impl Pub {
    pub fn new(service: &str) -> Option<Self> {
        let name = CString::new(service).ok()?;
        let p = unsafe { sn_pub_new(name.as_ptr()) };
        (!p.is_null()).then_some(Pub(p))
    }

    pub fn send(&self, data: &[u8]) -> bool {
        unsafe { sn_pub_send(self.0, data.as_ptr(), data.len()) >= 0 }
    }
}

impl Drop for Pub {
    fn drop(&mut self) {
        unsafe { sn_pub_free(self.0) }
    }
}

/// A received VisionIPC buffer: camerad's shared memory, valid while camerad has not reused the slot.
pub struct VipcFrame {
    pub data: *const u8,
    pub len: usize,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub uv_offset: usize,
    pub frame_id: u32,
    pub timestamp_eof: u64,
    buf: *mut c_void,
}

impl VipcFrame {
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    pub fn current_frame_id(&self) -> u32 {
        unsafe { sn_vipc_buf_frame_id(self.buf) as u32 }
    }
}

pub const VISION_STREAM_ROAD: i32 = 0;

pub struct Vipc(*mut c_void);

impl Vipc {
    pub fn new(name: &str, stream: i32, conflate: bool) -> Self {
        let name = CString::new(name).unwrap();
        Vipc(unsafe { sn_vipc_new(name.as_ptr(), stream, conflate as c_int) })
    }

    pub fn connect(&self, blocking: bool) -> bool {
        unsafe { sn_vipc_connect(self.0, blocking as c_int) != 0 }
    }

    pub fn connected(&self) -> bool {
        unsafe { sn_vipc_connected(self.0) != 0 }
    }

    pub fn recv(&self, timeout_ms: i32) -> Option<VipcFrame> {
        let mut f = SnFrame { data: std::ptr::null(), len: 0, width: 0, height: 0, stride: 0, uv_offset: 0, frame_id: 0, timestamp_eof: 0,
                              buf: std::ptr::null_mut() };
        (unsafe { sn_vipc_recv(self.0, timeout_ms, &mut f) } != 0).then(|| VipcFrame {
            data: f.data, len: f.len, width: f.width, height: f.height, stride: f.stride, uv_offset: f.uv_offset, frame_id: f.frame_id,
            timestamp_eof: f.timestamp_eof, buf: f.buf,
        })
    }
}

impl Drop for Vipc {
    fn drop(&mut self) {
        unsafe { sn_vipc_free(self.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// msgq round trip on this host (/dev/shm): a framed customReservedRawData2 event published and read back.
    #[test]
    fn msgq_round_trip() {
        // msgq's queues live under /dev/shm/msgq_<OPENPILOT_PREFIX> (openpilot's OpenpilotPrefix creates it).
        let prefix = format!("stripnet_test_{}", std::process::id());
        let dir = std::path::PathBuf::from(format!("/dev/shm/msgq_{prefix}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("OPENPILOT_PREFIX", &prefix);
        let publisher = Pub::new("customReservedRawData2").expect("pub");
        let sub = Sub::new("customReservedRawData2", false).expect("sub");
        let event = crate::messages::raw_data_event(crate::daemon::EVENT_CUSTOM_RAW_2, 42, b"{\"type\":\"ccncLaneMarking\"}");
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(publisher.send(&event));
        assert!(sub.poll(1000));
        let got = sub.drain();
        assert_eq!(got, vec![event]);
        drop((sub, publisher));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
