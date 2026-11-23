//! Linux 64-bit VisionIPC client and the fixed-size MSGQ packet layout of msgq_repo.

use crate::nv12::Layout;
use anyhow::{bail, ensure, Context, Result};
use std::{
    fs::File,
    io::{self, Seek, SeekFrom},
    mem::{size_of, zeroed},
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    path::PathBuf,
    ptr::{self, NonNull},
    sync::atomic::{fence, AtomicU64, Ordering},
    time::{Duration, Instant},
};

const MAX_BUFS: usize = 128;
const BUF_DESC: usize = 112;
const PACKET: usize = 48;
const SLOT: usize = 8 + PACKET;
const QUEUE_HEADER: usize = (3 + 3 * 40) * 8;
const QUEUE_SIZE: usize = 1024 * 1024;
const NO_FRAME_TIMEOUT: Duration = Duration::from_secs(1);
const RETRY: Duration = Duration::from_millis(50);

#[derive(Clone, Copy)]
pub enum Stream {
    Road = 0,
    Wide = 2,
}

impl Stream {
    pub fn camera(self) -> &'static str {
        match self {
            Self::Road => "road",
            Self::Wide => "wideRoad",
        }
    }
}

pub struct Frame {
    pub frame_id: u32,
    pub timestamp_sof: u64,
    pub layout: Layout,
    pub nv12: Vec<u8>,
}

pub struct Client {
    stream: Stream,
    socket_path: PathBuf,
    queue_path: PathBuf,
    connection: Option<Connection>,
    waiting: bool,
    retry_at: Option<Instant>,
}

impl Client {
    pub fn new(stream: Stream, server: &str) -> Self {
        let prefix = std::env::var("OPENPILOT_PREFIX").ok();
        let socket_prefix = prefix.as_ref().map(|s| format!("{s}_")).unwrap_or_default();
        let queue_prefix = prefix.as_ref().map(|s| format!("{s}/")).unwrap_or_default();
        Self {
            stream,
            socket_path: format!("/tmp/{socket_prefix}visionipc_{server}").into(),
            queue_path: format!(
                "/dev/shm/msgq_{queue_prefix}visionipc_{server}_{}",
                stream as u32
            )
            .into(),
            connection: None,
            waiting: false,
            retry_at: None,
        }
    }

    /// The newest frame not returned before. None when there is none yet or the server is away.
    /// It reconnects by itself, so call again after a short sleep.
    pub fn poll(&mut self) -> Option<Frame> {
        if self.connection.is_none() {
            if self.retry_at.is_some_and(|at| Instant::now() < at) {
                return None;
            }
            match Connection::open(self) {
                Ok(conn) => {
                    eprintln!("VisionIPC connected {}", self.stream.camera());
                    self.connection = Some(conn);
                    self.waiting = false;
                }
                Err(err) => {
                    if !self.waiting {
                        eprintln!("VisionIPC waiting for {} ({err:#})", self.stream.camera());
                        self.waiting = true;
                    }
                    self.retry_at = Some(Instant::now() + RETRY);
                    return None;
                }
            }
        }
        let conn = self.connection.as_mut().unwrap();
        match conn.latest() {
            Ok(Some(frame)) => return Some(frame),
            Ok(None) if conn.last_packet.elapsed() < NO_FRAME_TIMEOUT => {}
            Ok(None) => self.connection = None,
            Err(err) => {
                eprintln!("VisionIPC reconnecting ({err})");
                self.connection = None;
            }
        }
        None
    }

    pub fn connected(&self) -> bool {
        self.connection.is_some()
    }

    /// Forget the frames published so far, so `poll` only returns later ones.
    pub fn discard(&mut self) {
        if let Some(conn) = &mut self.connection {
            conn.cursor = conn.ring.load_u64(8);
            conn.last_packet = Instant::now();
        }
    }
}

struct Connection {
    ring: Mapping,
    buffers: Vec<Buffer>,
    server_id: u64,
    publisher: u64,
    cursor: u64,
    last_packet: Instant,
}

impl Connection {
    fn open(client: &Client) -> Result<Self> {
        let (server_id, buffers) = get_buffers(client).context("camera buffer handshake")?;
        let ring = Mapping::open(File::open(&client.queue_path)?, QUEUE_HEADER + QUEUE_SIZE)?;
        let publisher = ring.load_u64(16);
        let cursor = ring.load_u64(8);
        Ok(Self {
            ring,
            buffers,
            server_id,
            publisher,
            cursor,
            last_packet: Instant::now(),
        })
    }

    fn latest(&mut self) -> Result<Option<Frame>> {
        ensure!(
            self.ring.load_u64(16) == self.publisher,
            "MSGQ publisher changed"
        );
        let write = self.ring.load_u64(8);
        let end = write as u32 as usize;
        // The writer briefly publishes a zero offset before committing the first packet after wrap.
        if write == self.cursor || end == 0 {
            return Ok(None);
        }
        ensure!(
            end >= SLOT && end <= QUEUE_SIZE && end % SLOT == 0,
            "invalid MSGQ write offset"
        );
        let start = QUEUE_HEADER + end - SLOT;
        let mut packet = [0; SLOT];
        self.ring.copy(start, &mut packet);
        fence(Ordering::SeqCst);
        if self.ring.load_u64(8) != write {
            return Ok(None);
        }
        ensure!(
            self.ring.load_u64(16) == self.publisher,
            "MSGQ publisher changed"
        );
        self.cursor = write;
        ensure!(
            u64_at(&packet, 0) == PACKET as u64,
            "invalid VisionIPC MSGQ packet length"
        );
        ensure!(
            u64_at(&packet, 8) == self.server_id,
            "VisionIPC server changed"
        );
        let idx = usize_at(&packet, 16);
        let buf = self
            .buffers
            .get(idx)
            .context("invalid VisionIPC buffer index")?;
        let frame_id = u32::from_ne_bytes(packet[24..28].try_into().unwrap());
        let timestamp_sof = u64_at(&packet, 32);
        self.last_packet = Instant::now();

        // Copy out before inference. Reject a buffer whose producer has already recycled it.
        if buf.frame_id() != u64::from(frame_id) {
            return Ok(None);
        }
        let mut nv12 = vec![0; buf.len];
        buf.map.copy(0, &mut nv12);
        fence(Ordering::SeqCst);
        if buf.frame_id() != u64::from(frame_id) || self.ring.load_u64(8) != write {
            // A newer frame is available too. Re-sample the queue before handing anything off.
            return Ok(None);
        }
        ensure!(
            self.ring.load_u64(16) == self.publisher,
            "MSGQ publisher changed"
        );
        Ok(Some(Frame {
            frame_id,
            timestamp_sof,
            layout: buf.layout,
            nv12,
        }))
    }
}

struct Buffer {
    map: Mapping,
    len: usize,
    layout: Layout,
}

impl Buffer {
    fn frame_id(&self) -> u64 {
        if self.len % 8 == 0 {
            return self.map.load_u64(self.len);
        }
        let mut id = [0; 8];
        self.map.copy(self.len, &mut id);
        u64::from_ne_bytes(id)
    }
}

struct Mapping {
    addr: NonNull<u8>,
    len: usize,
}

impl Mapping {
    fn open(mut file: File, len: usize) -> Result<Self> {
        let mut file_len = file.metadata()?.len();
        if file_len == 0 {
            // On-device VisionBuf uses ION DMA-BUF fds. Their fstat size is zero.
            file_len = file.seek(SeekFrom::End(0))?;
        }
        ensure!(
            len > 0 && len <= isize::MAX as usize && file_len >= len as u64,
            "invalid VisionIPC mapping length"
        );
        // The file remains alive via mmap even after its fd or the producer closes.
        let p = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error().into());
        }
        Ok(Self {
            addr: NonNull::new(p.cast()).unwrap(),
            len,
        })
    }

    fn load_u64(&self, offset: usize) -> u64 {
        assert!(offset % 8 == 0 && offset + 8 <= self.len);
        // 64-bit relaxed loads are supported on read-only memory on x86_64 and aarch64.
        // Acquire separately before copying data published by this word.
        let value =
            unsafe { &*self.addr.as_ptr().add(offset).cast::<AtomicU64>() }.load(Ordering::Relaxed);
        fence(Ordering::Acquire);
        value
    }

    fn copy(&self, offset: usize, out: &mut [u8]) {
        assert!(offset <= self.len && out.len() <= self.len - offset);
        // Memory is modified by the producer process. Readers verify the cursor before using it.
        unsafe {
            ptr::copy_nonoverlapping(self.addr.as_ptr().add(offset), out.as_mut_ptr(), out.len())
        }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.addr.as_ptr().cast(), self.len) };
    }
}

fn checked(code: libc::c_int) -> io::Result<libc::c_int> {
    if code < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(code)
    }
}

fn get_buffers(client: &Client) -> Result<(u64, Vec<Buffer>)> {
    let socket = unsafe {
        OwnedFd::from_raw_fd(checked(libc::socket(
            libc::AF_UNIX,
            libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
            0,
        ))?)
    };
    let fd = socket.as_raw_fd();
    let path = client.socket_path.as_os_str().as_encoded_bytes();
    let mut addr: libc::sockaddr_un = unsafe { zeroed() };
    ensure!(
        path.len() < addr.sun_path.len() && !path.contains(&0),
        "invalid VisionIPC socket path"
    );
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (dst, &byte) in addr.sun_path.iter_mut().zip(path) {
        *dst = byte as libc::c_char;
    }
    unsafe {
        checked(libc::connect(
            fd,
            (&addr as *const libc::sockaddr_un).cast(),
            size_of::<libc::sockaddr_un>() as _,
        ))?;
    }
    let request = (client.stream as u32).to_ne_bytes();
    let sent = unsafe {
        libc::send(
            fd,
            request.as_ptr().cast(),
            request.len(),
            libc::MSG_NOSIGNAL,
        )
    };
    ensure!(
        sent == request.len() as isize,
        "failed to send VisionIPC stream request"
    );
    let mut poll = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    ensure!(
        checked(unsafe { libc::poll(&mut poll, 1, 500) })? > 0,
        "VisionIPC handshake timed out"
    );
    let mut bytes = [0; BUF_DESC * MAX_BUFS];
    let (len, fds) = recv_fds(fd, &mut bytes)?;
    ensure!(
        !fds.is_empty() && len == fds.len() * BUF_DESC,
        "invalid VisionIPC buffer response ({len} bytes, {} fds)",
        fds.len()
    );
    // VisionBuf has native pointers for both Y and UV between uv_offset and server_id.
    let server_id = u64_at(&bytes, 88);
    let mut buffers = Vec::with_capacity(fds.len());
    for (idx, (desc, fd)) in bytes[..len].chunks_exact(BUF_DESC).zip(fds).enumerate() {
        let size = usize_at(desc, 0);
        let mmap_len = usize_at(desc, 8);
        let layout = Layout {
            width: usize_at(desc, 40),
            height: usize_at(desc, 48),
            stride: usize_at(desc, 56),
            uv_offset: usize_at(desc, 64),
        };
        ensure!(
            size.checked_add(8) == Some(mmap_len),
            "invalid VisionIPC buffer size"
        );
        ensure!(
            u64_at(desc, 88) == server_id
                && usize_at(desc, 96) == idx
                && u32::from_ne_bytes(desc[104..108].try_into().unwrap()) == client.stream as u32,
            "invalid VisionIPC buffer descriptor"
        );
        layout.validate(size)?;
        let map = Mapping::open(File::from(fd), mmap_len)?;
        buffers.push(Buffer {
            map,
            len: size,
            layout,
        });
    }
    Ok((server_id, buffers))
}

fn recv_fds(fd: libc::c_int, bytes: &mut [u8]) -> Result<(usize, Vec<OwnedFd>)> {
    let mut iov = libc::iovec {
        iov_base: bytes.as_mut_ptr().cast(),
        iov_len: bytes.len(),
    };
    // Word alignment for cmsghdr. Space for up to MAX_BUFS ints plus the header.
    let mut control = [0u64; MAX_BUFS / 2 + 4];
    let mut msg: libc::msghdr = unsafe { zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = size_of::<[u64; MAX_BUFS / 2 + 4]>();
    let len = unsafe { libc::recvmsg(fd, &mut msg, libc::MSG_CMSG_CLOEXEC) };
    if len < 0 {
        return Err(io::Error::last_os_error().into());
    }
    let mut fds = vec![];
    let mut header = unsafe { libc::CMSG_FIRSTHDR(&msg) };
    while !header.is_null() {
        unsafe {
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let n =
                    ((*header).cmsg_len - libc::CMSG_LEN(0) as usize) / size_of::<libc::c_int>();
                for i in 0..n {
                    fds.push(OwnedFd::from_raw_fd(
                        libc::CMSG_DATA(header).cast::<libc::c_int>().add(i).read(),
                    ));
                }
            }
            header = libc::CMSG_NXTHDR(&msg, header);
        }
    }
    if msg.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) != 0 {
        bail!("truncated VisionIPC buffer response");
    }
    Ok((len as usize, fds))
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_ne_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn usize_at(bytes: &[u8], offset: usize) -> usize {
    u64_at(bytes, offset) as usize
}
