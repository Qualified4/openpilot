//! MNN 3.6.2 OpenCL FP16 (Image2D) backend via thin C-ABI FFI (`libmnn_capi.so`, loaded dynamically with `dlopen`).
//!
//! Models (fixed shapes for zero-overhead Adreno 630v2 Image2D OpenCL caching):
//! * Front detector: `yolox_s_416x192_fp16.mnn` (`1x3x192x416`, `decode_in_inference=True`)
//! * Side detector:  `yolox_s_416x320_fp16.mnn` (`1x3x320x416`, `decode_in_inference=True`)
//! * Raw detector:   `yolox_s_416_fp16.mnn`     (`1x3x416x416`, `decode_in_inference=True`)
//! * Classifier:     `kind1_fp16.mnn` (`1x3x128x128`) and `kind4_fp16.mnn` (`4x3x128x128`)

use crate::engine::Engine;
use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub const MODEL_FRONT: &str = "yolox_s_416x192_fp16.mnn";
pub const MODEL_SIDE: &str = "yolox_s_416x320_fp16.mnn";
pub const MODEL_RAW: &str = "yolox_s_416_fp16.mnn";
pub const MODEL_KIND1: &str = "kind1_fp16.mnn";
pub const MODEL_KIND4: &str = "kind4_fp16.mnn";

pub const DETECTOR_MODELS: [&str; 3] = [MODEL_FRONT, MODEL_SIDE, MODEL_RAW];
pub const CLASSIFIER_MODELS: [&str; 2] = [MODEL_KIND1, MODEL_KIND4];
pub const ALL_MODELS: [&str; 5] = [MODEL_FRONT, MODEL_SIDE, MODEL_RAW, MODEL_KIND1, MODEL_KIND4];

pub const DEFAULT_CACHE_DIR: &str = "/data/cache/ccnc_visiond_mnn";
pub const CACHE_MANIFEST: &str = "cache.manifest";
pub const CAPI_LIB_NAME: &str = "libmnn_capi.so";
pub const CL_LIB_NAME: &str = "libMNN_CL.so";
const RUNTIME_LIBS: [&str; 3] = ["libMNN.so", CL_LIB_NAME, CAPI_LIB_NAME];
const OPENCL_KEY: &str = "opencl_driver";
const OPENCL_LIB_PATHS: [&str; 3] = [
    "/usr/lib/libOpenCL.so",
    "/usr/lib/aarch64-linux-gnu/libOpenCL.so",
    "/vendor/lib64/libOpenCL.so",
];

pub const BACKEND_CPU: i32 = 0;
pub const BACKEND_OPENCL: i32 = 3;
pub const PRECISION_LOW: i32 = 2;
/// `MNN_GPU_TUNING_NORMAL (8) | MNN_GPU_MEMORY_IMAGE (128)` — best latency & 0 DM drops on Adreno 630v2.
pub const GPU_MODE_DEFAULT: i32 = 136;

#[repr(C)]
struct MnnSessionHandle {
    _private: [u8; 0],
}

#[repr(C)]
struct MnnCreateOptions {
    backend: i32,
    precision: i32,
    num_threads: i32,
    gpu_mode: i32,
    allow_cpu_backup: i32,
    cache_file: *const c_char,
}

type FnCreate = unsafe extern "C" fn(*const c_char, *const MnnCreateOptions, *mut c_char, usize) -> *mut MnnSessionHandle;
type FnActiveBackend = unsafe extern "C" fn(*const MnnSessionHandle) -> i32;
type FnShape = unsafe extern "C" fn(*const MnnSessionHandle, *mut i32, i32) -> i32;
type FnRun = unsafe extern "C" fn(*mut MnnSessionHandle, *const f32, usize, *mut f32, usize, *mut usize, *mut c_char, usize) -> i32;
type FnUpdateCache = unsafe extern "C" fn(*mut MnnSessionHandle, *mut c_char, usize) -> i32;
type FnDestroy = unsafe extern "C" fn(*mut MnnSessionHandle);

struct MnnApi {
    _lib: *mut c_void,
    create: FnCreate,
    active_backend: FnActiveBackend,
    input_shape: FnShape,
    output_shape: FnShape,
    run: FnRun,
    update_cache: FnUpdateCache,
    destroy: FnDestroy,
}

// Safety: dlopen handle and resolved function pointers are immutable and thread-safe to read.
unsafe impl Send for MnnApi {}
unsafe impl Sync for MnnApi {}

fn dlerror_string() -> String {
    unsafe {
        let p = libc::dlerror();
        if p.is_null() {
            "unknown dlerror".into()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

unsafe fn load_sym<T>(lib: *mut c_void, name: &str) -> Result<T> {
    let cname = CString::new(name)?;
    let sym = libc::dlsym(lib, cname.as_ptr());
    if sym.is_null() {
        bail!("dlsym({name}) failed: {}", dlerror_string());
    }
    Ok(std::mem::transmute_copy(&sym))
}

impl MnnApi {
    fn candidate_paths(assets: &Path) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        if let Some(p) = std::env::var_os("MNN_CAPI_PATH") {
            if !p.is_empty() {
                paths.push(PathBuf::from(p));
                return paths;
            }
        }
        paths.push(assets.join(CAPI_LIB_NAME));
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                paths.push(dir.join(CAPI_LIB_NAME));
            }
        }
        paths.push(PathBuf::from(CAPI_LIB_NAME));
        paths
    }

    fn cl_candidate_paths(assets: &Path) -> Vec<PathBuf> {
        let mut paths = vec![assets.join(CL_LIB_NAME)];
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                paths.push(dir.join(CL_LIB_NAME));
            }
        }
        paths.push(PathBuf::from(CL_LIB_NAME));
        paths
    }

    /// With `MNN_SEP_BUILD=ON`, the OpenCL backend (`type=3`) is registered by `libMNN_CL.so`'s
    /// static constructor when loaded into the process. Missing `libMNN_CL.so` is non-fatal so
    /// host CPU tests and automatic CPU ORT fallback continue to work cleanly.
    fn try_load_opencl_plugin(assets: &Path) -> *mut c_void {
        for cand in Self::cl_candidate_paths(assets) {
            let s = cand.to_string_lossy();
            let Ok(cpath) = CString::new(s.as_ref()) else { continue };
            let handle = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL | libc::RTLD_NODELETE) };
            if !handle.is_null() {
                return handle;
            }
        }
        std::ptr::null_mut()
    }

    fn load(assets: &Path) -> Result<Arc<Self>> {
        let mut errors = Vec::new();
        for cand in Self::candidate_paths(assets) {
            let s = cand.to_string_lossy().into_owned();
            let cpath = CString::new(s.as_str())?;
            // RTLD_GLOBAL exposes libMNN.so symbols to libMNN_CL.so; RTLD_NODELETE avoids tearing down OpenCL driver state on drop.
            let lib = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL | libc::RTLD_NODELETE) };
            if lib.is_null() {
                errors.push(format!("{s}: {}", dlerror_string()));
                continue;
            }
            let _cl = Self::try_load_opencl_plugin(assets);
            let api = unsafe {
                MnnApi {
                    _lib: lib,
                    create: load_sym(lib, "mnn_session_create")?,
                    active_backend: load_sym(lib, "mnn_session_active_backend")?,
                    input_shape: load_sym(lib, "mnn_session_input_shape")?,
                    output_shape: load_sym(lib, "mnn_session_output_shape")?,
                    run: load_sym(lib, "mnn_session_run")?,
                    update_cache: load_sym(lib, "mnn_session_update_cache")?,
                    destroy: load_sym(lib, "mnn_session_destroy")?,
                }
            };
            return Ok(Arc::new(api));
        }
        bail!("cannot load {CAPI_LIB_NAME}: {}", errors.join("; "))
    }
}

fn c_err(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

pub struct MnnSession {
    api: Arc<MnnApi>,
    handle: *mut MnnSessionHandle,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
    out_elems: usize,
    active_backend: i32,
}

impl Drop for MnnSession {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.api.destroy)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    OpenClFp16,
    Cpu,
}

impl MnnSession {
    fn open(api: Arc<MnnApi>, model_path: &Path, backend: Backend, threads: usize, cache_file: Option<&Path>) -> Result<Self> {
        let model_str = model_path.to_string_lossy();
        if !model_path.is_file() {
            bail!("missing MNN model {model_str}");
        }
        let c_model = CString::new(model_str.as_ref())?;
        let c_cache = match cache_file {
            Some(p) => Some(CString::new(p.to_string_lossy().as_ref())?),
            None => None,
        };
        let opts = MnnCreateOptions {
            backend: match backend {
                Backend::OpenClFp16 => BACKEND_OPENCL,
                Backend::Cpu => BACKEND_CPU,
            },
            precision: PRECISION_LOW,
            num_threads: threads.clamp(1, 4) as i32,
            gpu_mode: GPU_MODE_DEFAULT,
            allow_cpu_backup: 0,
            cache_file: c_cache.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        };
        let mut err_buf = [0 as c_char; 512];
        let handle = unsafe { (api.create)(c_model.as_ptr(), &opts, err_buf.as_mut_ptr(), err_buf.len()) };
        if handle.is_null() {
            bail!("mnn_session_create({model_str}): {}", c_err(&err_buf));
        }

        let active_backend = unsafe { (api.active_backend)(handle) };
        let mut dims = [0i32; 8];
        let in_rank = unsafe { (api.input_shape)(handle, dims.as_mut_ptr(), dims.len() as i32) };
        if in_rank <= 0 {
            unsafe { (api.destroy)(handle) };
            bail!("mnn_session_input_shape failed for {model_str}");
        }
        let in_shape: Vec<usize> = dims[..in_rank as usize].iter().map(|&d| d as usize).collect();

        let out_rank = unsafe { (api.output_shape)(handle, dims.as_mut_ptr(), dims.len() as i32) };
        if out_rank <= 0 {
            unsafe { (api.destroy)(handle) };
            bail!("mnn_session_output_shape failed for {model_str}");
        }
        let out_shape: Vec<usize> = dims[..out_rank as usize].iter().map(|&d| d as usize).collect();
        let out_elems: usize = out_shape.iter().product();

        Ok(Self { api, handle, in_shape, out_shape, out_elems, active_backend })
    }

    pub fn run_slice(&mut self, input: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
        let mut out = vec![0.0f32; self.out_elems];
        let mut written = 0usize;
        let mut err_buf = [0 as c_char; 512];
        let rc = unsafe {
            (self.api.run)(
                self.handle,
                input.as_ptr(),
                input.len(),
                out.as_mut_ptr(),
                out.len(),
                &mut written,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc != 0 {
            bail!("mnn_session_run failed ({rc}): {}", c_err(&err_buf));
        }
        out.truncate(written);
        Ok((self.out_shape.clone(), out))
    }

    pub fn update_cache(&mut self) -> Result<()> {
        let mut err_buf = [0 as c_char; 512];
        let rc = unsafe { (self.api.update_cache)(self.handle, err_buf.as_mut_ptr(), err_buf.len()) };
        if rc != 0 {
            bail!("mnn_session_update_cache failed ({rc}): {}", c_err(&err_buf));
        }
        Ok(())
    }

    pub fn in_shape(&self) -> &[usize] {
        &self.in_shape
    }

    pub fn out_shape(&self) -> &[usize] {
        &self.out_shape
    }

    pub fn active_backend(&self) -> i32 {
        self.active_backend
    }
}

pub fn cache_dir() -> PathBuf {
    std::env::var("CCNC_VISIOND_MNN_CACHE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CACHE_DIR))
}

pub fn cache_file_for(dir: &Path, model_name: &str) -> PathBuf {
    let stem = Path::new(model_name).file_stem().and_then(|s| s.to_str()).unwrap_or(model_name);
    dir.join(format!("{stem}.clcache"))
}

pub fn missing_caches(dir: &Path, models: &[&str]) -> Vec<PathBuf> {
    models
        .iter()
        .map(|m| cache_file_for(dir, m))
        .filter(|p| !p.metadata().map(|m| m.is_file() && m.len() > 0).unwrap_or(false))
        .collect()
}

fn allow_cold_tuning() -> bool {
    std::env::var("CCNC_VISIOND_MNN_ALLOW_COLD").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {} for sha256", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{b:02x}");
    }
    Ok(hex)
}

fn opencl_driver_fingerprint() -> String {
    if let Some(p) = std::env::var_os("CCNC_VISIOND_OPENCL_LIB").filter(|s| !s.is_empty()) {
        let path = PathBuf::from(p);
        if let Ok(h) = sha256_file(&path) {
            return h;
        }
    }
    for cand in OPENCL_LIB_PATHS {
        let path = Path::new(cand);
        if let Ok(h) = sha256_file(path) {
            return h;
        }
    }
    "none".to_string()
}

pub fn compute_cache_manifest(assets: &Path, models: &[&str]) -> Result<BTreeMap<String, String>> {
    let mut map = BTreeMap::new();
    for &model in models {
        map.insert(model.to_string(), sha256_file(&assets.join(model))?);
    }
    for lib in RUNTIME_LIBS {
        let p = assets.join(lib);
        if p.is_file() {
            map.insert(lib.to_string(), sha256_file(&p)?);
        }
    }
    map.insert(OPENCL_KEY.to_string(), opencl_driver_fingerprint());
    Ok(map)
}

pub fn write_cache_manifest(assets: &Path, dir: &Path, models: &[&str]) -> Result<()> {
    let entries = compute_cache_manifest(assets, models)?;
    let mut text = String::from("# ccnc_visiond mnn opencl cache manifest v1\n");
    for (key, hash) in &entries {
        text.push_str(hash);
        text.push_str("  ");
        text.push_str(key);
        text.push('\n');
    }
    let tmp = dir.join(format!("{CACHE_MANIFEST}.tmp"));
    let dst = dir.join(CACHE_MANIFEST);
    std::fs::write(&tmp, text).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, &dst).with_context(|| format!("rename {} -> {}", tmp.display(), dst.display()))?;
    Ok(())
}

fn read_cache_manifest(dir: &Path) -> Result<BTreeMap<String, String>> {
    let path = dir.join(CACHE_MANIFEST);
    let text = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "OpenCL cache manifest missing in {} ({}); run 'ccnc_visiond --warmup-cache' offroad",
            dir.display(),
            CACHE_MANIFEST
        )
    })?;
    let mut map = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        if let (Some(hash), Some(key)) = (parts.next(), parts.next()) {
            map.insert(key.to_string(), hash.to_string());
        }
    }
    Ok(map)
}

fn verify_cache_ready(assets: &Path, dir: &Path, models: &[&str], backend: Backend) -> Result<()> {
    if backend != Backend::OpenClFp16 || allow_cold_tuning() {
        return Ok(());
    }
    let missing = missing_caches(dir, models);
    if !missing.is_empty() {
        let names: Vec<String> = missing
            .iter()
            .map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())
            .collect();
        bail!(
            "OpenCL cache missing in {} ({}); run 'ccnc_visiond --warmup-cache' offroad to avoid DM frame drops",
            dir.display(),
            names.join(", ")
        );
    }
    let cached = read_cache_manifest(dir)?;
    let expected = compute_cache_manifest(assets, models)?;
    for (key, cur_hash) in &expected {
        match cached.get(key) {
            Some(old_hash) if old_hash == cur_hash => {}
            Some(old_hash) => {
                bail!(
                    "OpenCL cache hash mismatch for {key} (cached={old_hash}, current={cur_hash}); run 'ccnc_visiond --warmup-cache' offroad"
                );
            }
            None => {
                bail!(
                    "OpenCL cache manifest entry missing for {key}; run 'ccnc_visiond --warmup-cache' offroad"
                );
            }
        }
    }
    Ok(())
}

/// Check whether all 5 MNN OpenCL `.clcache` files and `cache.manifest` match current assets/driver.
pub fn verify_all_caches_ready(assets: &Path, dir: &Path) -> Result<()> {
    verify_cache_ready(assets, dir, &ALL_MODELS, Backend::OpenClFp16)
}

/// Multi-shape YOLOX-s MNN detector: routes `416x192` (front), `416x320` (side), and `416x416` (raw).
pub struct MnnDetectorEngine {
    front: MnnSession,
    side: MnnSession,
    raw: MnnSession,
}

impl MnnDetectorEngine {
    pub fn open(assets: &Path, backend: Backend, threads: usize) -> Result<Self> {
        let cdir = cache_dir();
        verify_cache_ready(assets, &cdir, &DETECTOR_MODELS, backend)?;
        let api = MnnApi::load(assets)?;
        let open_one = |name: &str, expected: [usize; 4]| -> Result<MnnSession> {
            let cf = (backend == Backend::OpenClFp16).then(|| cache_file_for(&cdir, name));
            let s = MnnSession::open(Arc::clone(&api), &assets.join(name), backend, threads, cf.as_deref())?;
            if s.in_shape() != expected {
                bail!("{name}: input shape {:?} != expected {:?}", s.in_shape(), expected);
            }
            Ok(s)
        };
        let front = open_one(MODEL_FRONT, [1, 3, 192, 416])?;
        let side = open_one(MODEL_SIDE, [1, 3, 320, 416])?;
        let raw = open_one(MODEL_RAW, [1, 3, 416, 416])?;
        Ok(Self { front, side, raw })
    }
}

impl Engine for MnnDetectorEngine {
    fn run(&mut self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
        match shape {
            [1, 3, 192, 416] => self.front.run_slice(&data),
            [1, 3, 320, 416] => self.side.run_slice(&data),
            [1, 3, 416, 416] => self.raw.run_slice(&data),
            other => bail!("MnnDetectorEngine: unsupported input shape {other:?}"),
        }
    }

    /// `yolox_s_*_fp16.mnn` models are exported with `decode_in_inference=True` (already in input pixel coords).
    fn needs_grid_decode(&self) -> bool {
        false
    }
}

/// Fixed-batch MNv3 classifier: `n == 1` uses `kind1_fp16.mnn`, `n in 2..=4` pads to `kind4_fp16.mnn`.
pub struct MnnClassifierEngine {
    kind1: MnnSession,
    kind4: MnnSession,
}

impl MnnClassifierEngine {
    pub fn open(assets: &Path, backend: Backend, threads: usize) -> Result<Self> {
        let cdir = cache_dir();
        verify_cache_ready(assets, &cdir, &CLASSIFIER_MODELS, backend)?;
        let api = MnnApi::load(assets)?;
        let open_one = |name: &str, expected: [usize; 4]| -> Result<MnnSession> {
            let cf = (backend == Backend::OpenClFp16).then(|| cache_file_for(&cdir, name));
            let s = MnnSession::open(Arc::clone(&api), &assets.join(name), backend, threads, cf.as_deref())?;
            if s.in_shape() != expected {
                bail!("{name}: input shape {:?} != expected {:?}", s.in_shape(), expected);
            }
            Ok(s)
        };
        let kind1 = open_one(MODEL_KIND1, [1, 3, 128, 128])?;
        let kind4 = open_one(MODEL_KIND4, [4, 3, 128, 128])?;
        Ok(Self { kind1, kind4 })
    }
}

impl Engine for MnnClassifierEngine {
    fn run(&mut self, shape: [usize; 4], mut data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
        let [n, c, h, w] = shape;
        if c != 3 || h != 128 || w != 128 || n == 0 || n > 4 {
            bail!("MnnClassifierEngine: unsupported input shape {shape:?}");
        }
        if n == 1 {
            return self.kind1.run_slice(&data);
        }
        let crop_elems = 3 * 128 * 128;
        if data.len() != n * crop_elems {
            bail!("MnnClassifierEngine: input len {} != {}", data.len(), n * crop_elems);
        }
        data.resize(4 * crop_elems, 0.0);
        let (out_shape, mut out) = self.kind4.run_slice(&data)?;
        let classes = *out_shape.last().ok_or_else(|| anyhow!("empty kind4 output shape"))?;
        out.truncate(n * classes);
        Ok((vec![n, classes], out))
    }
}

/// Offroad OpenCL cache pre-generation (`ccnc_visiond --warmup-cache`).
/// Compiles and tunes all 5 `.mnn` models into `cache_dir` so onroad startup reloads in ~1.2 s with 0 DM drops.
pub fn warmup_cache(assets: &Path, cache_dir: &Path, backend: Backend, threads: usize) -> Result<()> {
    std::fs::create_dir_all(cache_dir).with_context(|| format!("create cache dir {}", cache_dir.display()))?;
    if backend == Backend::OpenClFp16 && verify_cache_ready(assets, cache_dir, &ALL_MODELS, backend).is_err() {
        // Purge stale .clcache files before rebuilding so MNN never attempts to load mismatched kernels.
        for &model_name in &ALL_MODELS {
            let _ = std::fs::remove_file(cache_file_for(cache_dir, model_name));
        }
        let _ = std::fs::remove_file(cache_dir.join(CACHE_MANIFEST));
    }
    let api = MnnApi::load(assets)?;
    let t_all = Instant::now();
    for &model_name in &ALL_MODELS {
        let model_path = assets.join(model_name);
        let cf = cache_file_for(cache_dir, model_name);
        let t0 = Instant::now();
        let mut sess = MnnSession::open(
            Arc::clone(&api),
            &model_path,
            backend,
            threads,
            (backend == Backend::OpenClFp16).then_some(cf.as_path()),
        )?;
        let in_elems: usize = sess.in_shape().iter().product();
        let dummy = vec![0.0f32; in_elems];
        let (_out_shape, out) = sess.run_slice(&dummy)?;
        if !out.iter().all(|v| v.is_finite()) {
            bail!("warmup {model_name}: non-finite output detected");
        }
        if backend == Backend::OpenClFp16 {
            sess.update_cache()?;
        }
        let cache_bytes = cf.metadata().map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "ccnc_visiond: warmup {} backend={} in={:?} out={:?} ms={:.1} cache={} ({} B)",
            model_name,
            sess.active_backend(),
            sess.in_shape(),
            sess.out_shape(),
            t0.elapsed().as_secs_f64() * 1000.0,
            cf.display(),
            cache_bytes,
        );
    }
    if backend == Backend::OpenClFp16 {
        write_cache_manifest(assets, cache_dir, &ALL_MODELS)?;
    }
    eprintln!(
        "ccnc_visiond: warmup complete ({} models) in {:.2} s -> {}",
        ALL_MODELS.len(),
        t_all.elapsed().as_secs_f64(),
        cache_dir.display(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_file_naming_and_missing_check() {
        let dir = std::env::temp_dir().join(format!("ccnc_visiond_mnn_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(cache_file_for(&dir, MODEL_FRONT), dir.join("yolox_s_416x192_fp16.clcache"));
        assert_eq!(missing_caches(&dir, &DETECTOR_MODELS).len(), 3);
        for m in DETECTOR_MODELS {
            std::fs::write(cache_file_for(&dir, m), b"dummy-cache").unwrap();
        }
        assert!(missing_caches(&dir, &DETECTOR_MODELS).is_empty());
        // Empty file does not count as a valid cache.
        std::fs::write(cache_file_for(&dir, MODEL_SIDE), b"").unwrap();
        assert_eq!(missing_caches(&dir, &DETECTOR_MODELS).len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn opencl_requires_warm_cache_by_default() {
        let dir = std::env::temp_dir().join(format!("ccnc_visiond_mnn_cold_{}", std::process::id()));
        let assets = dir.join("assets");
        let cache = dir.join("cache");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::create_dir_all(&cache).unwrap();
        std::env::remove_var("CCNC_VISIOND_MNN_ALLOW_COLD");
        let err = verify_cache_ready(&assets, &cache, &DETECTOR_MODELS, Backend::OpenClFp16).unwrap_err();
        assert!(format!("{err:#}").contains("OpenCL cache missing"));
        // CPU backend does not require OpenCL cache.
        assert!(verify_cache_ready(&assets, &cache, &DETECTOR_MODELS, Backend::Cpu).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_manifest_verifies_hashes_and_rejects_mismatch() {
        let root = std::env::temp_dir().join(format!("ccnc_visiond_mnn_hash_{}", std::process::id()));
        let assets = root.join("assets");
        let cache = root.join("cache");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::create_dir_all(&cache).unwrap();
        std::env::remove_var("CCNC_VISIOND_MNN_ALLOW_COLD");

        for m in ALL_MODELS {
            std::fs::write(assets.join(m), format!("model-bytes-{m}")).unwrap();
            std::fs::write(cache_file_for(&cache, m), b"compiled-opencl-kernel").unwrap();
        }
        std::fs::write(assets.join("libMNN.so"), b"mnn-runtime-v362").unwrap();
        std::fs::write(assets.join("libMNN_CL.so"), b"mnn-cl-v362").unwrap();

        // Missing manifest must fail even when .clcache files exist.
        let err = verify_cache_ready(&assets, &cache, &DETECTOR_MODELS, Backend::OpenClFp16).unwrap_err();
        assert!(format!("{err:#}").contains("manifest missing"), "got: {err:#}");

        // Writing manifest makes both detector and classifier verification succeed.
        write_cache_manifest(&assets, &cache, &ALL_MODELS).unwrap();
        assert!(verify_cache_ready(&assets, &cache, &DETECTOR_MODELS, Backend::OpenClFp16).is_ok());
        assert!(verify_cache_ready(&assets, &cache, &CLASSIFIER_MODELS, Backend::OpenClFp16).is_ok());

        // Replacing a .mnn model file triggers hash mismatch.
        std::fs::write(assets.join(MODEL_SIDE), b"updated-side-model-v2").unwrap();
        let err = verify_cache_ready(&assets, &cache, &DETECTOR_MODELS, Backend::OpenClFp16).unwrap_err();
        assert!(format!("{err:#}").contains("hash mismatch"), "got: {err:#}");
        assert!(format!("{err:#}").contains(MODEL_SIDE), "got: {err:#}");

        // Restoring model and mutating libMNN.so triggers runtime hash mismatch.
        std::fs::write(assets.join(MODEL_SIDE), format!("model-bytes-{MODEL_SIDE}")).unwrap();
        assert!(verify_cache_ready(&assets, &cache, &DETECTOR_MODELS, Backend::OpenClFp16).is_ok());
        std::fs::write(assets.join("libMNN.so"), b"mnn-runtime-v370").unwrap();
        let err = verify_cache_ready(&assets, &cache, &DETECTOR_MODELS, Backend::OpenClFp16).unwrap_err();
        assert!(format!("{err:#}").contains("hash mismatch"), "got: {err:#}");
        assert!(format!("{err:#}").contains("libMNN.so"), "got: {err:#}");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn stale_clcache_rejects_mnn_and_falls_back_to_cpu_ort() {
        let root = std::env::temp_dir().join(format!("ccnc_visiond_mnn_fallback_{}", std::process::id()));
        let assets = root.join("assets");
        let cache = root.join("cache");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::create_dir_all(&cache).unwrap();

        for m in ALL_MODELS {
            std::fs::write(assets.join(m), format!("v1-{m}")).unwrap();
            std::fs::write(cache_file_for(&cache, m), b"stale-clcache").unwrap();
        }
        write_cache_manifest(&assets, &cache, &ALL_MODELS).unwrap();
        // Mutate models so cache manifest is now stale.
        std::fs::write(assets.join(MODEL_FRONT), b"v2-front-model").unwrap();
        std::fs::write(assets.join(MODEL_KIND1), b"v2-kind1-model").unwrap();

        std::env::remove_var("CCNC_VISIOND_MNN_ALLOW_COLD");
        std::env::set_var("CCNC_VISIOND_MNN_CACHE", &cache);
        let fake_ort = assets.join("nonexistent_libonnxruntime.so");
        std::env::set_var("ORT_DYLIB_PATH", &fake_ort);

        // Detector::load("mnn") must reject stale MNN cache and attempt CPU ORT fallback.
        let det_err = crate::detect::Detector::load(&assets, "mnn", 1).err().unwrap();
        let det_msg = format!("{det_err:#}");
        assert!(
            det_msg.contains("nonexistent_libonnxruntime.so") || det_msg.contains("loading"),
            "expected CPU ORT fallback attempt after MNN cache rejection, got: {det_msg}"
        );

        // Classifier open_engine("mnn") must likewise reject stale MNN cache and attempt CPU ORT fallback.
        let cls_err = crate::classify::open_engine(&assets, "mnn").err().unwrap();
        let cls_msg = format!("{cls_err:#}");
        assert!(
            cls_msg.contains("nonexistent_libonnxruntime.so") || cls_msg.contains("loading"),
            "expected classifier CPU ORT fallback attempt, got: {cls_msg}"
        );

        std::env::remove_var("CCNC_VISIOND_MNN_CACHE");
        std::env::remove_var("ORT_DYLIB_PATH");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[ignore = "requires CCNC_VISIOND_OFFLINE_EVAL_DIR + host libmnn_capi.so + .mnn assets"]
    fn offline_real_car_eval_pipeline() {
        use std::io::Write;
        let eval_dir = PathBuf::from(std::env::var("CCNC_VISIOND_OFFLINE_EVAL_DIR").expect("CCNC_VISIOND_OFFLINE_EVAL_DIR"));
        let assets = PathBuf::from(std::env::var("CCNC_VISIOND_ASSETS").expect("CCNC_VISIOND_ASSETS"));
        let mut det = MnnDetectorEngine::open(&assets, Backend::Cpu, 2).expect("open MnnDetectorEngine");
        let mut cls = MnnClassifierEngine::open(&assets, Backend::Cpu, 2).expect("open MnnClassifierEngine");
        assert!(!det.needs_grid_decode());

        let manifest = std::fs::read_to_string(eval_dir.join("manifest.tsv")).expect("read manifest.tsv");
        let mut out_file = std::fs::File::create(eval_dir.join("rust_mnn_out.tsv")).expect("create rust_mnn_out.tsv");
        for line in manifest.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            let id = parts[0];
            let mode = parts[1]; // "front" (192x416) or "side" (320x416) or "raw" (416x416)
            let w: usize = parts[2].parse().unwrap();
            let h: usize = parts[3].parse().unwrap();
            let bgr = std::fs::read(eval_dir.join(parts[4])).expect("read raw bgr");
            assert_eq!(bgr.len(), w * h * 3);
            let img = crate::pre::Img { w, h, data: bgr };
            let (in_h, in_w) = match mode {
                "front" => (192, 416),
                "side" => (320, 416),
                "raw" => (416, 416),
                other => panic!("unknown mode {other}"),
            };
            let (lb, ratio) = crate::pre::letterbox(&img, in_h, in_w, None);
            let (_, out) = det.run([1, 3, in_h, in_w], crate::pre::to_chw_raw(&lb)).expect("det run");
            let cands = crate::yolox::candidates(&out, 80, 0.05);
            let keep = crate::yolox::nms(&cands, 0.05, 0.45);
            let boxes = crate::yolox::to_image(&cands, &keep, ratio, img.w, img.h);
            // Classify vehicles only (car=2, bus=5, truck=7) using MNv3 kind1/kind4
            let picked = crate::classify::pick(&boxes, |_| false);
            let mut kind_map: Vec<([f64; 4], u8, f32)> = Vec::new();
            if !picked.is_empty() {
                let job = crate::classify::Job::new(
                    0,
                    ccnc_vision_msg::View::CropFront,
                    ccnc_vision_msg::Calibration::UNKNOWN,
                    &img,
                    &picked,
                );
                let n = job.boxes.len();
                let (kshape, probs) = cls.run([n, 3, 128, 128], job.input).expect("cls run");
                assert_eq!(kshape, vec![n, 3]);
                let kinds = [ccnc_vision_msg::KIND_CAR, ccnc_vision_msg::KIND_TRUCK, ccnc_vision_msg::KIND_BUS];
                for (det_ref, p) in picked.iter().zip(probs.chunks_exact(3)) {
                    let (best, prob) = p.iter().enumerate().fold((0, f32::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
                    kind_map.push((det_ref.b, kinds[best], prob));
                }
            }
            for b in &boxes {
                if b.score < 0.30 {
                    continue;
                }
                let kind_str = kind_map
                    .iter()
                    .find(|(kb, _, _)| *kb == b.b)
                    .map(|(_, k, p)| format!("{k}:{p:.4}"))
                    .unwrap_or_else(|| "none".into());
                writeln!(
                    out_file,
                    "{id}\t{mode}\t{}\t{:.4}\t{:.1},{:.1},{:.1},{:.1}\t{kind_str}",
                    b.cls, b.score, b.b[0], b.b[1], b.b[2], b.b[3]
                )
                .unwrap();
            }
        }
    }
}
