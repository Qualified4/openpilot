//! StripNet inference on onnxruntime (feature `ort`; STRIPNET_ENGINE=ort-cpu | ort-xnnpack), the library loaded at run
//! time (ORT_DYLIB_PATH, the same `ort` 2.0.0-rc.15 / api-31 setup as the detection runner).
//!
//! N=1: two sessions of one intra-op thread each, so both sides run in parallel like the tract plan (a side takes the
//! first free session). N=2: one session, STRIPNET_ORT_THREADS intra-op threads (default 1).
//! STRIPNET_ORT_KLEIDIAI=0 sets `mlas.disable_kleidiai` (MLAS's KleidiAI kernels off); STRIPNET_ORT_SPIN=1 lets the
//! intra-op pool spin (off by default: spinning only burns CPU between the short runs).

use std::sync::{Mutex, OnceLock};

use ort::ep::XNNPACK;
use ort::environment::Environment;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;

static ENV: OnceLock<Environment> = OnceLock::new();

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name).map(|v| v == "1").unwrap_or(default)
}

pub fn ort_threads() -> usize {
    std::env::var("STRIPNET_ORT_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

pub fn kleidiai() -> bool {
    env_flag("STRIPNET_ORT_KLEIDIAI", true)
}

fn b<T>(r: Result<T, ort::Error<SessionBuilder>>) -> anyhow::Result<T> {
    r.map_err(|e| anyhow::anyhow!("ort session builder: {e}"))
}

fn session(data: &[u8], xnnpack: bool, threads: usize) -> anyhow::Result<Session> {
    if ENV.get().is_none() {
        let lib = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "libonnxruntime.so".into());
        let env = ort::init_from(&lib).map_err(|e| anyhow::anyhow!("loading {lib}: {e}"))?.build()?;
        let _ = ENV.set(env);
    }
    let mut sb = Session::builder(ENV.get().unwrap())?;
    sb = b(sb.with_optimization_level(GraphOptimizationLevel::Level3))?;
    sb = b(sb.with_intra_threads(threads))?;
    sb = b(sb.with_inter_threads(1))?;
    sb = b(sb.with_parallel_execution(false))?;
    sb = b(sb.with_config_entry("session.intra_op.allow_spinning", if env_flag("STRIPNET_ORT_SPIN", false) { "1" } else { "0" }))?;
    if !kleidiai() {
        sb = b(sb.with_config_entry("mlas.disable_kleidiai", "1"))?;
    }
    if xnnpack {
        let n = std::num::NonZeroUsize::new(threads).unwrap();
        sb = b(sb.with_execution_providers([XNNPACK::default().with_intra_op_num_threads(n).build().error_on_failure()]))?;
    }
    Ok(sb.commit_from_memory(data)?)
}

pub struct OrtModel {
    singles: Vec<Mutex<Session>>,
    batch: Option<Mutex<Session>>,
    data: Vec<u8>,
    xnnpack: bool,
    shape: [usize; 3],
}

impl OrtModel {
    pub fn new(data: Vec<u8>, xnnpack: bool, shape: [usize; 3]) -> anyhow::Result<Self> {
        let singles = (0..2).map(|_| session(&data, xnnpack, 1).map(Mutex::new)).collect::<anyhow::Result<_>>()?;
        Ok(OrtModel { singles, batch: None, data, xnnpack, shape })
    }

    pub fn enable_batch(&mut self) -> anyhow::Result<()> {
        self.batch = Some(Mutex::new(session(&self.data, self.xnnpack, ort_threads())?));
        Ok(())
    }

    pub fn batched(&self) -> bool {
        self.batch.is_some()
    }

    /// (type, colour) logits, n strips concatenated: each Vec holds n rows.
    fn run(s: &mut Session, n: usize, shape: [usize; 3], input: Vec<f32>) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let t = Tensor::from_array(([n, shape[0], shape[1], shape[2]], input))?;
        let out = s.run(ort::inputs!["strip_ycbcr" => t])?;
        let get = |k: &str| -> anyhow::Result<Vec<f32>> { Ok(out[k].try_extract_tensor::<f32>()?.1.to_vec()) };
        Ok((get("type")?, get("color")?))
    }

    pub fn forward(&self, strip: &[f32]) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let mut guard = loop {
            if let Some(g) = self.singles.iter().find_map(|m| m.try_lock().ok()) {
                break g;
            }
            std::thread::yield_now();
        };
        Self::run(&mut guard, 1, self.shape, strip.to_vec())
    }

    pub fn forward2(&self, a: &[f32], b: &[f32]) -> anyhow::Result<[(Vec<f32>, Vec<f32>); 2]> {
        let m = self.batch.as_ref().ok_or_else(|| anyhow::anyhow!("batch session not built"))?;
        let mut both = Vec::with_capacity(a.len() * 2);
        both.extend_from_slice(a);
        both.extend_from_slice(b);
        let (t, c) = Self::run(&mut m.lock().unwrap(), 2, self.shape, both)?;
        let (tw, cw) = (t.len() / 2, c.len() / 2);
        Ok([(t[..tw].to_vec(), c[..cw].to_vec()), (t[tw..].to_vec(), c[cw..].to_vec())])
    }
}
