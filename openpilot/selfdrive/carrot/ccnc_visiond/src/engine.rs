//! ONNX inference backends: onnxruntime (`ort`, library loaded at run time) and `tract` (pure Rust).
use anyhow::{bail, Result};

pub trait Engine {
    /// Run a single-input model on an NCHW f32 tensor; returns (output shape, data) of output 0.
    fn run(&mut self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)>;
    /// Whether YOLOX outputs are raw logits needing `yolox::grid_decode`.
    /// `true` for CPU ORT `yolox_tiny_int8.onnx`, `false` for MNN `yolox_s_*_fp16.mnn` (`decode_in_inference=True`).
    fn needs_grid_decode(&self) -> bool {
        true
    }
}

pub fn open(engine: &str, path: &str, shape: [usize; 4], threads: usize) -> Result<Box<dyn Engine>> {
    match engine {
        #[cfg(feature = "ort")]
        "ort" => Ok(Box::new(ort_backend::OrtEngine::new(path, threads)?)),
        #[cfg(feature = "tract")]
        "tract" => Ok(Box::new(tract_backend::TractEngine::new(path, shape, threads)?)),
        e => {
            let _ = (path, shape, threads);
            bail!("engine '{e}' is not built in (features: ort, tract)")
        }
    }
}

#[cfg(feature = "ort")]
mod ort_backend {
    use super::Engine;
    use anyhow::{Context, Result};
    use ort::environment::Environment;
    use ort::session::{builder::{GraphOptimizationLevel, SessionBuilder}, Session};
    use ort::value::Tensor;
    use std::sync::OnceLock;

    static ENV: OnceLock<Environment> = OnceLock::new();

    pub struct OrtEngine {
        s: Session,
        input: String,
    }

    fn b<T>(r: std::result::Result<T, ort::Error<SessionBuilder>>) -> Result<T> {
        r.map_err(|e| anyhow::anyhow!("ort session builder: {e}"))
    }

    impl OrtEngine {
        pub fn new(path: &str, threads: usize) -> Result<Self> {
            if ENV.get().is_none() {
                // ORT_DYLIB_PATH points at libonnxruntime.so (e.g. the Python wheel's copy)
                let lib = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "libonnxruntime.so".into());
                let env = ort::init_from(&lib).map_err(|e| anyhow::anyhow!("loading {lib}: {e}"))?.build()?;
                let _ = ENV.set(env);
            }
            let mut sb = Session::builder(ENV.get().unwrap())?;
            sb = b(sb.with_optimization_level(GraphOptimizationLevel::Level3))?;
            sb = b(sb.with_intra_threads(threads))?;
            sb = b(sb.with_inter_threads(1))?;
            sb = b(sb.with_parallel_execution(false))?;
            sb = b(sb.with_intra_op_spinning(false))?;
            sb = b(sb.with_inter_op_spinning(false))?;
            let s = sb.commit_from_file(path).map_err(|e| anyhow::anyhow!("{e}")).with_context(|| format!("ort: loading {path}"))?;
            let input = s.inputs()[0].name().to_string();
            Ok(Self { s, input })
        }
    }

    impl Engine for OrtEngine {
        fn run(&mut self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
            let t = Tensor::from_array((shape, data))?;
            let out = self.s.run(ort::inputs![self.input.as_str() => t])?;
            let (sh, d) = out[0].try_extract_tensor::<f32>()?;
            Ok((sh.iter().map(|&x| x as usize).collect(), d.to_vec()))
        }
    }
}

#[cfg(feature = "tract")]
mod tract_backend {
    use super::Engine;
    use anyhow::{Context, Result};
    use tract_onnx::prelude::*;

    pub struct TractEngine {
        m: std::sync::Arc<TypedRunnableModel>,
    }

    impl TractEngine {
        pub fn new(path: &str, shape: [usize; 4], threads: usize) -> Result<Self> {
            if threads > 1 {
                tract_linalg::multithread::set_default_executor(tract_linalg::multithread::Executor::multithread(threads));
            }
            let m = tract_onnx::onnx()
                .model_for_path(path)
                .with_context(|| format!("tract: loading {path}"))?
                .with_input_fact(0, f32::fact(shape).into())?
                .into_optimized()?
                .into_runnable()?;
            Ok(Self { m })
        }
    }

    impl Engine for TractEngine {
        fn run(&mut self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
            let t = Tensor::from_shape(&shape, &data)?;
            let out = self.m.run(tvec!(t.into()))?;
            let v = out[0].to_plain_array_view::<f32>()?;
            Ok((v.shape().to_vec(), v.iter().copied().collect()))
        }
    }
}
