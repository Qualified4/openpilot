//! StripNet v4 (assets/lane_strip_v4.{onnx,json}) on tract, and strip_model.StripModel.decide4 / decide.
//!
//! Only the product contract is accepted: a 5-type v4 manifest (OUTPUTS_V4) whose sha256 matches the ONNX file. A
//! 4-type manifest (OUTPUTS_V4_4) and the replay switches HOLD_COLOR / HOLD_BY_FAMILY are not ported.

use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use tract_onnx::prelude::*;

use crate::strip::{StripShape, DEFAULT_STRIP};

/// tract's intra-op threads (its matrix products split over a rayon pool shared by both sides): STRIPNET_TRACT_THREADS=n,
/// unset or 1 = single-threaded (each side's inference on its own thread only). Set once, before the first run.
pub fn tract_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("STRIPNET_TRACT_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize);
        if n > 1 {
            tract_linalg::multithread::set_default_executor(tract_linalg::multithread::Executor::multithread(n));
        }
        n
    })
}

pub const TYPES_V4: [&str; 5] = ["dashed", "solid", "dbl_near", "dbl_far", "unknown"];
pub const COLORS_V4: [&str; 3] = ["white", "yellow", "blue"];

#[derive(Deserialize)]
struct Input {
    name: String,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct Outputs {
    color: Vec<String>,
    #[serde(rename = "type")]
    kind: Vec<String>,
}

#[derive(Deserialize, Default)]
struct Strip {
    half_m: Option<f64>,
    cols: Option<usize>,
    rows: Option<usize>,
}

#[derive(Deserialize)]
struct Manifest {
    version: serde_json::Value,
    model_sha256: String,
    outputs: Outputs,
    #[serde(default)]
    mirror_right: bool,
    #[serde(default)]
    strip: Option<Strip>,
    input: Input,
    threshold: f64,
}

/// One side's decide4 result (the dict strip_model returns; `typeClass` etc. are the v4 extras).
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub reason: &'static str,
    pub kind: &'static str,
    pub type_confidence: f64,
    pub type_candidate: &'static str,
    pub color: &'static str,
    pub color_confidence: f64,
    pub color_candidate: &'static str,
    pub type_class: &'static str,
    pub color_class: &'static str,
    pub double: bool,
    pub blue: bool,
    pub p_unknown: f64,
}

type Plan = std::sync::Arc<TypedRunnableModel>;

pub struct StripModel {
    pub name: String,
    pub version: serde_json::Value,
    pub sha256: String,
    pub mirror_right: bool,
    pub strip: StripShape,
    pub threshold: f64,
    plan: Plan,
    /// The N=2 plan (both sides in one run), built by `enable_batch`.
    plan2: Option<Plan>,
    data: Vec<u8>,
}

fn build_plan(data: &[u8], n: usize, strip: StripShape) -> anyhow::Result<Plan> {
    let mut model = tract_onnx::onnx().model_for_read(&mut &data[..])?;
    model.set_input_fact(0, f32::fact([n, 3, strip.rows, strip.cols]).into())?;
    let outputs: Vec<OutletId> = ["type", "color"]
        .iter()
        .map(|n| model.find_outlet_label(n))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow::anyhow!("lane strip ONNX outputs type / color not found"))?;
    model.select_output_outlets(&outputs)?;
    Ok(model.into_optimized()?.into_runnable()?)
}

fn softmax(x: &[f32]) -> Vec<f64> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = x.iter().map(|&v| (v as f64 - max).exp()).collect();
    let sum = e.iter().fold(0., |a, b| a + b);
    e.iter().map(|v| v / sum).collect()
}

fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

impl StripModel {
    pub fn load(directory: &Path, name: &str) -> anyhow::Result<Self> {
        tract_threads();
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(directory.join(format!("{name}.json")))?)?;
        let data = std::fs::read(directory.join(format!("{name}.onnx")))?;
        let digest = format!("{:x}", Sha256::digest(&data));
        if digest != manifest.model_sha256 || manifest.outputs.color != COLORS_V4 || manifest.outputs.kind != TYPES_V4 {
            anyhow::bail!("lane strip model artifact contract mismatch");
        }
        let given = manifest.strip.unwrap_or_default();
        let strip = StripShape {
            half_m: given.half_m.unwrap_or(DEFAULT_STRIP.half_m),
            cols: given.cols.unwrap_or(DEFAULT_STRIP.cols),
            rows: given.rows.unwrap_or(DEFAULT_STRIP.rows),
        };
        if manifest.input.shape != [3, strip.rows, strip.cols] {
            anyhow::bail!("lane strip manifest input shape {:?} does not match its strip", manifest.input.shape);
        }
        let _ = &manifest.input.name;
        let plan = build_plan(&data, 1, strip)?;
        Ok(StripModel { name: name.into(), version: manifest.version, sha256: manifest.model_sha256, mirror_right: manifest.mirror_right,
                        strip, threshold: manifest.threshold, plan, plan2: None, data })
    }

    pub fn enable_batch(&mut self) -> anyhow::Result<()> {
        self.plan2 = Some(build_plan(&self.data, 2, self.strip)?);
        Ok(())
    }

    pub fn batched(&self) -> bool {
        self.plan2.is_some()
    }

    /// Both strips in one N=2 run: [(type logits, colour logits); 2].
    pub fn forward2(&self, a: &[f32], b: &[f32]) -> anyhow::Result<[(Vec<f32>, Vec<f32>); 2]> {
        let plan = self.plan2.as_ref().ok_or_else(|| anyhow::anyhow!("batch plan not built"))?;
        let mut both = Vec::with_capacity(a.len() * 2);
        both.extend_from_slice(a);
        both.extend_from_slice(b);
        let input = Tensor::from_shape(&[2, 3, self.strip.rows, self.strip.cols], &both)?;
        let out = plan.run(tvec!(input.into()))?;
        let rows = |i: usize| -> anyhow::Result<Vec<Vec<f32>>> {
            let v = out[i].to_plain_array_view::<f32>()?;
            let flat: Vec<f32> = v.iter().copied().collect();
            let w = flat.len() / 2;
            Ok(vec![flat[..w].to_vec(), flat[w..].to_vec()])
        };
        let (mut t, mut c) = (rows(0)?, rows(1)?);
        let (t1, c1) = (t.pop().unwrap(), c.pop().unwrap());
        Ok([(t.pop().unwrap(), c.pop().unwrap()), (t1, c1)])
    }

    /// (type logits, colour logits) of one 3 x rows x cols strip.
    pub fn forward(&self, strip: &[f32]) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let input = Tensor::from_shape(&[1, 3, self.strip.rows, self.strip.cols], strip)?;
        let out = self.plan.run(tvec!(input.into()))?;
        let flat = |i: usize| -> anyhow::Result<Vec<f32>> { Ok(out[i].to_plain_array_view::<f32>()?.iter().copied().collect()) };
        Ok((flat(0)?, flat(1)?))
    }

    /// strip_model.StripModel.decide4 for the 5-type head.
    pub fn decide4(&self, type_logits: &[f32], color_logits: &[f32]) -> Decision {
        decide4(self.threshold, type_logits, color_logits)
    }
}

pub fn decide4(threshold: f64, type_logits: &[f32], color_logits: &[f32]) -> Decision {
    let (pt, pc) = (softmax(type_logits), softmax(color_logits));
    let (t, c) = (argmax(&pt), argmax(&pc));
    let u = pt[4];
    let base = Decision {
        reason: "unknown", kind: "unknown", type_confidence: 0., type_candidate: "unknown", color: "unknown",
        color_confidence: 0., color_candidate: "unknown", type_class: TYPES_V4[t], color_class: COLORS_V4[c],
        double: t == 2 || t == 3, blue: c == 2, p_unknown: u,
    };
    if t == 4 {
        return base;
    }
    let dashed = 1f64.min((pt[0] + pt[2]) / (1. - u));
    let yellow = pc[1];
    // decide(): argmax of the two-class vectors (ties: the first), confident at >= threshold.
    let tp = [dashed, 1. - dashed];
    let cp = [1. - yellow, yellow];
    let (ti, ci) = (argmax(&tp), argmax(&cp));
    let (types, colors) = (["dashed", "solid"], ["white", "yellow"]);
    Decision {
        reason: "strip_model",
        kind: if tp[ti] >= threshold { types[ti] } else { "unknown" },
        type_confidence: tp[ti],
        type_candidate: types[ti],
        color: if cp[ci] >= threshold { colors[ci] } else { "unknown" },
        color_confidence: cp[ci],
        color_candidate: colors[ci],
        ..base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_argmax_abstains() {
        let d = decide4(0.9, &[0., 0., 0., 0., 5.], &[0., 3., -1e4]);
        assert_eq!(d.reason, "unknown");
        assert_eq!(d.kind, "unknown");
        assert_eq!(d.type_class, "unknown");
        assert_eq!(d.color_class, "yellow");
    }

    #[test]
    fn dashed_yellow_passes() {
        let d = decide4(0.9, &[8., 0., 0., 0., 0.], &[0., 8., -1e4]);
        assert_eq!((d.reason, d.kind, d.color), ("strip_model", "dashed", "yellow"));
        assert!(d.type_confidence > 0.99);
    }

    #[test]
    fn product_model_loads_and_runs() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets");
        let m = StripModel::load(&dir, "lane_strip_v4").unwrap();
        assert!(m.mirror_right);
        assert_eq!(m.strip, DEFAULT_STRIP);
        let (t, c) = m.forward(&vec![128.; 3 * 128 * 128]).unwrap();
        assert_eq!((t.len(), c.len()), (5, 3));
        let mut m = m;
        m.enable_batch().unwrap();
        let b: Vec<f32> = (0..3 * 128 * 128).map(|i| (i % 251) as f32).collect();
        let [x, y] = m.forward2(&vec![128.; 3 * 128 * 128], &b).unwrap();
        let (tb, cb) = m.forward(&b).unwrap();
        let close = |p: &[f32], q: &[f32]| p.iter().zip(q).all(|(a, b)| (a - b).abs() < 1e-3);
        assert!(close(&x.0, &t) && close(&x.1, &c) && close(&y.0, &tb) && close(&y.1, &cb));
    }
}
