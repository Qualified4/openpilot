//! The wide camera's lens and the side views' remap tables, built from the device calibration.
//! Same model and steps as `wcam.py` (`WCam.project` and `Virt`) of the offline evaluation, in f64, stored as f32.

use crate::remap::Map;
use ccnc_vision_msg::Calibration;

/// Fisheye fit of the os04c10 wide camera (wide-side-20261121/data/lens.json): r = F (θ + K1 θ³) around (CX, CY).
const F: f64 = 442.80633225081266;
const K1: f64 = 0.03044466692675767;
const CX: f64 = 669.4770181606455;
const CY: f64 = 379.8620143972138;

/// Side view: a 512x384 pinhole with an 80 degree horizontal field, 8 degrees down, 55 degrees to each side.
pub const VIEW_W: usize = 512;
pub const VIEW_H: usize = 384;
const HFOV_DEG: f64 = 80.0;
const PITCH_DEG: f64 = 8.0;
/// Yaw of the left and right views (+ is right), in `Side` order.
const YAW_DEG: [f64; 2] = [-55.0, 55.0];

type Mat = [[f64; 3]; 3];

fn mul(a: &Mat, b: &Mat) -> Mat {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]))
}

fn apply(m: &Mat, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| v[0] * m[i][0] + v[1] * m[i][1] + v[2] * m[i][2])
}

fn transpose(m: &Mat) -> Mat {
    std::array::from_fn(|i| std::array::from_fn(|j| m[j][i]))
}

fn rot_x(a: f64) -> Mat {
    let (s, c) = a.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn rot_y(a: f64) -> Mat {
    let (s, c) = a.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

fn rot_z(a: f64) -> Mat {
    let (s, c) = a.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// openpilot `rot_from_euler`: Rz(yaw) Ry(pitch) Rx(roll).
fn rot_from_euler(e: [f32; 3]) -> Mat {
    mul(&mul(&rot_z(e[2].into()), &rot_y(e[1].into())), &rot_x(e[0].into()))
}

/// Wide camera view frame (x right, y down, z forward) from the calibrated frame (x forward, y right, z down).
fn wide_from_calib(cal: &Calibration) -> Mat {
    const VIEW_FROM_DEVICE: Mat = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
    mul(&mul(&VIEW_FROM_DEVICE, &rot_from_euler(cal.wide_from_device_euler)), &rot_from_euler(cal.rpy_calib))
}

/// Calibrated frame from a side view's frame (x right, y down, z forward).
fn calib_from_view(yaw_deg: f64) -> Mat {
    const BASE: Mat = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    mul(&mul(&rot_z(yaw_deg.to_radians()), &transpose(&rot_y(PITCH_DEG.to_radians()))), &BASE)
}

/// Wide camera pixel of a ray in the wide camera's view frame.
fn project(w: [f64; 3]) -> (f64, f64) {
    let n = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let theta = (w[2] / n).clamp(-1.0, 1.0).acos();
    let phi = w[1].atan2(w[0]);
    let r = F * (theta + K1 * theta.powi(3));
    (CX + r * phi.cos(), CY + r * phi.sin())
}

/// Remap table of one side view (0 left, 1 right): for each view pixel, where it is in the wide camera frame.
pub fn side_map(cal: &Calibration, side: usize) -> Map {
    let wide = wide_from_calib(cal);
    let view = calib_from_view(YAW_DEG[side]);
    let f = VIEW_W as f64 / 2.0 / (HFOV_DEG / 2.0).to_radians().tan();
    let (half_w, half_h) = (VIEW_W as f64 / 2.0, VIEW_H as f64 / 2.0);
    let n = VIEW_W * VIEW_H;
    let (mut mx, mut my) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for v in 0..VIEW_H {
        for u in 0..VIEW_W {
            let ray = apply(&view, [(u as f64 - half_w) / f, (v as f64 - half_h) / f, 1.0]);
            let (x, y) = project(apply(&wide, ray));
            mx.push(x as f32);
            my.push(y as f32);
        }
    }
    Map { w: VIEW_W, h: VIEW_H, mx, my }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// liveCalibration of route 0000046c--eaf2b5dd20--10, the route the earlier fixed maps were made from.
    const ROUTE_46C: Calibration = Calibration {
        rpy_calib: [-8.625534974271432e-06, -0.01063278317451477, 0.003989458549767733],
        wide_from_device_euler: [-6.496889568552433e-07, -0.009829160757362843, -0.018073642626404762],
        height: 1.2227749824523926,
    };

    #[test]
    fn matches_the_python_side_views() {
        // (side, u, v, mx, my) from wcam.py Virt(WCam(rpy, wfd, h), -+55) for ROUTE_46C, as f32.
        const PYTHON: [(usize, usize, usize, f32, f32); 6] = [
            (0, 0, 0, -62.8766975402832, 119.17070007324219),
            (0, 256, 192, 228.24192810058594, 461.10284423828125),
            (0, 511, 383, 578.050537109375, 638.4431762695312),
            (1, 0, 0, 790.7670288085938, 235.08348083496094),
            (1, 256, 192, 1097.3414306640625, 460.1588134765625),
            (1, 511, 383, 1358.9832763671875, 806.2539672851562),
        ];
        let maps = [side_map(&ROUTE_46C, 0), side_map(&ROUTE_46C, 1)];
        for (side, u, v, x, y) in PYTHON {
            let i = v * VIEW_W + u;
            let (got_x, got_y) = (maps[side].mx[i], maps[side].my[i]);
            assert!((got_x - x).abs() <= 1e-3 && (got_y - y).abs() <= 1e-3, "side {side} ({u}, {v}): ({got_x}, {got_y}) != ({x}, {y})");
        }
    }

    /// Writes the side maps of calibrations for the parity check against wcam.py. Not part of the normal run:
    /// `CCNC_VISIOND_MAP_DUMP=<list> cargo test --release -- --ignored write_side_maps`. Each line of the list is
    /// `<out prefix> roll pitch yaw wide_roll wide_pitch wide_yaw`; it writes `<out prefix>_L.bin` and `_R.bin` (u32 w,
    /// u32 h, f32 mx, f32 my) and prints the build time of each map.
    #[test]
    #[ignore]
    fn write_side_maps() {
        let list = std::env::var("CCNC_VISIOND_MAP_DUMP").expect("CCNC_VISIOND_MAP_DUMP");
        for line in std::fs::read_to_string(list).unwrap().lines().filter(|l| !l.trim().is_empty()) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let e: Vec<f32> = parts[1..7].iter().map(|s| s.parse().unwrap()).collect();
            let cal = Calibration { rpy_calib: [e[0], e[1], e[2]], wide_from_device_euler: [e[3], e[4], e[5]], height: 1.0 };
            for (side, name) in ["L", "R"].iter().enumerate() {
                let start = std::time::Instant::now();
                let map = side_map(&cal, side);
                println!("built {} {name} {:.2} ms", parts[0], start.elapsed().as_secs_f64() * 1000.0);
                let mut bytes = Vec::with_capacity(8 + 8 * map.mx.len());
                bytes.extend((map.w as u32).to_le_bytes());
                bytes.extend((map.h as u32).to_le_bytes());
                map.mx.iter().chain(&map.my).for_each(|v| bytes.extend(v.to_le_bytes()));
                std::fs::write(format!("{}_{name}.bin", parts[0]), bytes).unwrap();
            }
        }
    }
}
