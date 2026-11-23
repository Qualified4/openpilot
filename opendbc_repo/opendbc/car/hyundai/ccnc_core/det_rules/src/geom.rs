//! Wide camera geometry (port of wcam.py / geom.py): fisheye model, calibration frames, the two virtual side views,
//! ground points of a box and the own-car band masks.
//! calib frame: x fwd, y right, z down (from the camera). wide view: x right, y down, z fwd.
use crate::band::Mask;
use crate::{View, WideDet};

pub type M3 = [[f64; 3]; 3];
pub type V3 = [f64; 3];

pub fn matmul(a: &M3, b: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}
pub fn transpose(a: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[j][i];
        }
    }
    o
}
pub fn mulv(a: &M3, v: V3) -> V3 {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}
fn norm(v: V3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// openpilot common/transformations/orientation.rot_from_euler (z-y-x).
pub fn rot_from_euler(r: f64, p: f64, y: f64) -> M3 {
    let (cr, sr, cp, sp, cy, sy) = (r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin());
    let rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]];
    let ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]];
    let rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]];
    matmul(&matmul(&rz, &ry), &rx)
}

/// Fitted os04c10 wide lens (wide-side-20261121/data/lens.json): r = f (theta + k1 theta^3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    pub f: f64,
    pub k1: f64,
    pub cx: f64,
    pub cy: f64,
}
pub const LENS: Lens = Lens { f: 442.80633225081266, k1: 0.03044466692675767, cx: 669.4770181606455, cy: 379.8620143972138 };

/// liveCalibration values the wide camera needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calib {
    pub rpy: [f64; 3],
    pub wide_from_device: [f64; 3],
    pub height: f64,
}

const VD: M3 = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];

#[derive(Clone, Debug)]
pub struct WCam {
    pub r: M3, // wide view <- calib
    pub h: f64,
    pub lens: Lens,
}
impl WCam {
    pub fn new(c: &Calib, lens: Lens) -> WCam {
        let wfd = rot_from_euler(c.wide_from_device[0], c.wide_from_device[1], c.wide_from_device[2]);
        let rpy = rot_from_euler(c.rpy[0], c.rpy[1], c.rpy[2]);
        WCam { r: matmul(&matmul(&VD, &wfd), &rpy), h: c.height, lens }
    }
    /// calib point -> wide pixel.
    pub fn project(&self, p: V3) -> (f64, f64) {
        let v = mulv(&self.r, p);
        let th = (v[2] / norm(v)).clamp(-1.0, 1.0).acos();
        let ph = v[1].atan2(v[0]);
        let r = self.lens.f * (th + self.lens.k1 * th.powf(3.0));
        (self.lens.cx + r * ph.cos(), self.lens.cy + r * ph.sin())
    }
    /// wide pixel -> unit ray in the calib frame.
    pub fn ray(&self, u: f64, v: f64) -> V3 {
        let l = &self.lens;
        let (dx, dy) = (u - l.cx, v - l.cy);
        let r = dx.hypot(dy);
        let mut th = r / l.f;
        for _ in 0..6 {
            th -= (l.f * (th + l.k1 * th.powf(3.0)) - r) / (l.f * (1.0 + 3.0 * l.k1 * (th * th)));
        }
        let ph = dy.atan2(dx);
        let w = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
        mulv(&transpose(&self.r), w)
    }
    /// calib ray -> (x fwd from the camera, y right) on the flat road z = h.
    pub fn ground(&self, d: V3) -> Option<[f64; 2]> {
        if d[2] <= 1e-3 {
            return None;
        }
        let s = self.h / d[2];
        Some([s * d[0], s * d[1]])
    }
}

pub const VIRT_W: usize = 512;
pub const VIRT_H: usize = 384;
pub const VIRT_YAW_DEG: f64 = 55.0;
const VIRT_PITCH_DEG: f64 = 8.0;
const VIRT_HFOV_DEG: f64 = 80.0;
pub const CAR_H: f64 = 1.5; // box height of a car in the size-based distance

/// Virtual pinhole looking at yaw (deg, + = right) and pitch 8 deg down in the calib frame, 512x384, hfov 80.
#[derive(Clone, Debug)]
pub struct Virt {
    pub f: f64,
    pub rcv: M3,
    pub mx: Vec<f32>, // wide pixel of every view pixel (the cv2.remap maps)
    pub my: Vec<f32>,
}
impl Virt {
    pub fn new(cam: &WCam, yaw_deg: f64) -> Virt {
        let f = VIRT_W as f64 / 2.0 / (VIRT_HFOV_DEG / 2.0).to_radians().tan();
        let (y, p) = (yaw_deg.to_radians(), VIRT_PITCH_DEG.to_radians());
        let base = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let rz = [[y.cos(), -y.sin(), 0.0], [y.sin(), y.cos(), 0.0], [0.0, 0.0, 1.0]];
        let ry = [[p.cos(), 0.0, p.sin()], [0.0, 1.0, 0.0], [-p.sin(), 0.0, p.cos()]];
        let rcv = matmul(&matmul(&rz, &transpose(&ry)), &base);
        let mut mx = Vec::with_capacity(VIRT_W * VIRT_H);
        let mut my = Vec::with_capacity(VIRT_W * VIRT_H);
        for vv in 0..VIRT_H {
            for uu in 0..VIRT_W {
                let ray = [(uu as f64 - VIRT_W as f64 / 2.0) / f, (vv as f64 - VIRT_H as f64 / 2.0) / f, 1.0];
                let (px, py) = cam.project(mulv(&rcv, ray));
                mx.push(px as f32);
                my.push(py as f32);
            }
        }
        Virt { f, rcv, mx, my }
    }
    pub fn ray(&self, u: f64, v: f64) -> V3 {
        let d = mulv(&self.rcv, [(u - VIRT_W as f64 / 2.0) / self.f, (v - VIRT_H as f64 / 2.0) / self.f, 1.0]);
        let n = norm(d);
        [d[0] / n, d[1] / n, d[2] / n]
    }
    /// view pixel -> wide pixel (map lookup, coordinates truncated like the Python int()).
    pub fn to_wide(&self, u: f64, v: f64) -> (f64, f64) {
        let iy = v.clamp(0.0, (VIRT_H - 1) as f64) as usize;
        let ix = u.clamp(0.0, (VIRT_W - 1) as f64) as usize;
        (self.mx[iy * VIRT_W + ix] as f64, self.my[iy * VIRT_W + ix] as f64)
    }
    /// cv2.remap(band * 255, mx, my, INTER_LINEAR, borderValue=0) > 127 for a 0/1 source mask (OpenCV 5.x f32 lerp).
    pub fn warp_mask(&self, src: &Mask) -> Mask {
        let (sw, sh) = (src.w as i64, src.h as i64);
        let px = |x: i64, y: i64| -> f32 {
            if x >= 0 && y >= 0 && x < sw && y < sh && src.get(x as usize, y as usize) {
                255.0
            } else {
                0.0
            }
        };
        Mask::from_fn(VIRT_W, VIRT_H, |u, v| {
            let (x, y) = (self.mx[v * VIRT_W + u], self.my[v * VIRT_W + u]);
            let (x0, y0) = (x.floor(), y.floor());
            let (fx, fy) = (x - x0, y - y0);
            let (sx, sy) = (x0 as i64, y0 as i64);
            let (p00, p01, p10, p11) = (px(sx, sy), px(sx + 1, sy), px(sx, sy + 1), px(sx + 1, sy + 1));
            let t = p00 + (p01 - p00) * fx;
            let b = p10 + (p11 - p10) * fx;
            (t + (b - t) * fy).round_ties_even().clamp(0.0, 255.0) > 127.0
        })
    }
}

/// Ground evidence of one wide detection (what n2_det_all.py stores next to the box).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ground {
    pub bl: Option<[f64; 2]>, // bottom-left corner on the road: (x fwd from the camera, y right)
    pub bc: Option<[f64; 2]>,
    pub br: Option<[f64; 2]>,
    pub gsize: Option<[f64; 2]>, // side views: ground point from the box height (car 1.5 m)
    pub wbox: Option<[f64; 4]>,  // side views: bottom corners mapped back to the raw wide frame
}

/// Everything derived from one calibration: camera, side views and the band mask of each view.
#[derive(Clone, Debug)]
pub struct WideGeometry {
    pub cam: WCam,
    pub left: Virt,
    pub right: Virt,
    pub mask_raw: Mask,
    pub mask_left: Mask,
    pub mask_right: Mask,
}
impl WideGeometry {
    pub fn new(calib: &Calib) -> WideGeometry {
        WideGeometry::with_lens(calib, LENS)
    }
    pub fn with_lens(calib: &Calib, lens: Lens) -> WideGeometry {
        let cam = WCam::new(calib, lens);
        let (left, right) = (Virt::new(&cam, -VIRT_YAW_DEG), Virt::new(&cam, VIRT_YAW_DEG));
        let mask_raw = Mask::band();
        let (mask_left, mask_right) = (left.warp_mask(&mask_raw), right.warp_mask(&mask_raw));
        WideGeometry { cam, left, right, mask_raw, mask_left, mask_right }
    }
    pub fn mask(&self, view: View) -> &Mask {
        match view {
            View::Raw => &self.mask_raw,
            View::SideL => &self.mask_left,
            View::SideR => &self.mask_right,
        }
    }
    /// box -> ground points (bottom corners; side views also the size-based point and the raw-frame span).
    pub fn ground(&self, d: &WideDet) -> Ground {
        let [x1, y1, x2, y2] = d.bbox;
        let us = [x1, (x1 + x2) / 2.0, x2];
        let vw = match d.view {
            View::Raw => {
                let g = us.map(|u| self.cam.ground(self.cam.ray(u, y2)));
                return Ground { bl: g[0], bc: g[1], br: g[2], gsize: None, wbox: None };
            }
            View::SideL => &self.left,
            View::SideR => &self.right,
        };
        let g = us.map(|u| self.cam.ground(vw.ray(u, y2)));
        let dir = vw.ray((x1 + x2) / 2.0, (y1 + y2) / 2.0);
        let fwd = [vw.rcv[0][2], vw.rcv[1][2], vw.rcv[2][2]];
        let cosv = dir[0] * fwd[0] + dir[1] * fwd[1] + dir[2] * fwd[2];
        let zh = vw.f * CAR_H / (y2 - y1).max(1.0);
        let rng = zh / cosv.max(0.2);
        let (a, b) = (vw.to_wide(x1, y2), vw.to_wide(x2, y2));
        Ground { bl: g[0], bc: g[1], br: g[2], gsize: Some([rng * dir[0], rng * dir[1]]), wbox: Some([a.0, a.1, b.0, b.1]) }
    }
}
