//! Wide camera candidates (port of pipeline.py): box -> side-lane candidate with the own-car band post-processing in the
//! fixed order, the display filters, and the two-cars-on-one-side split.
use crate::band::{band_stats, Mask};
use crate::geom::Ground;
use crate::{round2, Lane, Track, View, WideDet, CAM2B, HALFW, LANE_IN, LANE_OUT};

pub const CUT_SCORE_MIN: f64 = 0.5;
pub const INNER_LINE_PMIN: f64 = 0.3;
pub const SAME_CAR_OVERLAP: f64 = 0.3; // raw-image spans overlapping more than this share of the narrower one = same car
pub const REAR_GAP: f64 = 1.5; // the front car must be at least this much farther along than the rear car

/// One detection as a possible side-lane car. x: metres ahead of the front bumper, yc: lateral centre, right positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cand {
    pub view: View,
    pub class: i32,
    pub score: f64,
    pub x: f64,
    pub yc: f64,
    pub bbox: [f64; 4],
    pub cut: bool,               // bottom on the own-car band: x is not a bottom distance
    pub xs: Option<f64>,         // side views: size-based distance
    pub ys: Option<f64>,         // side views: size-based lateral
    pub wbox: Option<[f64; 4]>,  // side views: bottom corners in the raw wide frame
    pub inside: f64,             // share of the box inside the band
    pub x_bottom: Option<f64>,   // cut boxes: the bottom distance that was replaced
    pub near03: bool,            // raw cut box: only '0-3 m close' is known, x is the 1.5 m placeholder
    pub lane: Lane,
}
impl Cand {
    /// A candidate with only position fields set, for callers that build candidates themselves.
    pub fn at(view: View, x: f64, yc: f64) -> Cand {
        Cand {
            view,
            class: 2,
            score: 0.8,
            x,
            yc,
            bbox: [0.0; 4],
            cut: false,
            xs: None,
            ys: None,
            wbox: None,
            inside: 0.0,
            x_bottom: None,
            near03: false,
            lane: lane_of(yc),
        }
    }
}

pub fn lane_of(yc: f64) -> Lane {
    let a = yc.abs();
    if a < LANE_IN {
        Lane::Ego
    } else if a < LANE_OUT {
        if yc < 0.0 {
            Lane::Left
        } else {
            Lane::Right
        }
    } else {
        Lane::Outside
    }
}

/// detection + ground points -> candidate or None. Order is fixed: band drop FIRST, then the cut-box exception.
/// `post = false` skips both (the candidate keeps its bottom distance).
pub fn wide_candidate(d: &WideDet, g: &Ground, mask: &Mask, post: bool) -> Option<Cand> {
    let sy = g.bc?;
    let left = sy[1] < 0.0;
    let inner = (if left { g.br } else { g.bl }).unwrap_or(sy); // inner bottom corner
    let mut o = Cand {
        view: d.view,
        class: d.class,
        score: d.score,
        x: round2(inner[0] - CAM2B),
        yc: round2(inner[1] + if left { -HALFW } else { HALFW }),
        bbox: d.bbox,
        cut: false,
        xs: None,
        ys: None,
        wbox: g.wbox,
        inside: 0.0,
        x_bottom: None,
        near03: false,
        lane: Lane::Ego,
    };
    if let Some(gs) = g.gsize {
        let k = if d.class == 5 || d.class == 7 { 2.0 } else { 1.0 }; // bus/truck ~3 m tall
        o.xs = Some(round2(k * gs[0] - CAM2B));
        o.ys = Some(round2(k * gs[1]));
    }
    let (inside, touch) = band_stats(mask, &d.bbox);
    o.inside = inside;
    if post {
        if inside >= 0.5 {
            return None; // (1) own-car band: dashboard / bonnet reflection
        }
        if touch {
            // (2) cut box
            o.cut = true;
            o.x_bottom = Some(o.x);
            match (o.xs, o.ys) {
                (Some(xs), Some(ys)) => (o.x, o.yc) = (xs, ys), // side view: size-based distance and lateral
                _ => (o.x, o.near03) = (1.5, true),             // raw: only '0-3 m close'
            }
        }
    }
    o.lane = lane_of(o.yc);
    Some(o)
}

pub fn valid(c: &Cand) -> bool {
    let w = if c.view == View::Raw { 1344.0 } else { 512.0 };
    (c.cut || c.x >= 0.5) && (c.bbox[2] - c.bbox[0]) < 0.6 * w
}
pub fn near(c: &Cand) -> bool {
    matches!(c.lane, Lane::Left | Lane::Right) && -1.5 <= c.x && c.x <= 6.0 && valid(c)
}
/// side-view bottom distance and size distance agree (cut boxes already use the size distance).
pub fn consistent(c: &Cand) -> bool {
    c.cut || c.xs.map_or(true, |xs| (c.x - xs).abs() <= 2.5)
}

/// (chosen, side-view pick, raw pick): side view first, raw fallback; nearest |x|, the first one on a tie.
pub fn pick_side_both<'a>(cands: &[&'a Cand], side: Lane) -> (Option<&'a Cand>, Option<&'a Cand>, Option<&'a Cand>) {
    let pick = |raw: bool| {
        let mut best: Option<&'a Cand> = None;
        for &c in cands {
            if near(c) && c.lane == side && (c.view == View::Raw) == raw && best.map_or(true, |b| c.x.abs() < b.x.abs()) {
                best = Some(c);
            }
        }
        best
    };
    let (virt, raw) = (pick(false), pick(true));
    (virt.or(raw), virt, raw)
}
pub fn pick_side<'a>(cands: &[&'a Cand], side: Lane) -> Option<&'a Cand> {
    pick_side_both(cands, side).0
}

// ---------------------------------------------------------------- oncoming / parked filter evidence
/// radar tracks around the camera car. The camera x is the car's rear end, radar points sit ~0-4 m further along the
/// body, hence the +2 m centre. Track yRel is left positive, camera yc right positive.
pub fn radar_near<'a>(tracks: &'a [Track], c: &'a Cand) -> impl Iterator<Item = &'a Track> + 'a {
    tracks.iter().filter(move |t| (t.d_rel - c.x - 2.0).abs() <= 4.5 && (-t.y_rel - c.yc).abs() <= 1.5)
}
/// any nearby radar track moving in either direction (|vRel + vEgo| > 1.5).
pub fn radar_moving_near(tracks: &[Track], c: &Cand, v_ego: f64) -> bool {
    radar_near(tracks, c).any(|t| (t.v_rel + v_ego).abs() > 1.5)
}
/// nearby radar track coming toward us (ground speed vRel + vEgo < -1.5): oncoming traffic while we stand.
pub fn radar_oncoming_near(tracks: &[Track], c: &Cand, v_ego: f64) -> bool {
    radar_near(tracks, c).any(|t| t.v_rel + v_ego < -1.5)
}
/// cut box with a low score: building walls and car-wash brushes next to the bonnet.
pub fn weak_cut(c: &Cand) -> bool {
    c.cut && c.score < CUT_SCORE_MIN
}
/// no lane line on that side of our lane (model prob < 0.3) AND no radar point near the camera car.
pub fn no_road_context(lprob: Option<&[f64]>, tracks: &[Track], c: &Cand) -> bool {
    let Some(lp) = lprob.filter(|lp| lp.len() >= 4) else { return false };
    let inner = if c.yc < 0.0 { lp[1] } else { lp[2] };
    inner < INNER_LINE_PMIN && radar_near(tracks, c).next().is_none()
}
/// np.interp: linear interpolation on ascending xs, clamped at both ends.
fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n == 0 {
        return f64::NAN;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let j = xs[..n].partition_point(|&v| v <= x);
    let (x0, x1, y0, y1) = (xs[j - 1], xs[j], ys[j - 1], ys[j]);
    (y1 - y0) / (x1 - x0) * (x - x0) + y0
}
/// Evidence only (not a display filter): the car centre is beyond the model's outer lane line (0: left-left,
/// 3: right-right) at the car's distance. lanes: 4 x (xs, ys) in the calib frame (x from the camera, y right).
pub fn outside_outer_line(lanes: &[(Vec<f64>, Vec<f64>)], lprob: Option<&[f64]>, c: &Cand) -> bool {
    let i = if c.yc < 0.0 { 0 } else { 3 };
    let Some(lp) = lprob.filter(|lp| lp.len() > i && lanes.len() > i) else { return false };
    if lp[i] < 0.3 {
        return false;
    }
    let ly = interp(c.x + CAM2B + 2.0, &lanes[i].0, &lanes[i].1);
    if i == 0 {
        c.yc < ly - 0.5
    } else {
        c.yc > ly + 0.5
    }
}
/// Evidence only: horizontal motion of the box centre between processed frames, px (same view only).
pub fn image_motion(prev: Option<&Cand>, c: Option<&Cand>) -> Option<f64> {
    let (p, c) = (prev?, c?);
    (p.view == c.view).then(|| (c.bbox[0] + c.bbox[2]) / 2.0 - (p.bbox[0] + p.bbox[2]) / 2.0)
}

// ---------------------------------------------------------------- display filters
/// Set of filters that veto a candidate. Names sort like the Python list: no_road, radar_oncoming, weak_cut.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Filters(pub u8);
impl Filters {
    pub const NO_ROAD: Filters = Filters(1); // (b) no road context
    pub const RADAR_ONCOMING: Filters = Filters(2); // (a) oncoming track
    pub const WEAK_CUT: Filters = Filters(4); // (c) weak cut box on the bonnet line
    /// vetoes the LF/RF and LR/RR display.
    pub const DISPLAY: Filters = Filters(1 | 2);
    /// decides whether a candidate counts as one of 'two cars on one side'. weak_cut is here only: on the display it
    /// removed real side cars (tried and reverted 2026-11-22).
    pub const CAR: Filters = Filters(1 | 2 | 4);
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub fn contains(self, o: Filters) -> bool {
        self.0 & o.0 == o.0
    }
    pub fn names(self) -> Vec<&'static str> {
        [(Filters::NO_ROAD, "no_road"), (Filters::RADAR_ONCOMING, "radar_oncoming"), (Filters::WEAK_CUT, "weak_cut")]
            .iter()
            .filter(|(f, _)| self.contains(*f))
            .map(|(_, n)| *n)
            .collect()
    }
}
/// the filters of `names` that veto candidate c; empty = passes.
pub fn filter_hits(c: &Cand, v_ego: f64, tracks: &[Track], lprob: Option<&[f64]>, names: Filters) -> Filters {
    let mut hit = 0;
    if names.contains(Filters::RADAR_ONCOMING) && radar_oncoming_near(tracks, c, v_ego) {
        hit |= Filters::RADAR_ONCOMING.0;
    }
    if names.contains(Filters::NO_ROAD) && no_road_context(lprob, tracks, c) {
        hit |= Filters::NO_ROAD.0;
    }
    if names.contains(Filters::WEAK_CUT) && weak_cut(c) {
        hit |= Filters::WEAK_CUT.0;
    }
    Filters(hit)
}
pub fn display_filters(c: &Cand, v_ego: f64, tracks: &[Track], lprob: Option<&[f64]>) -> Filters {
    filter_hits(c, v_ego, tracks, lprob, Filters::DISPLAY)
}
/// may c count as one of 'two cars on one side'? CAR filters and the side-view distance consistency.
pub fn counts_as_car(c: &Cand, v_ego: f64, tracks: &[Track], lprob: Option<&[f64]>) -> bool {
    consistent(c) && filter_hits(c, v_ego, tracks, lprob, Filters::CAR).is_empty()
}

// ---------------------------------------------------------------- two cars on one side -> LF/RF + LR/RR
/// horizontal extent of the box in the raw wide frame: raw box, or the side-view bottom corners mapped back.
pub fn raw_span(c: &Cand) -> Option<(f64, f64)> {
    if c.view == View::Raw {
        return Some((c.bbox[0], c.bbox[2]));
    }
    c.wbox.map(|w| (w[0].min(w[2]), w[0].max(w[2])))
}
pub fn same_car(a: &Cand, b: &Cand) -> bool {
    let (Some(sa), Some(sb)) = (raw_span(a), raw_span(b)) else { return (a.x - b.x).abs() < 2.5 };
    let ov = sa.1.min(sb.1) - sa.0.max(sb.0);
    ov > SAME_CAR_OVERLAP * (sa.1 - sa.0).min(sb.1 - sb.0).max(1.0)
}
/// greedy single-link grouping of the candidates of one side into physical cars.
pub fn car_groups<'a>(cands: &[&'a Cand]) -> Vec<Vec<&'a Cand>> {
    let mut groups: Vec<Vec<&'a Cand>> = Vec::new();
    for &c in cands {
        let mut merged = Vec::new();
        let mut rest = Vec::with_capacity(groups.len() + 1);
        for g in groups {
            if g.iter().any(|o| same_car(c, o)) {
                merged.extend(g);
            } else {
                rest.push(g);
            }
        }
        merged.push(c);
        rest.push(merged);
        groups = rest;
    }
    groups
}
/// -> (front, rear) candidates for LF/RF and LR/RR. One car (or cars that are not clearly one behind the other) ->
/// (pick_side, None). Two or more cars -> rear = the outermost car in the raw image (beside/behind ego), front = the
/// nearest of the others, provided the front one is at least REAR_GAP farther along and is not a cut box. Only
/// candidates passing `keep` are counted. The rule-E condition on the front car is `SplitGate`.
pub fn split_side<'a>(cands: &[&'a Cand], side: Lane, keep: impl Fn(&Cand) -> bool) -> (Option<&'a Cand>, Option<&'a Cand>) {
    let single = pick_side(cands, side);
    let near_c: Vec<&Cand> = cands.iter().copied().filter(|c| near(c) && c.lane == side && keep(c)).collect();
    let mut groups = car_groups(&near_c);
    if groups.len() < 2 {
        return (single, None);
    }
    let centre = |g: &Vec<&Cand>| {
        let (mut sum, mut n) = (0.0, 0);
        for (a, b) in g.iter().filter_map(|c| raw_span(c)) {
            sum += (a + b) / 2.0;
            n += 1;
        }
        let u = if n > 0 { sum / n as f64 } else { 672.0 };
        if side == Lane::Left {
            u
        } else {
            -u // smaller = more outer (image edge = beside/behind ego)
        }
    };
    groups.sort_by(|a, b| centre(a).total_cmp(&centre(b)));
    let rear = pick_side(&groups[0], side);
    let inner: Vec<&Cand> = groups[1..].iter().flatten().copied().filter(|c| !c.cut).collect();
    let front = pick_side(&inner, side);
    match (front, rear) {
        (Some(f), Some(r)) if f.x - r.x >= REAR_GAP => (front, rear),
        _ => (single, None),
    }
}
/// LR/RR distance (m behind the front bumper, >= 0) of a rear camera car; a car beside ego is 0.
pub fn rear_distance(c: &Cand) -> f64 {
    (-c.x).max(0.0)
}
