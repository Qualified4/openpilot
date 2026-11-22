//! Stationary roadside rows and their fixed spatial support in the legacy
//! selector. Suppression changes drawings only.

use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};

use super::{
    completed_windows, in_domain, tuple_lt, update_windows, Curve, Model, Point, Window, Xy,
};
use crate::model_v2::numpy_interp;
use crate::pymath::{py_max, py_min, py_sum};

struct Anchor {
    xy: Xy,
    side: usize,
    first: i64,
    last: i64,
    count: usize,
    distance: f64,
    static_first: Option<i64>,
    static_last: i64,
    static_count: usize,
    mean: Xy,
    spread: Xy,
    windows: VecDeque<Window>,
    jitter: bool,
}

impl Anchor {
    fn new(point: Point, side: usize, frame: i64, distance: f64) -> Self {
        Self {
            xy: (point.x, point.y),
            side,
            first: frame,
            last: frame,
            count: 0,
            distance,
            static_first: None,
            static_last: -1,
            static_count: 0,
            mean: (0., 0.),
            spread: (0., 0.),
            windows: VecDeque::new(),
            jitter: false,
        }
    }

    fn observe(&mut self, p: Point, frame: i64, distance: f64, still: bool) {
        self.last = frame;
        self.count += 1;
        self.distance = distance;
        if still {
            self.static_first.get_or_insert(frame);
            self.static_last = frame;
            self.static_count += 1;
        }
        let (ex, ey) = (p.x - self.xy.0, p.y - self.xy.1);
        let (mx, my) = &mut self.mean;
        *mx = py_min(
            0.25,
            py_max(
                -0.25,
                *mx + py_min(0.025, py_max(-0.025, (ex - *mx) * 0.125)),
            ),
        );
        *my = py_min(
            0.15,
            py_max(
                -0.15,
                *my + py_min(0.015, py_max(-0.015, (ey - *my) * 0.125)),
            ),
        );
        let (sx, sy) = &mut self.spread;
        *sx = py_min(0.4, *sx * 0.875 + (ex - *mx).abs() * 0.125);
        *sy = py_min(0.2, *sy * 0.875 + (ey - *my).abs() * 0.125);
        update_windows(&mut self.windows, frame, (ex, ey));
        self.jitter = self.count >= 8
            && frame - self.first >= 60
            && completed_windows(&self.windows).is_some_and(|centers| {
                let [min_x, max_x, min_y, max_y] = centers.iter().fold(
                    [
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                    ],
                    |[min_x, max_x, min_y, max_y], &(x, y)| {
                        [
                            py_min(min_x, x),
                            py_max(max_x, x),
                            py_min(min_y, y),
                            py_max(max_y, y),
                        ]
                    },
                );
                max_x - min_x <= 0.15
                    && max_y - min_y <= 0.1
                    && centers
                        .iter()
                        .all(|&(x, y)| x.abs() <= 0.35 && y.abs() <= 0.25)
            });
    }

    fn matches(&self, p: Point, moving_noise: bool) -> bool {
        let (x, y) = self.xy;
        (p.x - (x + if moving_noise { self.mean.0 } else { 0. })).abs()
            <= if moving_noise {
                py_min(0.8, 0.5 + 2. * self.spread.0)
            } else {
                0.5
            }
            && (p.y - (y + if moving_noise { self.mean.1 } else { 0. })).abs()
                <= if moving_noise {
                    py_min(0.4, 0.25 + 2. * self.spread.1)
                } else {
                    0.25
                }
    }

    fn seen(&self) -> i64 {
        if self.jitter {
            self.last
        } else {
            self.static_last
        }
    }
}

struct Region {
    side: usize,
    start: i64,
    last: i64,
    anchors: Vec<Xy>,
    slope: f64,
    intercept: f64,
    extended: bool,
}

impl Region {
    fn extend(&mut self, row: &[Xy], distance: f64) {
        let mut available = self.anchors.clone();
        let mut pairs = Vec::new();
        let mut extra = Vec::new();
        for &(x, y) in row {
            if let Some(i) = nearest_anchor(&available, x, y) {
                let (a, b) = available.remove(i);
                pairs.push((x - a, y - b));
            } else {
                extra.push((x, y));
            }
        }
        if pairs.len() < 2 || extra.is_empty() {
            return;
        }
        let dx = py_sum(pairs.iter().map(|p| p.0)).unwrap_or(0.) / pairs.len() as f64;
        let dy = py_sum(pairs.iter().map(|p| p.1)).unwrap_or(0.) / pairs.len() as f64;
        let mut anchors = self.anchors.clone();
        for (x, y) in extra.into_iter().map(|(x, y)| (x - dx, y - dy)) {
            let gap = anchors
                .iter()
                .map(|a| (x - a.0).abs())
                .fold(f64::INFINITY, py_min);
            if (x < anchors[0].0 || x > anchors.last().unwrap().0)
                && (0.6..=3.).contains(&gap)
                && anchors.len() < 16
            {
                anchors.push((x, y));
                anchors.sort_by(xy_order);
            }
        }
        anchors.retain(|a| a.0 >= distance - 3.);
        if anchors.len() != self.anchors.len() && !anchors.is_empty() {
            self.anchors = anchors;
            self.extended = true;
        }
    }

    fn suppresses(&self, frame: i64, x: f64, y: f64) -> bool {
        if self.last - self.start < 30
            || !(0..=15).contains(&(frame - self.last))
            || !(self.anchors[0].0 - 0.75 <= x && x <= self.anchors.last().unwrap().0 + 0.75)
        {
            return false;
        }
        let expected = if self.extended {
            let curve: Curve = self.anchors.iter().copied().unzip();
            numpy_interp(&curve, x)
        } else {
            self.slope * x + self.intercept
        };
        (y - expected).abs() <= 0.35
    }
}

fn xy_order(a: &Xy, b: &Xy) -> Ordering {
    if tuple_lt(*a, *b) {
        Ordering::Less
    } else if tuple_lt(*b, *a) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn nearest_boundary(
    x: f64,
    y: f64,
    curves: [Option<&Curve>; 2],
    limit: f64,
) -> Option<(f64, usize)> {
    let mut best = None;
    for (i, curve) in curves.into_iter().enumerate() {
        if let Some(curve) = curve.filter(|c| in_domain(c, x)) {
            let distance = (y + numpy_interp(curve, x)).abs();
            if distance <= limit && best.is_none_or(|(d, _)| distance < d) {
                best = Some((distance, i));
            }
        }
    }
    best
}

fn nearest_anchor(points: &[Xy], x: f64, y: f64) -> Option<usize> {
    points
        .iter()
        .enumerate()
        .filter_map(|(i, &(a, b))| {
            let (dx, dy) = ((x - a).abs(), (y - b).abs());
            (dx <= 0.75 && dy <= 0.35).then_some((i, dx + dy))
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
}

fn support(points: &[Xy], anchors: &[Xy]) -> usize {
    let mut available = anchors.to_vec();
    let mut count = 0;
    for &(x, y) in points {
        if let Some(i) = nearest_anchor(&available, x, y) {
            available.remove(i);
            count += 1;
        }
    }
    count
}

fn gaps(row: &[Xy]) -> (f64, f64, f64) {
    let steps = row.windows(2).map(|p| p[1].0 - p[0].0);
    let min = steps.clone().fold(f64::INFINITY, py_min);
    let max = steps.clone().fold(f64::NEG_INFINITY, py_max);
    let sum = py_sum(steps).unwrap_or(0.);
    (min, max, sum)
}

#[derive(Default)]
pub(super) struct Structure {
    anchors: Vec<Anchor>,
    rows: Vec<(usize, Vec<usize>)>,
    map_frame: Option<i64>,
    map_distance: f64,
    map_reader: Option<u64>,
    region_reader: Option<u64>,
    regions: Vec<Region>,
}

impl Structure {
    pub fn clear(&mut self) {
        self.regions.clear();
        self.region_reader = None;
        self.map_frame = None;
        self.map_reader = None;
        self.anchors.clear();
        self.rows.clear();
    }

    #[allow(clippy::too_many_arguments)]
    fn update_map(
        &mut self,
        frame: i64,
        model: &Model,
        curves: [Option<&Curve>; 2],
        distance: f64,
        reverse: bool,
        usable: bool,
        raw: &[Point],
        raw_generation: u64,
        hold: i64,
    ) {
        self.rows.clear();
        let delta = self.map_frame.map_or(0, |last| frame - last);
        let travel = distance - self.map_distance;
        let rate = model.yaw.filter(|r| r.is_finite() && r.abs() <= 0.2);
        let Some(rate) = rate.filter(|_| {
            usable
                && !reverse
                && !model.changing
                && curves.iter().any(Option::is_some)
                && (0..=15).contains(&delta)
                && (0. ..=5.).contains(&travel)
        }) else {
            self.anchors.clear();
            self.map_frame = None;
            self.map_reader = None;
            self.map_distance = distance;
            return;
        };
        let yaw = -rate * delta as f64 * 0.01;
        let (c, s) = (yaw.cos(), yaw.sin());
        let (dx, dy) = (travel * (yaw * 0.5).cos(), travel * (yaw * 0.5).sin());
        let rot = |(x, y): Xy| (c * x + s * y, -s * x + c * y);
        for a in &mut self.anchors {
            a.xy = rot((a.xy.0 - dx, a.xy.1 - dy));
            a.mean = rot(a.mean);
            for w in &mut a.windows {
                (w.x, w.y) = rot((w.x, w.y));
            }
        }
        self.map_frame = Some(frame);
        self.map_distance = distance;
        self.anchors.retain(|a| {
            (0..=300).contains(&(frame - a.last))
                && distance - a.distance <= 15.
                && (-3. ..=50.).contains(&a.xy.0)
                && a.xy.1.abs() <= 6.
        });
        if self.map_reader != Some(raw_generation) {
            let mut matched = HashSet::new();
            for &p in raw {
                if !p.front
                    || !p.measured
                    || ![p.x, p.y, p.v_lead, p.vy].iter().all(|v| v.is_finite())
                    || !(0.2..=50.).contains(&p.x)
                    || !(0.2..=6.).contains(&p.y.abs())
                    || p.v_lead.abs() > 2.5
                    || p.vy.abs() > 0.6
                {
                    continue;
                }
                let Some((gap, side)) = nearest_boundary(p.x, p.y, curves, 1.5) else {
                    continue;
                };
                let strict = p.v_lead.abs() <= 0.6 && gap <= 0.75;
                let result = self
                    .anchors
                    .iter()
                    .enumerate()
                    .filter(|(i, a)| {
                        !matched.contains(i)
                            && a.side == side
                            && (strict || a.static_count == 0)
                            && a.matches(p, !strict)
                    })
                    .map(|(i, a)| {
                        (
                            i,
                            strict && a.static_count == 0,
                            (p.x - a.xy.0).abs() + (p.y - a.xy.1).abs(),
                        )
                    })
                    .min_by(|a, b| {
                        a.1.cmp(&b.1)
                            .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
                    });
                let i = if let Some((i, _, _)) = result {
                    i
                } else if self.anchors.len() < 96 {
                    self.anchors.push(Anchor::new(p, side, frame, distance));
                    self.anchors.len() - 1
                } else {
                    continue;
                };
                self.anchors[i].observe(p, frame, distance, strict);
                matched.insert(i);
            }
        }
        self.map_reader = Some(raw_generation);
        for (side, curve) in curves
            .into_iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|c| (i, c)))
        {
            let mut keys: Vec<usize> = self
                .anchors
                .iter()
                .enumerate()
                .filter_map(|(i, a)| {
                    if a.side != side {
                        return None;
                    }
                    let error = (a.xy.1 + numpy_interp(curve, a.xy.0)).abs();
                    let still = a.static_count >= 4
                        && a.static_first.is_some_and(|f| a.static_last - f >= 30)
                        && error <= 0.75;
                    (still || (a.jitter && error <= 1.5)).then_some(i)
                })
                .collect();
            keys.sort_by(|a, b| {
                self.anchors[*a]
                    .xy
                    .0
                    .partial_cmp(&self.anchors[*b].xy.0)
                    .unwrap_or(Ordering::Equal)
            });
            let mut distinct: Vec<usize> = Vec::new();
            for key in keys {
                let point = &self.anchors[key];
                if let Some(last) = distinct.last_mut() {
                    let old = &self.anchors[*last];
                    if point.xy.0 - old.xy.0 < 0.6 {
                        if (point.count, point.last) > (old.count, old.last) {
                            *last = key;
                        }
                        continue;
                    }
                }
                distinct.push(key);
            }
            for first in 0..distinct.len().saturating_sub(3) {
                let mut row = distinct[first..first + 4].to_vec();
                let xy: Vec<_> = row.iter().map(|&i| self.anchors[i].xy).collect();
                let (min, max, sum) = gaps(&xy);
                if min < 0.6 || max > 3. || max - min > 0.75 || sum < 5.5 {
                    continue;
                }
                let (x0, y0) = xy[0];
                let slope = (xy[3].1 - y0) / sum;
                let error = |(x, y): Xy| (y - y0 - slope * (x - x0)).abs();
                if slope.abs() > 0.4 || xy.iter().any(|&p| error(p) > 0.25) {
                    continue;
                }
                let spacing = sum / 3.;
                let mut missed = false;
                for &i in &distinct[first + 4..] {
                    let p = self.anchors[i].xy;
                    let gap = p.0 - self.anchors[*row.last().unwrap()].xy.0;
                    let missing = gap > 1.5 * spacing;
                    if row.len() >= 16 || gap > 2. * spacing + 0.35 || (missing && missed) {
                        break;
                    }
                    if py_min((gap - spacing).abs(), (gap - 2. * spacing).abs()) > 0.35
                        || error(p) > 0.25
                    {
                        break;
                    }
                    row.push(i);
                    missed |= missing;
                }
                if row
                    .iter()
                    .any(|&i| (0..=hold).contains(&(frame - self.anchors[i].seen())))
                {
                    self.rows.push((side, row));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        frame: i64,
        model: &Model,
        distance: f64,
        reverse: bool,
        usable: bool,
        stopped: bool,
        raw: &[Point],
        raw_generation: u64,
        view: &[Point],
        view_generation: u64,
        hold: i64,
    ) {
        let curves = model.curves(0.3);
        self.update_map(
            frame,
            model,
            curves,
            distance,
            reverse,
            usable,
            raw,
            raw_generation,
            hold,
        );
        if !usable
            || !model.present
            || model.changing
            || (!stopped && model.yaw.is_none_or(|r| !r.is_finite() || r.abs() > 0.03))
            || curves.iter().all(Option::is_none)
        {
            self.regions.clear();
            self.region_reader = None;
            return;
        }
        self.regions.retain(|region| curves[region.side].is_some());
        if self.region_reader == Some(view_generation) {
            return;
        }
        self.region_reader = Some(view_generation);
        let mut groups = [Vec::new(), Vec::new()];
        for p in view {
            if p.front
                && p.measured
                && (0.2..=50.).contains(&p.x)
                && (0.2..=6.).contains(&p.y.abs())
            {
                if let Some((_, side)) = nearest_boundary(p.x, p.y, curves, 1.5) {
                    groups[side].push((distance + p.x, p.y));
                }
            }
        }
        self.regions.retain_mut(|region| {
            if !(0..=15).contains(&(frame - region.last)) {
                return false;
            }
            if support(&groups[region.side], &region.anchors) >= 2 {
                region.last = frame;
            }
            true
        });
        for (side, mut points) in groups.into_iter().enumerate() {
            points.sort_by(xy_order);
            for first in 0..points.len() {
                for last in first + 2..points.len() {
                    let row = &points[first..=last];
                    let length = row[row.len() - 1].0 - row[0].0;
                    if length < 4. - 1e-5 {
                        continue;
                    }
                    let (min, max, _) = gaps(row);
                    if min < 0.6 || max > 3. || max - min > 0.75 {
                        break;
                    }
                    let slope = (row[row.len() - 1].1 - row[0].1) / length;
                    let intercept = row[0].1 - slope * row[0].0;
                    if slope.abs() > 0.4
                        || row
                            .iter()
                            .any(|&(x, y)| (y - slope * x - intercept).abs() > 0.25)
                    {
                        continue;
                    }
                    if let Some(region) = self
                        .regions
                        .iter_mut()
                        .find(|r| r.side == side && support(row, &r.anchors) >= 2)
                    {
                        region.extend(row, distance);
                        continue;
                    }
                    self.regions.push(Region {
                        side,
                        start: frame,
                        last: frame,
                        anchors: row.iter().take(16).copied().collect(),
                        slope,
                        intercept,
                        extended: false,
                    });
                    break;
                }
            }
        }
    }

    pub fn suppresses(
        &self,
        side: usize,
        frame: i64,
        sample: Point,
        stamp: i64,
        distance: f64,
        hold: i64,
    ) -> bool {
        let mapped = (0..=hold).contains(&(frame - stamp))
            && sample.vy.abs() <= 0.6
            && self
                .rows
                .iter()
                .filter(|(row_side, _)| *row_side == side)
                .any(|(_, row)| {
                    row.iter().any(|&i| {
                        let a = &self.anchors[i];
                        (frame - a.static_last <= hold
                            && sample.v_lead.abs() <= 0.6
                            && a.matches(sample, false))
                            || (a.jitter
                                && frame - a.last <= hold
                                && sample.v_lead.abs() <= 2.5
                                && a.matches(sample, true))
                    })
                });
        mapped
            || self
                .regions
                .iter()
                .any(|r| r.side == side && r.suppresses(frame, distance + sample.x, sample.y))
    }
}
