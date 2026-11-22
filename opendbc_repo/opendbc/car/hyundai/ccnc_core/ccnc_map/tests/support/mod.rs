use ccnc_map::input::model_v2::{Line, Model};

const GRID: [f32; 10] = [0., 5., 10., 15., 20., 30., 40., 50., 60., 80.];
const BOUNDARIES: [f64; 6] = [8.75, 5.25, 1.75, -1.75, -5.25, -8.75];

pub fn model(y: f64, heading: f64, far_lag: f64) -> Model {
    let right = BOUNDARIES.iter().position(|&b| b < y).unwrap();
    let lane_lines = BOUNDARIES[right - 2..right + 2]
        .iter()
        .map(|&boundary| Line {
            x: GRID.to_vec(),
            y: GRID
                .iter()
                .map(|&x| {
                    let x = f64::from(x);
                    -(boundary - y - x * heading.tan()
                        + far_lag * heading.tan() * (x - 20.).max(0.)) as f32
                })
                .collect(),
        })
        .collect();
    Model {
        lane_lines,
        lane_line_probs: vec![0.9; 4],
        road_edges: Vec::new(),
    }
}

pub fn ego_path(t: f64) -> (f64, f64, f64) {
    let phase = (t - 3.) / 4.;
    let angle = std::f64::consts::PI * phase.clamp(0., 1.);
    let y = -1.75 * (1. - angle.cos());
    let speed = -1.75 * std::f64::consts::PI / 4.;
    let vy = speed * angle.sin();
    let ay = if (0. ..1.).contains(&phase) {
        speed * std::f64::consts::PI / 4. * angle.cos()
    } else {
        0.
    };
    (y, (vy / 25.).atan(), ay / 25. / (1. + (vy / 25.).powi(2)))
}
