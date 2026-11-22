//! CPython float operations that must give the same bits as the Python code they replace.

const DBL_MIN: f64 = f64::MIN_POSITIVE;

fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    (x, (a - x) + b)
}

fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

/// frexp() exponent of a positive finite value.
fn frexp_exponent(x: f64) -> i32 {
    let biased = ((x.to_bits() >> 52) & 0x7ff) as i32;
    if biased == 0 {
        frexp_exponent(x * f64::from_bits(0x4350_0000_0000_0000)) - 54 // x * 2**54
    } else {
        biased - 1022
    }
}

/// ldexp(1.0, k) for -1074 <= k <= 1023.
fn power_of_two(k: i32) -> f64 {
    if k >= -1022 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (k + 1074))
    }
}

fn vector_norm(mut vec: [f64; 2], max: f64, found_nan: bool) -> f64 {
    if max.is_infinite() {
        return max;
    }
    if found_nan {
        return f64::NAN;
    }
    if max == 0.0 {
        return max;
    }
    let max_e = frexp_exponent(max);
    if max_e < -1023 {
        for x in &mut vec {
            *x /= DBL_MIN;
        }
        return DBL_MIN * vector_norm(vec, max / DBL_MIN, found_nan);
    }
    let scale = power_of_two(-max_e);
    let (mut csum, mut frac1, mut frac2) = (1.0, 0.0, 0.0);
    for x in vec {
        let x = x * scale;
        let pr = dl_mul(x, x);
        let sm = dl_fast_sum(csum, pr.0);
        csum = sm.0;
        frac1 += pr.1;
        frac2 += sm.1;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let pr = dl_mul(-h, h);
    let sm = dl_fast_sum(csum, pr.0);
    csum = sm.0;
    frac1 += pr.1;
    frac2 += sm.1;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

/// `math.hypot(x, y)`: a port of CPython 3.12 `math_hypot`/`vector_norm` (Modules/mathmodule.c), not libm hypot().
pub fn hypot(x: f64, y: f64) -> f64 {
    let (x, y) = (x.abs(), y.abs());
    let mut max = 0.0;
    for v in [x, y] {
        if v > max {
            max = v;
        }
    }
    vector_norm([x, y], max, x.is_nan() || y.is_nan())
}

/// Python `max(a, b)` for floats: the first argument unless the second is larger.
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python `min(a, b)` for floats: the first argument unless the second is smaller.
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `sum(values)` of floats in CPython 3.12 (Python/bltinmodule.c `builtin_sum_impl`): the int start 0 is added to
/// the first value (so -0.0 becomes 0.0), the rest are added with Neumaier's compensation, and the compensation
/// is added at the end only when it is nonzero and finite. None for no values (Python returns the int 0).
pub fn py_sum(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut values = values.into_iter();
    let mut total = 0.0 + values.next()?;
    let mut c = 0.0;
    for x in values {
        let t = total + x;
        if total.abs() >= x.abs() {
            c += (total - t) + x;
        } else {
            c += (x - t) + total;
        }
        total = t;
    }
    if c != 0.0 && c.is_finite() {
        total += c;
    }
    Some(total)
}

/// `math.degrees(x)`: CPython multiplies by its `radToDeg = 180.0 / Py_MATH_PI`, not by Rust's rounded constant.
pub fn degrees(x: f64) -> f64 {
    x * (180.0 / std::f64::consts::PI)
}

/// `math.radians(x)`: CPython's `x * degToRad`, `degToRad = Py_MATH_PI / 180.0`.
pub fn radians(x: f64) -> f64 {
    x * (std::f64::consts::PI / 180.0)
}

/// CPython `math_1` raises a domain error for a NaN result from a non-NaN input.
fn math_1(x: f64, result: f64) -> crate::radar::Result<f64> {
    if result.is_nan() && !x.is_nan() {
        return Err(crate::radar::Error::Value("math domain error"));
    }
    Ok(result)
}

pub fn acos(x: f64) -> crate::radar::Result<f64> {
    math_1(x, x.acos())
}

pub fn cos(x: f64) -> crate::radar::Result<f64> {
    math_1(x, x.cos())
}

/// CPython `round(x, digits)` rounds the binary float to the nearest decimal at
/// this precision (ties to even), then converts that decimal back to a float.
pub fn round_decimal(x: f64, digits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.digits$}").parse().expect("formatted finite float")
}
