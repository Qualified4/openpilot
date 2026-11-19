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
