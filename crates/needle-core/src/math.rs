//! Thin wrappers around libm for no_std float math.

#[inline(always)]
pub fn exp(x: f32) -> f32 {
    libm::expf(x)
}

#[inline(always)]
pub fn ln(x: f32) -> f32 {
    libm::logf(x)
}

#[inline(always)]
pub fn sin(x: f32) -> f32 {
    libm::sinf(x)
}

#[inline(always)]
pub fn cos(x: f32) -> f32 {
    libm::cosf(x)
}

#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    libm::tanhf(x)
}

#[inline(always)]
pub fn powf(base: f32, exp: f32) -> f32 {
    libm::powf(base, exp)
}

#[inline(always)]
pub fn round(x: f32) -> f32 {
    libm::roundf(x)
}

#[inline(always)]
pub fn abs(x: f32) -> f32 {
    libm::fabsf(x)
}
