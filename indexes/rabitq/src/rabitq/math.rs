use simsimd::SpatialSimilarity;

/// Compute the dot product between two vectors.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    f32::dot(a, b).unwrap_or(0.0) as f32
}

/// Compute the squared L2 norm of a vector.
#[inline]
pub fn l2_norm_sqr(v: &[f32]) -> f32 {
    dot(v, v)
}

/// Compute `a - b` element-wise.
#[inline]
pub fn subtract(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut out = vec![0.0f32; len];

    if len == 0 {
        return out;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked that AVX2 is available on this CPU.
            unsafe {
                x86::subtract_avx2(a, b, &mut out);
            }
            return out;
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: We just checked that SSE2 is available on this CPU.
            unsafe {
                x86::subtract_sse2(a, b, &mut out);
            }
            return out;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::subtract_neon(a, b, &mut out);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        subtract_scalar_into(a, b, &mut out);
    }

    out
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn subtract_scalar_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((dst, x), y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *dst = *x - *y;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn subtract_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = out.len();
        let mut i = 0usize;
        let chunks = len / 8;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i < chunks * 8 {
            let va = unsafe { _mm256_loadu_ps(a_ptr.add(i)) };
            let vb = unsafe { _mm256_loadu_ps(b_ptr.add(i)) };
            let diff = _mm256_sub_ps(va, vb);
            unsafe { _mm256_storeu_ps(out_ptr.add(i), diff) };
            i += 8;
        }

        while i < len {
            unsafe { *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i) };
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn subtract_sse2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = out.len();
        let mut i = 0usize;
        let chunks = len / 4;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i < chunks * 4 {
            let va = unsafe { _mm_loadu_ps(a_ptr.add(i)) };
            let vb = unsafe { _mm_loadu_ps(b_ptr.add(i)) };
            let diff = _mm_sub_ps(va, vb);
            unsafe { _mm_storeu_ps(out_ptr.add(i), diff) };
            i += 4;
        }

        while i < len {
            unsafe { *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i) };
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    pub fn subtract_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = out.len();
        let mut i = 0usize;
        let chunks = len / 4;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i < chunks * 4 {
            let va = unsafe { vld1q_f32(a_ptr.add(i)) };
            let vb = unsafe { vld1q_f32(b_ptr.add(i)) };
            let diff = vsubq_f32(va, vb);
            unsafe { vst1q_f32(out_ptr.add(i), diff) };
            i += 4;
        }

        while i < len {
            unsafe { *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i) };
            i += 1;
        }
    }
}
