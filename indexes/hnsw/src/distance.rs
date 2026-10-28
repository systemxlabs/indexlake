use simsimd::SpatialSimilarity;

/// Below this dimension the call overhead of the SIMD implementation is larger
/// than the work itself, so a small scalar kernel is faster.
const SIMD_MIN_DIM: usize = 32;

/// Squared Euclidean distance.
///
/// HNSW only needs the ordering induced by the distance, so all graph
/// operations work on squared distances and only the final scores take a
/// square root. Keeping this function free of `sqrt` matters because it is the
/// hottest function in both build and search.
#[inline]
pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if a.len() < SIMD_MIN_DIM {
        l2_sq_scalar(a, b)
    } else {
        // `None` only happens on a length mismatch, which callers guarantee
        // cannot happen.
        f32::l2sq(a, b).map_or(f32::INFINITY, |d| d as f32)
    }
}

#[inline]
fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let (a_chunks, a_tail) = a.as_chunks::<4>();
    let (b_chunks, b_tail) = b.as_chunks::<4>();
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        let d0 = x[0] - y[0];
        let d1 = x[1] - y[1];
        let d2 = x[2] - y[2];
        let d3 = x[3] - y[3];
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
    }
    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    for (x, y) in a_tail.iter().zip(b_tail) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_matches_simd() {
        for dim in [1usize, 3, 7, 15, 16, 31, 32, 33, 128, 1024] {
            let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5 - 3.0).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32).sin() * 2.0 + 1.0).collect();
            let scalar = l2_sq_scalar(&a, &b);
            let simd = f32::l2sq(&a, &b).unwrap() as f32;
            let tolerance = scalar.abs().max(1.0) * 1e-5;
            assert!(
                (scalar - simd).abs() <= tolerance,
                "dim={dim} scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn known_distance() {
        assert_eq!(l2_sq(&[26.0, 26.0, 26.0], &[30.0, 30.0, 30.0]), 48.0);
        assert_eq!(l2_sq(&[26.0, 26.0, 26.0], &[20.0, 20.0, 20.0]), 108.0);
    }

    #[test]
    fn scores_match_previous_f32_scores() {
        let score = |b: [f32; 3]| l2_sq(&[26.0, 26.0, 26.0], &b).max(0.0).sqrt() as f64;
        assert_eq!(score([30.0, 30.0, 30.0]), 6.928203105926514);
        assert_eq!(score([20.0, 20.0, 20.0]), 10.392304420471191);
    }
}
