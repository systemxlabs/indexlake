//! FastScan query lookup table construction.
//!
//! Provides LUT builders that quantize a rotated query into i8 (or split i16)
//! tables consumed by the SIMD accumulate kernels in [`super::simd`].

use super::simd;

/// Query lookup table for FastScan batch search
#[derive(Debug, Clone)]
pub struct QueryLut {
    /// Quantized lookup table (i8 values for SIMD)
    pub lut_i8: Vec<i8>,
    /// Quantization delta factor
    pub delta: f32,
    /// Sum of vl across all lookup tables
    pub sum_vl_lut: f32,
}

impl QueryLut {
    /// Build LUT from rotated query.
    pub fn new(query: &[f32], padded_dim: usize) -> Self {
        assert!(
            padded_dim.is_multiple_of(4),
            "padded_dim must be multiple of 4 for LUT"
        );

        let table_length = padded_dim * 4; // padded_dim << 2

        // Step 1: Generate float LUT using pack_lut_f32
        let mut lut_float = vec![0.0f32; table_length];
        simd::pack_lut_f32(query, &mut lut_float);

        // Step 2: Find min and max of LUT
        let vl_lut = lut_float
            .iter()
            .copied()
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);
        let vr_lut = lut_float
            .iter()
            .copied()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);

        // Step 3: Compute delta for i8 quantization (8 bits)
        let delta = (vr_lut - vl_lut) / 255.0f32; // (1 << 8) - 1 = 255

        // Step 4: Quantize float LUT to i8
        let mut lut_i8 = vec![0i8; table_length];
        if delta > 0.0 {
            for i in 0..table_length {
                let quantized = ((lut_float[i] - vl_lut) / delta).round();
                // Must convert to u8 first, then to i8 to avoid saturation
                // f32 as i8 saturates at 127, but we need wrapping behavior (128-255 -> -128 to -1)
                lut_i8[i] = (quantized.clamp(0.0, 255.0) as u8) as i8;
            }
        }

        // Step 5: Compute sum_vl_lut
        let num_table = table_length / 16;
        let sum_vl_lut = vl_lut * (num_table as f32);

        Self {
            lut_i8,
            delta,
            sum_vl_lut,
        }
    }
}
