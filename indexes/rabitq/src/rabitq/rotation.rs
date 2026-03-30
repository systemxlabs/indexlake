use rand::prelude::*;
use rand_distr::{Distribution, Uniform};

use super::RabitqError;

/// Fast Hadamard Transform (FHT) rotator with Kac Walk.
/// This matches the C++ FhtKacRotator implementation for performance.
#[derive(Debug, Clone)]
pub struct FhtKacRotator {
    dim: usize,
    padded_dim: usize,
    flip: Vec<u8>, // 4 * padded_dim / 8 bytes of random flip bits
    trunc_dim: usize,
    fac: f32,
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

impl FhtKacRotator {
    fn padding_requirement(dim: usize) -> usize {
        round_up_to_multiple(dim, 64)
    }
}

impl FhtKacRotator {
    /// Create a new FHT rotator with the provided seed.
    pub fn new(dim: usize, seed: u64) -> Self {
        let padded_dim = Self::padding_requirement(dim);
        assert_eq!(
            padded_dim % 64,
            0,
            "FHT rotator requires dimension to be multiple of 64"
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let uniform = Uniform::new_inclusive(0u8, 255u8).unwrap();

        // Generate 4 sets of random flip bits (4 rounds of FHT)
        let flip_bytes = 4 * padded_dim / 8;
        let flip: Vec<u8> = (0..flip_bytes).map(|_| uniform.sample(&mut rng)).collect();

        // Compute truncated dimension (largest power of 2 <= dim)
        let bottom_log_dim = floor_log2(dim);
        let trunc_dim = 1 << bottom_log_dim;
        let fac = 1.0 / (trunc_dim as f32).sqrt();

        Self {
            dim,
            padded_dim,
            flip,
            trunc_dim,
            fac,
        }
    }

    /// Apply sign flip based on bit mask
    fn flip_sign(data: &mut [f32], flip_bits: &[u8]) {
        for (i, value) in data.iter_mut().enumerate() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            if byte_idx < flip_bits.len() {
                let bit = (flip_bits[byte_idx] >> bit_idx) & 1;
                if bit == 1 {
                    *value = -*value;
                }
            }
        }
    }

    /// Fast Hadamard Transform (FHT) for power-of-2 dimensions
    fn fht(data: &mut [f32]) {
        let n = data.len();
        assert!(
            n.is_power_of_two(),
            "FHT requires power-of-2 dimension, got {}",
            n
        );

        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..(i + h) {
                    let x = data[j];
                    let y = data[j + h];
                    data[j] = x + y;
                    data[j + h] = x - y;
                }
            }
            h *= 2;
        }
    }

    /// Kac's walk: Hadamard-like operation
    fn kacs_walk(data: &mut [f32]) {
        let len = data.len();
        let half = len / 2;
        for i in 0..half {
            let x = data[i];
            let y = data[i + half];
            data[i] = x + y;
            data[i + half] = x - y;
        }
    }

    /// Rescale vector by constant factor
    fn rescale(data: &mut [f32], factor: f32) {
        for value in data.iter_mut() {
            *value *= factor;
        }
    }
}

impl FhtKacRotator {
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    pub fn rotate(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.dim);
        let mut output = vec![0.0f32; self.padded_dim];
        self.rotate_into(input, &mut output);
        output
    }

    pub fn rotate_into(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.dim);
        assert_eq!(output.len(), self.padded_dim);

        // Copy input and pad with zeros
        output[..self.dim].copy_from_slice(input);
        output[self.dim..].fill(0.0);

        let flip_offset = self.padded_dim / 8;

        if self.trunc_dim == self.padded_dim {
            // Case 1: trunc_dim == padded_dim (dimension is power of 2)
            // Apply 4 rounds of: flip_sign -> FHT -> rescale
            for round in 0..4 {
                let flip_start = round * flip_offset;
                let flip_end = flip_start + flip_offset;
                Self::flip_sign(output, &self.flip[flip_start..flip_end]);
                Self::fht(output);
                Self::rescale(output, self.fac);
            }
        } else {
            // Case 2: trunc_dim < padded_dim (dimension is not power of 2)
            let start = self.padded_dim - self.trunc_dim;

            // Round 1
            Self::flip_sign(output, &self.flip[0..flip_offset]);
            Self::fht(&mut output[..self.trunc_dim]);
            Self::rescale(&mut output[..self.trunc_dim], self.fac);
            Self::kacs_walk(output);

            // Round 2
            Self::flip_sign(output, &self.flip[flip_offset..2 * flip_offset]);
            Self::fht(&mut output[start..]);
            Self::rescale(&mut output[start..], self.fac);
            Self::kacs_walk(output);

            // Round 3
            Self::flip_sign(output, &self.flip[2 * flip_offset..3 * flip_offset]);
            Self::fht(&mut output[..self.trunc_dim]);
            Self::rescale(&mut output[..self.trunc_dim], self.fac);
            Self::kacs_walk(output);

            // Round 4
            Self::flip_sign(output, &self.flip[3 * flip_offset..4 * flip_offset]);
            Self::fht(&mut output[start..]);
            Self::rescale(&mut output[start..], self.fac);
            Self::kacs_walk(output);

            // Final rescale to match C++ normalization
            Self::rescale(output, 0.25);
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        // Only store the flip bits (much smaller than full matrix)
        self.flip.clone()
    }

    pub fn deserialize(dim: usize, padded_dim: usize, data: &[u8]) -> Result<Self, RabitqError> {
        let expected_len = 4 * padded_dim / 8;
        if data.len() != expected_len {
            return Err(RabitqError::InvalidPersistence(
                "FHT rotator flip bits length mismatch",
            ));
        }

        let bottom_log_dim = floor_log2(dim);
        let trunc_dim = 1 << bottom_log_dim;
        let fac = 1.0 / (trunc_dim as f32).sqrt();

        Ok(Self {
            dim,
            padded_dim,
            flip: data.to_vec(),
            trunc_dim,
            fac,
        })
    }
}

/// Compute floor(log2(x)) for positive integers
fn floor_log2(x: usize) -> usize {
    assert!(x > 0, "floor_log2 requires positive input");
    (usize::BITS - 1 - x.leading_zeros()) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_log2() {
        assert_eq!(floor_log2(1), 0);
        assert_eq!(floor_log2(2), 1);
        assert_eq!(floor_log2(3), 1);
        assert_eq!(floor_log2(4), 2);
        assert_eq!(floor_log2(7), 2);
        assert_eq!(floor_log2(8), 3);
        assert_eq!(floor_log2(960), 9);
    }

    #[test]
    fn test_fht_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        FhtKacRotator::fht(&mut data);
        // FHT is self-inverse (up to scaling)
        FhtKacRotator::fht(&mut data);
        for (i, &val) in data.iter().enumerate() {
            assert!((val - (i + 1) as f32 * 4.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fht_rotator_basic() {
        let dim = 64;
        let rotator = FhtKacRotator::new(dim, 54321);

        assert_eq!(rotator.padded_dim(), 64);

        let input = vec![1.0; dim];
        let output = rotator.rotate(&input);

        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_fht_rotator_non_power_of_2() {
        let dim = 960; // GIST dataset dimension
        let rotator = FhtKacRotator::new(dim, 98765);

        assert_eq!(rotator.padded_dim(), 960); // Already multiple of 64

        let input = vec![1.0; dim];
        let output = rotator.rotate(&input);

        assert_eq!(output.len(), 960);
    }
}
