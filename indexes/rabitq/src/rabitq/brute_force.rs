use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crc32fast::Hasher;
use rayon::prelude::*;
use roaring::RoaringBitmap;

use super::fastscan::QueryLut;
use super::fastscan_kernel::{self, FastScanLutView};
use super::quantizer::{QuantizedVector, RabitqConfig, quantize_with_centroid};
use super::rotation::FhtKacRotator;
use super::simd;
use super::{Metric, RabitqError};

const PERSIST_MAGIC: [u8; 4] = *b"RBF1";
const PERSIST_VERSION: u32 = 1;

/// Parameters for brute-force search.
#[derive(Debug, Clone, Copy)]
pub struct BruteForceSearchParams {
    pub top_k: usize,
}

impl BruteForceSearchParams {
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }
}

/// Result entry returned by brute-force search.
#[derive(Debug, Clone, PartialEq)]
pub struct BruteForceSearchResult {
    pub id: usize,
    pub score: f32,
}

/// Unified cluster data with contiguous memory layout for FastScan batch search.
#[derive(Debug, Clone)]
struct ClusterData {
    /// Single contiguous memory block for all batches
    /// Layout: [Batch 0][Batch 1]...[Batch N]
    /// Each batch layout:
    ///   - packed_binary_codes: padded_dim * 32 / 8 bytes
    ///   - f_add: 32 * 4 bytes (f32)
    ///   - f_rescale: 32 * 4 bytes (f32)
    ///   - f_error: 32 * 4 bytes (f32)
    batch_data: Vec<u8>,

    /// Packed extended codes (per-vector, C++-style on-demand unpacking)
    /// Each vector: Vec<u8> containing bit-packed ex_code
    /// Unpacked on-demand only after lower-bound filtering
    ex_codes_packed: Vec<Vec<u8>>,

    /// Per-vector ex parameters
    f_add_ex: Vec<f32>,
    f_rescale_ex: Vec<f32>,

    /// Metadata
    num_vectors: usize,
    padded_dim: usize,
}

impl ClusterData {
    /// Calculate bytes per batch in contiguous layout
    #[inline(always)]
    fn batch_stride(padded_dim: usize) -> usize {
        let binary_codes_bytes = padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;
        let params_bytes = std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE * 3; // f_add, f_rescale, f_error
        binary_codes_bytes + params_bytes
    }

    /// Get number of complete batches
    #[inline(always)]
    fn num_complete_batches(&self) -> usize {
        self.num_vectors / simd::FASTSCAN_BATCH_SIZE
    }

    /// Get number of remainder vectors
    #[inline(always)]
    fn num_remainder_vectors(&self) -> usize {
        self.num_vectors % simd::FASTSCAN_BATCH_SIZE
    }

    /// Zero-copy access to packed binary codes for a batch
    #[inline(always)]
    fn batch_bin_codes(&self, batch_idx: usize) -> &[u8] {
        let stride = Self::batch_stride(self.padded_dim);
        let offset = batch_idx * stride;
        let len = self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;
        &self.batch_data[offset..offset + len]
    }

    /// Zero-copy access to f_add parameters for a batch
    #[inline(always)]
    fn batch_f_add(&self, batch_idx: usize) -> &[f32] {
        let stride = Self::batch_stride(self.padded_dim);
        let offset = batch_idx * stride + (self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8);
        unsafe {
            std::slice::from_raw_parts(
                self.batch_data[offset..].as_ptr() as *const f32,
                simd::FASTSCAN_BATCH_SIZE,
            )
        }
    }

    /// Zero-copy access to f_rescale parameters for a batch
    #[inline(always)]
    fn batch_f_rescale(&self, batch_idx: usize) -> &[f32] {
        let stride = Self::batch_stride(self.padded_dim);
        let binary_bytes = self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;
        let f_add_bytes = std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let offset = batch_idx * stride + binary_bytes + f_add_bytes;
        unsafe {
            std::slice::from_raw_parts(
                self.batch_data[offset..].as_ptr() as *const f32,
                simd::FASTSCAN_BATCH_SIZE,
            )
        }
    }

    /// Zero-copy access to f_error parameters for a batch
    #[inline(always)]
    fn batch_f_error(&self, batch_idx: usize) -> &[f32] {
        let stride = Self::batch_stride(self.padded_dim);
        let binary_bytes = self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;
        let f_add_bytes = std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let f_rescale_bytes = std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let offset = batch_idx * stride + binary_bytes + f_add_bytes + f_rescale_bytes;
        unsafe {
            std::slice::from_raw_parts(
                self.batch_data[offset..].as_ptr() as *const f32,
                simd::FASTSCAN_BATCH_SIZE,
            )
        }
    }

    /// Build ClusterData from quantized vectors
    /// Implements unified memory layout with contiguous batch storage
    fn from_quantized_vectors(
        vectors: Vec<QuantizedVector>,
        padded_dim: usize,
        ex_bits: usize,
    ) -> Self {
        let num_vectors = vectors.len();
        let num_complete_batches = num_vectors / simd::FASTSCAN_BATCH_SIZE;
        let num_remainder = num_vectors % simd::FASTSCAN_BATCH_SIZE;

        let total_batches = if num_remainder > 0 {
            num_complete_batches + 1
        } else {
            num_complete_batches
        };

        // Allocate contiguous memory for all batches
        let stride = Self::batch_stride(padded_dim);
        let mut batch_data = vec![0u8; stride * total_batches];

        // Pre-allocate storage for packed ex_codes and parameters
        let mut ex_codes_packed = Vec::with_capacity(num_vectors);
        let mut f_add_ex = Vec::with_capacity(num_vectors);
        let mut f_rescale_ex = Vec::with_capacity(num_vectors);

        let dim_bytes = padded_dim / 8;

        // Process complete batches
        for batch_idx in 0..num_complete_batches {
            let start_idx = batch_idx * simd::FASTSCAN_BATCH_SIZE;
            let end_idx = start_idx + simd::FASTSCAN_BATCH_SIZE;
            let batch_vectors = &vectors[start_idx..end_idx];

            Self::pack_batch_into_memory(
                batch_vectors,
                &mut batch_data,
                batch_idx,
                padded_dim,
                ex_bits,
                &mut ex_codes_packed,
                &mut f_add_ex,
                &mut f_rescale_ex,
            );
        }

        // Process remainder vectors
        if num_remainder > 0 {
            let start_idx = num_complete_batches * simd::FASTSCAN_BATCH_SIZE;
            let batch_vectors = &vectors[start_idx..];

            // Pad with zeros
            let mut padded_vectors = batch_vectors.to_vec();
            let ex_bytes_per_vec = if ex_bits > 0 {
                padded_dim * ex_bits / 8
            } else {
                0
            };
            padded_vectors.resize(
                simd::FASTSCAN_BATCH_SIZE,
                QuantizedVector {
                    binary_code_packed: vec![0u8; dim_bytes],
                    ex_code_packed: vec![0u8; ex_bytes_per_vec],
                    ex_bits: ex_bits as u8,
                    dim: padded_dim,
                    delta: 0.0,
                    vl: 0.0,
                    f_add: 0.0,
                    f_rescale: 0.0,
                    f_error: 0.0,
                    residual_norm: 0.0,
                    f_add_ex: 0.0,
                    f_rescale_ex: 0.0,
                },
            );

            Self::pack_batch_into_memory_partial(
                &padded_vectors,
                num_remainder,
                &mut batch_data,
                num_complete_batches,
                padded_dim,
                ex_bits,
                &mut ex_codes_packed,
                &mut f_add_ex,
                &mut f_rescale_ex,
            );
        }

        Self {
            batch_data,
            ex_codes_packed,
            f_add_ex,
            f_rescale_ex,
            num_vectors,
            padded_dim,
        }
    }

    /// Pack a partial batch into memory (handles the last batch with < 32 vectors)
    #[allow(clippy::too_many_arguments)]
    fn pack_batch_into_memory_partial(
        vectors: &[QuantizedVector],
        actual_count: usize,
        batch_data: &mut [u8],
        batch_idx: usize,
        padded_dim: usize,
        ex_bits: usize,
        ex_codes_packed: &mut Vec<Vec<u8>>,
        f_add_ex: &mut Vec<f32>,
        f_rescale_ex: &mut Vec<f32>,
    ) {
        assert_eq!(vectors.len(), simd::FASTSCAN_BATCH_SIZE);
        assert!(actual_count <= simd::FASTSCAN_BATCH_SIZE);

        let stride = Self::batch_stride(padded_dim);
        let dim_bytes = padded_dim / 8;

        let batch_offset = batch_idx * stride;
        let binary_bytes = padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;

        // Collect binary codes into flat buffer (including padding)
        let mut binary_codes_flat = Vec::with_capacity(simd::FASTSCAN_BATCH_SIZE * dim_bytes);
        for vec in vectors.iter() {
            binary_codes_flat.extend_from_slice(&vec.binary_code_packed);
        }

        // Pack binary codes using FastScan layout
        let packed_codes = &mut batch_data[batch_offset..batch_offset + binary_bytes];
        simd::pack_codes(
            &binary_codes_flat,
            simd::FASTSCAN_BATCH_SIZE,
            dim_bytes,
            packed_codes,
        );

        // Write f_add, f_rescale, f_error as contiguous arrays
        let f_add_offset = batch_offset + binary_bytes;
        let f_rescale_offset =
            f_add_offset + std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let f_error_offset =
            f_rescale_offset + std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;

        unsafe {
            let f_add_slice = std::slice::from_raw_parts_mut(
                batch_data[f_add_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );
            let f_rescale_slice = std::slice::from_raw_parts_mut(
                batch_data[f_rescale_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );
            let f_error_slice = std::slice::from_raw_parts_mut(
                batch_data[f_error_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );

            for (i, vec) in vectors.iter().enumerate() {
                f_add_slice[i] = vec.f_add;
                f_rescale_slice[i] = vec.f_rescale;
                f_error_slice[i] = vec.f_error;
            }
        }

        // Store packed ex_codes - ONLY for actual vectors, not padding
        for vec in vectors.iter().take(actual_count) {
            if ex_bits > 0 {
                ex_codes_packed.push(vec.ex_code_packed.clone());
                f_add_ex.push(vec.f_add_ex);
                f_rescale_ex.push(vec.f_rescale_ex);
            } else {
                ex_codes_packed.push(Vec::new());
                f_add_ex.push(0.0);
                f_rescale_ex.push(0.0);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn pack_batch_into_memory(
        vectors: &[QuantizedVector],
        batch_data: &mut [u8],
        batch_idx: usize,
        padded_dim: usize,
        ex_bits: usize,
        ex_codes_packed: &mut Vec<Vec<u8>>,
        f_add_ex: &mut Vec<f32>,
        f_rescale_ex: &mut Vec<f32>,
    ) {
        assert_eq!(vectors.len(), simd::FASTSCAN_BATCH_SIZE);

        let stride = Self::batch_stride(padded_dim);
        let dim_bytes = padded_dim / 8;

        let batch_offset = batch_idx * stride;
        let binary_bytes = padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;

        // Collect binary codes into flat buffer
        let mut binary_codes_flat = Vec::with_capacity(simd::FASTSCAN_BATCH_SIZE * dim_bytes);
        for vec in vectors.iter() {
            binary_codes_flat.extend_from_slice(&vec.binary_code_packed);
        }

        // Pack binary codes using FastScan layout
        let packed_codes = &mut batch_data[batch_offset..batch_offset + binary_bytes];
        simd::pack_codes(
            &binary_codes_flat,
            simd::FASTSCAN_BATCH_SIZE,
            dim_bytes,
            packed_codes,
        );

        // Write f_add, f_rescale, f_error as contiguous arrays
        let f_add_offset = batch_offset + binary_bytes;
        let f_rescale_offset =
            f_add_offset + std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let f_error_offset =
            f_rescale_offset + std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;

        unsafe {
            let f_add_slice = std::slice::from_raw_parts_mut(
                batch_data[f_add_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );
            let f_rescale_slice = std::slice::from_raw_parts_mut(
                batch_data[f_rescale_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );
            let f_error_slice = std::slice::from_raw_parts_mut(
                batch_data[f_error_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );

            for (i, vec) in vectors.iter().enumerate() {
                f_add_slice[i] = vec.f_add;
                f_rescale_slice[i] = vec.f_rescale;
                f_error_slice[i] = vec.f_error;
            }
        }

        // Store packed ex_codes for all vectors in batch
        for vec in vectors.iter() {
            if ex_bits > 0 {
                ex_codes_packed.push(vec.ex_code_packed.clone());
                f_add_ex.push(vec.f_add_ex);
                f_rescale_ex.push(vec.f_rescale_ex);
            } else {
                ex_codes_packed.push(Vec::new());
                f_add_ex.push(0.0);
                f_rescale_ex.push(0.0);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct HeapCandidate {
    id: usize,
    distance: f32,
    score: f32,
}

#[derive(Debug, Clone)]
struct HeapEntry {
    candidate: HeapCandidate,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.candidate
            .distance
            .to_bits()
            .eq(&other.candidate.distance.to_bits())
            && self.candidate.id == other.candidate.id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.candidate.distance.total_cmp(&other.candidate.distance)
    }
}

/// Precomputed query constants to avoid repeated calculations during search.
struct QueryPrecomputed {
    rotated_query: Vec<f32>,
    query_norm: f32,
    k1x_sum_q: f32, // c1 × sum_query (precomputed)
    kbx_sum_q: f32, // cb × sum_query (precomputed)
    binary_scale: f32,
    /// Optional LUT for FastScan batch search
    lut: Option<QueryLut>,
}

impl QueryPrecomputed {
    fn new(rotated_query: Vec<f32>, ex_bits: usize) -> Self {
        let sum_query: f32 = rotated_query.iter().sum();
        let query_norm: f32 = rotated_query.iter().map(|v| v * v).sum::<f32>().sqrt();
        let c1 = -0.5f32;
        let cb = -((1 << ex_bits) as f32 - 0.5);
        let binary_scale = (1 << ex_bits) as f32;

        Self {
            rotated_query,
            query_norm,
            k1x_sum_q: c1 * sum_query,
            kbx_sum_q: cb * sum_query,
            binary_scale,
            lut: None, // LUT built on demand for batch search
        }
    }

    /// Build LUT for FastScan batch search
    /// Should be called once per query before searching clusters
    fn build_lut(&mut self, padded_dim: usize) {
        self.lut = Some(QueryLut::new(&self.rotated_query, padded_dim));
    }
}

fn write_u8<W: Write>(writer: &mut W, value: u8, hasher: Option<&mut Hasher>) -> io::Result<()> {
    let bytes = [value];
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    writer.write_all(&bytes)
}

fn write_u32<W: Write>(writer: &mut W, value: u32, hasher: Option<&mut Hasher>) -> io::Result<()> {
    let bytes = value.to_le_bytes();
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    writer.write_all(&bytes)
}

fn write_u64<W: Write>(writer: &mut W, value: u64, hasher: Option<&mut Hasher>) -> io::Result<()> {
    let bytes = value.to_le_bytes();
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    writer.write_all(&bytes)
}

fn write_f32<W: Write>(writer: &mut W, value: f32, hasher: Option<&mut Hasher>) -> io::Result<()> {
    let bytes = value.to_le_bytes();
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    writer.write_all(&bytes)
}

fn read_u8<R: Read>(reader: &mut R, hasher: Option<&mut Hasher>) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    if let Some(h) = hasher {
        h.update(&buf);
    }
    Ok(buf[0])
}

fn read_u32<R: Read>(reader: &mut R, hasher: Option<&mut Hasher>) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    if let Some(h) = hasher {
        h.update(&buf);
    }
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R, hasher: Option<&mut Hasher>) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    if let Some(h) = hasher {
        h.update(&buf);
    }
    Ok(u64::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R, hasher: Option<&mut Hasher>) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    if let Some(h) = hasher {
        h.update(&buf);
    }
    Ok(f32::from_le_bytes(buf))
}

fn usize_from_u64(value: u64) -> Result<usize, RabitqError> {
    usize::try_from(value)
        .map_err(|_| RabitqError::InvalidPersistence("value exceeds platform limits"))
}

fn metric_to_tag(metric: Metric) -> u8 {
    match metric {
        Metric::L2 => 0,
        Metric::InnerProduct => 1,
    }
}

fn tag_to_metric(tag: u8) -> Option<Metric> {
    match tag {
        0 => Some(Metric::L2),
        1 => Some(Metric::InnerProduct),
        _ => None,
    }
}

/// Brute-force RaBitQ index without clustering.
///
/// This index quantizes all vectors relative to a zero centroid and performs
/// exhaustive search over all vectors. It's suitable for small to medium datasets
/// (e.g., <200K vectors with 1024 dimensions) where IVF clustering overhead is
/// not justified.
#[derive(Debug, Clone)]
pub struct BruteForceRabitqIndex {
    dim: usize,
    padded_dim: usize,
    metric: Metric,
    rotator: FhtKacRotator,
    vectors: Vec<QuantizedVector>,
    ex_bits: usize,
    /// FastScan batch data for SIMD-accelerated search
    cluster: ClusterData,
    /// Function pointer for ex-code inner product on packed data
    /// Selected based on ex_bits at index construction time
    ip_func: simd::ExIpFunc,
}

impl BruteForceRabitqIndex {
    /// Train a new brute-force index from the provided dataset.
    pub fn train(
        data: &[Vec<f32>],
        total_bits: usize,
        metric: Metric,
        seed: u64,
    ) -> Result<Self, RabitqError> {
        if data.is_empty() {
            return Err(RabitqError::InvalidConfig(
                "training data must be non-empty",
            ));
        }
        if total_bits == 0 || total_bits > 16 {
            return Err(RabitqError::InvalidConfig(
                "total_bits must be between 1 and 16",
            ));
        }

        let dim = data[0].len();
        if data.iter().any(|v| v.len() != dim) {
            return Err(RabitqError::InvalidConfig(
                "input vectors must share the same dimension",
            ));
        }

        let rotator = FhtKacRotator::new(dim, seed);
        let padded_dim = rotator.padded_dim();

        let rotated_data: Vec<Vec<f32>> = data.par_iter().map(|v| rotator.rotate(v)).collect();

        let config = RabitqConfig::new(padded_dim, total_bits, seed);

        // Use zero centroid for brute-force quantization
        let zero_centroid = vec![0.0f32; padded_dim];

        let vectors: Vec<QuantizedVector> = rotated_data
            .par_iter()
            .map(|vec| quantize_with_centroid(vec, &zero_centroid, &config, metric))
            .collect();

        let ex_bits = total_bits.saturating_sub(1);
        let ip_func = simd::select_excode_ipfunc(ex_bits);
        let cluster = ClusterData::from_quantized_vectors(vectors.clone(), padded_dim, ex_bits);

        Ok(Self {
            dim,
            padded_dim,
            metric,
            rotator,
            vectors,
            ex_bits,
            cluster,
            ip_func,
        })
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Persist the index to the provided filesystem path.
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), RabitqError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.save_to_writer(writer)
    }

    /// Persist the index using the supplied writer.
    pub fn save_to_writer<W: Write>(&self, writer: W) -> Result<(), RabitqError> {
        let mut writer = BufWriter::new(writer);
        writer.write_all(&PERSIST_MAGIC)?;
        write_u32(&mut writer, PERSIST_VERSION, None)?;

        let mut hasher = Hasher::new();

        let dim = u32::try_from(self.dim)
            .map_err(|_| RabitqError::InvalidPersistence("dimension exceeds persistence limits"))?;
        write_u32(&mut writer, dim, Some(&mut hasher))?;

        let padded_dim = u32::try_from(self.padded_dim).map_err(|_| {
            RabitqError::InvalidPersistence("padded_dim exceeds persistence limits")
        })?;
        write_u32(&mut writer, padded_dim, Some(&mut hasher))?;

        let metric_tag = metric_to_tag(self.metric);
        write_u8(&mut writer, metric_tag, Some(&mut hasher))?;

        let ex_bits = u8::try_from(self.ex_bits)
            .map_err(|_| RabitqError::InvalidPersistence("ex_bits exceeds persistence limits"))?;
        write_u8(&mut writer, ex_bits, Some(&mut hasher))?;

        let total_bits = self
            .ex_bits
            .checked_add(1)
            .ok_or(RabitqError::InvalidPersistence("total_bits overflow"))?;
        let total_bits_u8 = u8::try_from(total_bits).map_err(|_| {
            RabitqError::InvalidPersistence("total_bits exceeds persistence limits")
        })?;
        write_u8(&mut writer, total_bits_u8, Some(&mut hasher))?;

        let vector_count = u64::try_from(self.len()).map_err(|_| {
            RabitqError::InvalidPersistence("vector count exceeds persistence limits")
        })?;
        write_u64(&mut writer, vector_count, Some(&mut hasher))?;

        // Save rotator state
        let rotator_data = self.rotator.serialize();
        let rotator_len = u64::try_from(rotator_data.len())
            .map_err(|_| RabitqError::InvalidPersistence("rotator data too large"))?;
        write_u64(&mut writer, rotator_len, Some(&mut hasher))?;
        writer.write_all(&rotator_data)?;
        hasher.update(&rotator_data);

        // Save vectors
        for vector in &self.vectors {
            if vector.dim != self.padded_dim {
                return Err(RabitqError::InvalidPersistence(
                    "quantized vector dimension mismatch",
                ));
            }

            // Write packed binary_code (already in packed format)
            writer.write_all(&vector.binary_code_packed)?;
            hasher.update(&vector.binary_code_packed);

            // Write packed ex_code (already in packed format)
            writer.write_all(&vector.ex_code_packed)?;
            hasher.update(&vector.ex_code_packed);

            // Metadata (8 × f32 = 32 bytes)
            write_f32(&mut writer, vector.delta, Some(&mut hasher))?;
            write_f32(&mut writer, vector.vl, Some(&mut hasher))?;
            write_f32(&mut writer, vector.f_add, Some(&mut hasher))?;
            write_f32(&mut writer, vector.f_rescale, Some(&mut hasher))?;
            write_f32(&mut writer, vector.f_error, Some(&mut hasher))?;
            write_f32(&mut writer, vector.residual_norm, Some(&mut hasher))?;
            write_f32(&mut writer, vector.f_add_ex, Some(&mut hasher))?;
            write_f32(&mut writer, vector.f_rescale_ex, Some(&mut hasher))?;
        }

        let checksum = hasher.finalize();
        write_u32(&mut writer, checksum, None)?;

        writer.flush()?;
        Ok(())
    }

    /// Load an index from the provided filesystem path.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, RabitqError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::load_from_reader(reader)
    }

    /// Load an index from a persisted byte stream.
    pub fn load_from_reader<R: Read>(reader: R) -> Result<Self, RabitqError> {
        let mut reader = BufReader::new(reader);
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != PERSIST_MAGIC {
            return Err(RabitqError::InvalidPersistence("unrecognized file header"));
        }

        let version = read_u32(&mut reader, None)?;
        if version != 1 {
            return Err(RabitqError::InvalidPersistence(
                "unsupported index format version",
            ));
        }

        let mut hasher = Hasher::new();

        let dim = read_u32(&mut reader, Some(&mut hasher))? as usize;
        if dim == 0 {
            return Err(RabitqError::InvalidPersistence(
                "dimension must be positive",
            ));
        }

        let padded_dim = read_u32(&mut reader, Some(&mut hasher))? as usize;
        if padded_dim < dim {
            return Err(RabitqError::InvalidPersistence("padded_dim must be >= dim"));
        }

        let metric_tag = read_u8(&mut reader, Some(&mut hasher))?;
        let metric = tag_to_metric(metric_tag)
            .ok_or(RabitqError::InvalidPersistence("unknown metric tag"))?;

        let ex_bits = read_u8(&mut reader, Some(&mut hasher))? as usize;
        if ex_bits > 16 {
            return Err(RabitqError::InvalidPersistence("ex_bits out of range"));
        }

        let total_bits = read_u8(&mut reader, Some(&mut hasher))? as usize;
        if total_bits == 0 || total_bits > 16 {
            return Err(RabitqError::InvalidPersistence("total_bits out of range"));
        }
        if total_bits.saturating_sub(1) != ex_bits {
            return Err(RabitqError::InvalidPersistence(
                "total_bits does not match ex_bits",
            ));
        }

        let vector_count = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;

        let rotator_data_len = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;
        let mut rotator_data = vec![0u8; rotator_data_len];
        reader.read_exact(&mut rotator_data)?;
        hasher.update(&rotator_data);

        let rotator = FhtKacRotator::deserialize(dim, padded_dim, &rotator_data)?;

        let mut vectors = Vec::with_capacity(vector_count);
        for _ in 0..vector_count {
            // Read packed binary code
            let binary_packed_size = padded_dim.div_ceil(8);
            let mut binary_code_packed = vec![0u8; binary_packed_size];
            reader.read_exact(&mut binary_code_packed)?;
            hasher.update(&binary_code_packed);

            // Read packed ex code
            let ex_packed_size = if ex_bits > 0 {
                (padded_dim * ex_bits).div_ceil(8)
            } else {
                0
            };
            let mut ex_code_packed = vec![0u8; ex_packed_size];
            reader.read_exact(&mut ex_code_packed)?;
            hasher.update(&ex_code_packed);

            let delta = read_f32(&mut reader, Some(&mut hasher))?;
            let vl = read_f32(&mut reader, Some(&mut hasher))?;
            let f_add = read_f32(&mut reader, Some(&mut hasher))?;
            let f_rescale = read_f32(&mut reader, Some(&mut hasher))?;
            let f_error = read_f32(&mut reader, Some(&mut hasher))?;
            let residual_norm = read_f32(&mut reader, Some(&mut hasher))?;
            let f_add_ex = read_f32(&mut reader, Some(&mut hasher))?;
            let f_rescale_ex = read_f32(&mut reader, Some(&mut hasher))?;

            vectors.push(QuantizedVector {
                binary_code_packed,
                ex_code_packed,
                ex_bits: ex_bits as u8,
                dim: padded_dim,
                delta,
                vl,
                f_add,
                f_rescale,
                f_error,
                residual_norm,
                f_add_ex,
                f_rescale_ex,
            });
        }

        let computed_checksum = hasher.finalize();
        let stored_checksum = read_u32(&mut reader, None)?;
        if computed_checksum != stored_checksum {
            return Err(RabitqError::InvalidPersistence("checksum mismatch"));
        }

        let ip_func = simd::select_excode_ipfunc(ex_bits);
        let cluster = ClusterData::from_quantized_vectors(vectors.clone(), padded_dim, ex_bits);

        Ok(Self {
            dim,
            padded_dim,
            metric,
            rotator,
            vectors,
            ex_bits,
            cluster,
            ip_func,
        })
    }

    /// Search for the nearest neighbours of the provided query vector.
    pub fn search(
        &self,
        query: &[f32],
        params: BruteForceSearchParams,
    ) -> Result<Vec<BruteForceSearchResult>, RabitqError> {
        self.search_internal(query, params, None)
    }

    /// Search for the nearest neighbours of the provided query vector,
    /// filtering results to only include vector IDs present in the provided bitmap.
    pub fn search_filtered(
        &self,
        query: &[f32],
        params: BruteForceSearchParams,
        filter: &RoaringBitmap,
    ) -> Result<Vec<BruteForceSearchResult>, RabitqError> {
        self.search_internal(query, params, Some(filter))
    }

    fn search_internal(
        &self,
        query: &[f32],
        params: BruteForceSearchParams,
        filter: Option<&RoaringBitmap>,
    ) -> Result<Vec<BruteForceSearchResult>, RabitqError> {
        if self.is_empty() {
            return Err(RabitqError::EmptyIndex);
        }
        if query.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        if params.top_k == 0 {
            return Ok(Vec::new());
        }

        let rotated_query = self.rotator.rotate(query);
        let mut query_precomp = QueryPrecomputed::new(rotated_query, self.ex_bits);
        query_precomp.build_lut(self.padded_dim);

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        let g_add = 0.0f32;
        let g_error = 0.0f32;
        let dot_query_centroid = 0.0f32;
        let cluster = &self.cluster;
        let top_k = params.top_k;

        let lut_view = query_precomp.lut.as_ref().map(|lut| FastScanLutView {
            lut_i8: &lut.lut_i8,
            delta: lut.delta,
            sum_vl_lut: lut.sum_vl_lut,
        });

        let Some(lut_view) = lut_view else {
            return Ok(Vec::new());
        };

        // Process complete batches (32 vectors each)
        let num_batches = cluster.num_complete_batches();
        let num_remainder = cluster.num_remainder_vectors();
        let total_batches = if num_remainder > 0 {
            num_batches + 1
        } else {
            num_batches
        };

        for batch_idx in 0..total_batches {
            let batch_start = batch_idx * simd::FASTSCAN_BATCH_SIZE;
            let batch_end = (batch_start + simd::FASTSCAN_BATCH_SIZE).min(cluster.num_vectors);
            let actual_batch_size = batch_end - batch_start;

            // Step 1: Accumulate distances using FastScan SIMD with zero-copy access
            // Get batch parameters using zero-copy slices
            let batch_f_add = cluster.batch_f_add(batch_idx);
            let batch_f_rescale = cluster.batch_f_rescale(batch_idx);
            let batch_f_error = cluster.batch_f_error(batch_idx);

            // Allocate output arrays on stack (no heap allocation)
            let mut ip_x0_qr_values = [0.0f32; simd::FASTSCAN_BATCH_SIZE];
            let mut est_distances = [0.0f32; simd::FASTSCAN_BATCH_SIZE];
            let mut lower_bounds = [0.0f32; simd::FASTSCAN_BATCH_SIZE];

            fastscan_kernel::compute_fastscan_batch(
                lut_view,
                cluster.batch_bin_codes(batch_idx),
                self.padded_dim,
                batch_f_add,
                batch_f_rescale,
                batch_f_error,
                g_add,
                g_error,
                query_precomp.k1x_sum_q,
                &mut ip_x0_qr_values,
                &mut est_distances,
                &mut lower_bounds,
            );

            // Step 2: Process each vector in the batch (pruning and ex-code evaluation)
            // Distances are now pre-computed in vectorized fashion
            for i in 0..actual_batch_size {
                let vector_id = batch_start + i;

                // Apply filter if provided
                if let Some(filter_bitmap) = filter
                    && !filter_bitmap.contains(vector_id as u32)
                {
                    continue;
                }

                // Use pre-computed values (vectorized above)
                let ip_x0_qr = ip_x0_qr_values[i];
                let est_distance = est_distances[i];
                let lower_bound = fastscan_kernel::sanitize_lower_bound(
                    lower_bounds[i],
                    self.metric,
                    dot_query_centroid,
                    query_precomp.query_norm,
                );

                // Step 3: Check against current k-th distance
                let distk = if heap.len() < top_k {
                    f32::INFINITY
                } else {
                    heap.peek()
                        .map(|entry| entry.candidate.distance)
                        .unwrap_or(f32::INFINITY)
                };

                if lower_bound >= distk {
                    continue;
                }

                // Step 4: Compute final distance with ex-codes if available
                let mut distance = est_distance;
                if self.ex_bits > 0 {
                    distance = fastscan_kernel::refine_distance_with_ex(
                        &query_precomp.rotated_query,
                        &cluster.ex_codes_packed[vector_id],
                        self.padded_dim,
                        self.ex_bits,
                        ip_x0_qr,
                        query_precomp.binary_scale,
                        query_precomp.kbx_sum_q,
                        g_add,
                        cluster.f_add_ex[vector_id],
                        cluster.f_rescale_ex[vector_id],
                        Some(self.ip_func),
                    );
                }

                if !distance.is_finite() {
                    continue;
                }

                let score = match self.metric {
                    Metric::L2 => distance,
                    Metric::InnerProduct => -distance,
                };

                // Step 5: Update heap
                heap.push(HeapEntry {
                    candidate: HeapCandidate {
                        id: vector_id,
                        distance,
                        score,
                    },
                });

                if heap.len() > top_k {
                    heap.pop();
                }
            }
        }

        let candidates: Vec<HeapCandidate> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|entry| entry.candidate)
            .collect();

        Ok(candidates
            .into_iter()
            .map(|candidate| BruteForceSearchResult {
                id: candidate.id,
                score: match self.metric {
                    Metric::L2 => candidate.distance,
                    Metric::InnerProduct => candidate.score,
                },
            })
            .collect())
    }
}
