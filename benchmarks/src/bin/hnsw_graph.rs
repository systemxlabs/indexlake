//! Graph-level HNSW benchmark: build and query `HnswGraph` in memory.
//!
//! This isolates graph construction, distance work, heaps and traversal from
//! the storage, catalog and DataFusion overhead measured by the end-to-end
//! `indexlake_hnsw` benchmark. It needs no external services.
//!
//! Run with:
//!   cargo run --release -p indexlake-benchmarks --bin hnsw_graph

use std::hint::black_box;
use std::time::Instant;

use indexlake::index::RowValidity;
use indexlake_benchmarks::benchprintln;
use indexlake_index_hnsw::internals::HnswGraph;
use uuid::Uuid;

/// Deterministic generator shared with the unit-test sweeps.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

fn random_points(rng: &mut SplitMix64, count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
        .collect()
}

/// Independent ground truth for recall.
fn brute_force(points: &[Vec<f32>], query: &[f32], limit: usize) -> Vec<usize> {
    let mut scored: Vec<(f64, usize)> = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let dist_sq: f64 = point
                .iter()
                .zip(query)
                .map(|(a, b)| {
                    let delta = *a as f64 - *b as f64;
                    delta * delta
                })
                .sum();
            (dist_sq, index)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, index)| index)
        .collect()
}

fn main() {
    let mut rng = SplitMix64(0x1234_5678);
    let dim = 256usize;
    let count = 20_000usize;
    let ef_construction = 200usize;
    let limit = 10usize;
    let queries_count = 200usize;
    let ef_values = [50usize, 100, 200];
    let recall_queries = 50usize;

    let points = random_points(&mut rng, count, dim);
    let mut vectors = Vec::with_capacity(count * dim);
    for point in &points {
        vectors.extend_from_slice(point);
    }
    let row_ids: Vec<Uuid> = (0..count)
        .map(|index| Uuid::from_u128(index as u128))
        .collect();
    let row_offsets: Vec<u32> = (0..count as u32).collect();

    let start = Instant::now();
    let graph = HnswGraph::build(vectors, row_ids, row_offsets, dim, ef_construction)
        .expect("build HNSW graph");
    let build_time = start.elapsed();
    benchprintln!(
        "HNSW graph: built {count} x {dim} (ef_construction={ef_construction}) in {build_time:?}"
    );

    let validity = RowValidity::new(count);
    let queries = random_points(&mut rng, queries_count, dim);

    for ef in ef_values {
        let start = Instant::now();
        let mut total = 0usize;
        for query in &queries {
            let found = graph
                .search(query, ef, &validity, limit)
                .expect("search HNSW graph");
            total += black_box(found.len());
        }
        let elapsed = start.elapsed();
        assert_eq!(total, queries_count * limit);
        benchprintln!(
            "HNSW graph: {queries_count} queries, ef_search={ef}, limit={limit}: {:?} ({:.3} us/query)",
            elapsed,
            elapsed.as_secs_f64() * 1e6 / queries_count as f64
        );
    }

    let expected: Vec<Vec<usize>> = queries[..recall_queries]
        .iter()
        .map(|query| brute_force(&points, query, limit))
        .collect();
    let mut recall = vec![0.0f64; ef_values.len()];
    for (slot, &ef) in ef_values.iter().enumerate() {
        let mut hits = 0usize;
        for (query, expected) in queries[..recall_queries].iter().zip(&expected) {
            let actual = graph
                .search(query, ef, &validity, limit)
                .expect("search HNSW graph");
            hits += actual
                .iter()
                .filter(|(id, _)| expected.contains(&(*id as usize)))
                .count();
        }
        recall[slot] = hits as f64 / (recall_queries * limit) as f64;
    }
    let recall_summary: Vec<String> = ef_values
        .iter()
        .zip(&recall)
        .map(|(ef, recall)| format!("ef={ef} {recall:.4}"))
        .collect();
    benchprintln!("HNSW graph: recall@10 {}", recall_summary.join(", "));
}
