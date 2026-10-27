use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use indexlake::index::RowValidity;
use indexlake::{ILError, ILResult};
use uuid::Uuid;

use crate::distance::l2_sq;

/// Maximum number of neighbors on layers above zero.
pub(crate) const M: usize = 16;
/// Maximum number of neighbors on the zero layer. The paper recommends a
/// larger value on the bottom layer because it carries most of the routing.
pub(crate) const M0: usize = 32;
/// Hard cap for the randomly assigned level. The probability of ever reaching
/// this is `M^-MAX_LEVEL`, far below any realistic node count.
pub(crate) const MAX_LEVEL: usize = 32;

pub(crate) type NodeId = u32;
pub(crate) const NONE: NodeId = u32::MAX;

const LEVEL_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// A node in an upper layer. Layer `l` (`1..=level`) occupies
/// `links[(l - 1) * M .. (l - 1) * M + counts[l - 1]]`.
#[derive(Debug, Default)]
pub(crate) struct UpperNode {
    pub links: Vec<NodeId>,
    pub counts: Vec<u8>,
}

/// Immutable HNSW graph plus the row metadata needed to answer queries.
///
/// Layout notes:
/// * vectors are stored contiguously and indexed by `node * dim`, which keeps
///   distance scans cache friendly;
/// * layer zero is a flat `node_count * M0` adjacency matrix so its traversal
///   needs a single indexed load;
/// * every node records the row offset inside the index segment. Validity is
///   expressed in segment row coordinates, and gaps caused by NULL vectors must
///   not shift those coordinates.
#[derive(Debug)]
pub(crate) struct HnswGraph {
    pub dim: usize,
    pub vectors: Vec<f32>,
    pub row_ids: Vec<Uuid>,
    pub row_offsets: Vec<u32>,
    pub levels: Vec<u8>,
    pub links0: Vec<NodeId>,
    pub counts0: Vec<u8>,
    pub upper: Vec<UpperNode>,
    pub entry: Option<NodeId>,
    pub max_level: usize,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    dist_sq: f32,
    id: NodeId,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq.total_cmp(&other.dist_sq) == Ordering::Equal && self.id == other.id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist_sq
            .total_cmp(&other.dist_sq)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Reusable per-thread search state. Reusing the visited marks and heaps
/// removes allocation churn from the insert loop and from repeated queries.
struct Scratch {
    marks: Vec<u32>,
    epoch: u32,
    candidates: BinaryHeap<Reverse<Candidate>>,
    results: BinaryHeap<Candidate>,
}

impl Scratch {
    fn new(node_count: usize, ef: usize) -> Self {
        Self {
            marks: vec![0; node_count],
            epoch: 0,
            candidates: BinaryHeap::with_capacity(ef.saturating_mul(2).max(16)),
            results: BinaryHeap::with_capacity(ef.max(16)),
        }
    }

    /// Grow the heaps ahead of a search so a steady query stream performs no
    /// heap allocation. The marks are grown by [`Scratch::begin`]; both heaps
    /// are empty between searches.
    fn reserve(&mut self, ef: usize) {
        let candidates_capacity = ef.saturating_mul(2);
        if self.candidates.capacity() < candidates_capacity {
            self.candidates.reserve(candidates_capacity);
        }
        if self.results.capacity() < ef {
            self.results.reserve(ef);
        }
    }

    fn begin(&mut self, node_count: usize) -> u32 {
        if self.marks.len() < node_count {
            self.marks.resize(node_count, 0);
        }
        if self.epoch == u32::MAX {
            self.marks.fill(0);
            self.epoch = 1;
        } else {
            self.epoch += 1;
        }
        self.epoch
    }

    #[inline]
    fn mark(&mut self, epoch: u32, id: NodeId) -> bool {
        let slot = &mut self.marks[id as usize];
        if *slot == epoch {
            false
        } else {
            *slot = epoch;
            true
        }
    }
}

thread_local! {
    /// The query path only gets `&self` from the `Index` trait, so it cannot
    /// take a caller-owned workspace. Keep one scratch buffer per thread
    /// instead: at millions of nodes the visited array alone is several
    /// megabytes, and zeroing it per query would be a noticeable part of the
    /// search cost.
    static SEARCH_SCRATCH: RefCell<Option<Scratch>> = const { RefCell::new(None) };
}

/// Run `f` with the calling thread's reusable scratch state.
///
/// `try_borrow_mut` keeps a nested search on the same thread from panicking
/// should a future callback re-enter the search; it allocates a private
/// scratch instead.
fn with_search_scratch<R>(node_count: usize, ef: usize, f: impl FnOnce(&mut Scratch) -> R) -> R {
    SEARCH_SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => {
            let scratch = scratch.get_or_insert_with(|| Scratch::new(0, 0));
            scratch.reserve(ef);
            f(scratch)
        }
        Err(_) => f(&mut Scratch::new(node_count, ef)),
    })
}

impl HnswGraph {
    pub(crate) fn build(
        vectors: Vec<f32>,
        row_ids: Vec<Uuid>,
        row_offsets: Vec<u32>,
        dim: usize,
        ef_construction: usize,
    ) -> ILResult<Self> {
        let node_count = row_ids.len();
        if row_offsets.len() != node_count {
            return Err(ILError::index(
                "HNSW row offset count does not match row count",
            ));
        }
        if node_count >= NONE as usize {
            return Err(ILError::index(format!(
                "HNSW index supports at most {} nodes",
                NONE as usize - 1
            )));
        }
        if node_count > 0 && dim == 0 {
            return Err(ILError::index("HNSW vector dimension must be positive"));
        }
        if vectors.len() != node_count.saturating_mul(dim) {
            return Err(ILError::index(format!(
                "HNSW vector buffer has {} values, expected {}",
                vectors.len(),
                node_count.saturating_mul(dim)
            )));
        }
        if vectors.iter().any(|v| !v.is_finite()) {
            return Err(ILError::index(
                "HNSW vectors must contain only finite values",
            ));
        }

        let levels: Vec<u8> = (0..node_count)
            .map(|id| deterministic_level(id as NodeId) as u8)
            .collect();
        let links0 = vec![NONE; node_count.saturating_mul(M0)];
        let counts0 = vec![0u8; node_count];
        let upper: Vec<UpperNode> = levels
            .iter()
            .map(|&level| {
                let level = level as usize;
                UpperNode {
                    links: vec![NONE; level * M],
                    counts: vec![0u8; level],
                }
            })
            .collect();

        let mut graph = Self {
            dim,
            vectors,
            row_ids,
            row_offsets,
            levels,
            links0,
            counts0,
            upper,
            entry: None,
            max_level: 0,
        };

        if node_count > 0 {
            let ef = ef_construction.max(1);
            let mut scratch = Scratch::new(node_count, ef);
            let mut candidates = Vec::with_capacity(ef);
            let mut selected = Vec::with_capacity(M0);
            let mut query = Vec::with_capacity(dim);
            for id in 0..node_count {
                let start = id * dim;
                query.clear();
                query.extend_from_slice(&graph.vectors[start..start + dim]);
                graph.insert_preallocated(
                    id as NodeId,
                    &query,
                    ef,
                    &mut scratch,
                    &mut candidates,
                    &mut selected,
                )?;
            }
        }
        Ok(graph)
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.row_ids.len()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }

    /// Insert a node whose vector, row metadata and adjacency storage have
    /// already been materialised.
    fn insert_preallocated(
        &mut self,
        id: NodeId,
        query: &[f32],
        ef_construction: usize,
        scratch: &mut Scratch,
        candidates: &mut Vec<Candidate>,
        selected: &mut Vec<NodeId>,
    ) -> ILResult<()> {
        let level = self.levels[id as usize] as usize;
        let Some(mut ep) = self.entry else {
            self.entry = Some(id);
            self.max_level = level;
            return Ok(());
        };

        let mut ep_dist = self.distance_to(query, ep);
        let mut layer = self.max_level;
        while layer > level {
            (ep, ep_dist) = self.greedy_layer(query, ep, ep_dist, layer);
            layer -= 1;
        }

        let start_layer = level.min(self.max_level);
        for layer in (0..=start_layer).rev() {
            self.search_layer(query, ep, ef_construction, layer, None, scratch, candidates)?;
            // The entry point is always part of the result set, so this is
            // non-empty.
            debug_assert!(!candidates.is_empty());
            let cap = if layer == 0 { M0 } else { M };
            self.select_neighbors(candidates, cap, selected);
            self.set_node_links(id, layer, selected);
            for (rank, &neighbor) in selected.iter().enumerate() {
                // Forcing the nearest neighbor to accept a back link on layer
                // zero gives every inserted node an incoming edge at insertion
                // time. That edge can still be pruned later, so long-term
                // reachability is best effort. In particular, a graph made of
                // many exact duplicates ties every distance, every node picks
                // the same low-id hubs, and a full list never accepts an
                // equal-distance reverse link; such nodes may become
                // unreachable.
                self.add_backlink(neighbor, id, layer, layer == 0 && rank == 0);
            }
            ep = candidates[0].id;
        }

        if level > self.max_level {
            self.entry = Some(id);
            self.max_level = level;
        }
        Ok(())
    }

    /// Algorithm 2: greedy descent on a single upper layer.
    fn greedy_layer(
        &self,
        query: &[f32],
        mut node: NodeId,
        mut dist: f32,
        layer: usize,
    ) -> (NodeId, f32) {
        loop {
            let mut next = node;
            let mut next_dist = dist;
            for i in 0..self.neighbor_count(layer, node) {
                let neighbor = self.neighbor_at(layer, node, i);
                let neighbor_dist = self.distance_to(query, neighbor);
                if neighbor_dist < next_dist {
                    next = neighbor;
                    next_dist = neighbor_dist;
                }
            }
            if next == node {
                return (node, dist);
            }
            node = next;
            dist = next_dist;
        }
    }

    /// Best-first graph search on `layer`.
    ///
    /// When `validity` is supplied only valid nodes enter the result heap, but
    /// invalid nodes still participate in traversal. If fewer than `ef` valid
    /// nodes have been found the search keeps expanding (potentially over the
    /// whole reachable component), which is required to return up to `limit`
    /// valid rows when many rows have been deleted.
    #[allow(clippy::too_many_arguments)]
    fn search_layer(
        &self,
        query: &[f32],
        entry: NodeId,
        ef: usize,
        layer: usize,
        validity: Option<&RowValidity>,
        scratch: &mut Scratch,
        out: &mut Vec<Candidate>,
    ) -> ILResult<()> {
        let ef = ef.max(1);
        scratch.candidates.clear();
        scratch.results.clear();
        out.clear();

        let epoch = scratch.begin(self.len());
        let entry_dist = self.distance_to(query, entry);
        scratch.mark(epoch, entry);
        scratch.candidates.push(Reverse(Candidate {
            dist_sq: entry_dist,
            id: entry,
        }));
        let entry_is_valid = match validity {
            Some(validity) => self.node_is_valid(entry, validity)?,
            None => true,
        };
        if entry_is_valid {
            scratch.results.push(Candidate {
                dist_sq: entry_dist,
                id: entry,
            });
        }

        while let Some(Reverse(current)) = scratch.candidates.pop() {
            if scratch.results.len() >= ef {
                let worst = scratch
                    .results
                    .peek()
                    .expect("result heap is non-empty")
                    .dist_sq;
                if current.dist_sq.total_cmp(&worst) == Ordering::Greater {
                    break;
                }
            }

            for i in 0..self.neighbor_count(layer, current.id) {
                let neighbor = self.neighbor_at(layer, current.id, i);
                if !scratch.mark(epoch, neighbor) {
                    continue;
                }
                let dist = self.distance_to(query, neighbor);
                let acceptable = if scratch.results.len() < ef {
                    true
                } else {
                    let worst = scratch
                        .results
                        .peek()
                        .expect("result heap is non-empty")
                        .dist_sq;
                    dist.total_cmp(&worst) == Ordering::Less
                };
                if !acceptable {
                    continue;
                }
                scratch.candidates.push(Reverse(Candidate {
                    dist_sq: dist,
                    id: neighbor,
                }));
                let neighbor_is_valid = match validity {
                    Some(validity) => self.node_is_valid(neighbor, validity)?,
                    None => true,
                };
                if neighbor_is_valid {
                    if scratch.results.len() == ef {
                        scratch.results.pop();
                    }
                    scratch.results.push(Candidate {
                        dist_sq: dist,
                        id: neighbor,
                    });
                }
            }
        }

        while let Some(candidate) = scratch.results.pop() {
            out.push(candidate);
        }
        out.sort_unstable();
        Ok(())
    }

    /// Algorithm 4: keep a candidate only when it is closer to the query than
    /// to every already selected neighbor. This preserves links in different
    /// directions instead of producing a cluster of near-duplicates.
    ///
    /// Deliberately does not pad the result with pruned candidates
    /// (`keepPrunedConnections`). Padding fills the list with short edges,
    /// which then crowd long-range edges out of the reverse-edge pruning and
    /// badly hurt routing on clustered data.
    fn select_neighbors(&self, candidates: &[Candidate], m: usize, out: &mut Vec<NodeId>) {
        out.clear();
        if candidates.len() <= m {
            out.extend(candidates.iter().map(|c| c.id));
            return;
        }
        for candidate in candidates {
            if out.len() >= m {
                break;
            }
            let mut keep = true;
            for &selected in out.iter() {
                if self.distance_between(candidate.id, selected) < candidate.dist_sq {
                    keep = false;
                    break;
                }
            }
            if keep {
                out.push(candidate.id);
            }
        }
    }

    #[inline]
    fn distance_to(&self, query: &[f32], id: NodeId) -> f32 {
        let start = id as usize * self.dim;
        l2_sq(query, &self.vectors[start..start + self.dim])
    }

    #[inline]
    fn distance_between(&self, a: NodeId, b: NodeId) -> f32 {
        let a_start = a as usize * self.dim;
        let b_start = b as usize * self.dim;
        l2_sq(
            &self.vectors[a_start..a_start + self.dim],
            &self.vectors[b_start..b_start + self.dim],
        )
    }

    #[inline]
    fn node_is_valid(&self, id: NodeId, validity: &RowValidity) -> ILResult<bool> {
        validity.is_valid(self.row_offsets[id as usize] as usize)
    }

    #[inline]
    fn neighbor_count(&self, layer: usize, id: NodeId) -> usize {
        let id = id as usize;
        if layer == 0 {
            self.counts0[id] as usize
        } else {
            self.upper[id].counts[layer - 1] as usize
        }
    }

    #[inline]
    fn neighbor_at(&self, layer: usize, id: NodeId, index: usize) -> NodeId {
        let id = id as usize;
        if layer == 0 {
            self.links0[id * M0 + index]
        } else {
            self.upper[id].links[(layer - 1) * M + index]
        }
    }

    fn set_node_links(&mut self, id: NodeId, layer: usize, links: &[NodeId]) {
        let id = id as usize;
        if layer == 0 {
            let base = id * M0;
            self.links0[base..base + links.len()].copy_from_slice(links);
            self.counts0[id] = links.len() as u8;
        } else {
            let index = layer - 1;
            let node = &mut self.upper[id];
            let base = index * M;
            node.links[base..base + links.len()].copy_from_slice(links);
            node.counts[index] = links.len() as u8;
        }
    }

    fn add_backlink(&mut self, target: NodeId, new_id: NodeId, layer: usize, force: bool) {
        let cap = if layer == 0 { M0 } else { M };
        let count = self.neighbor_count(layer, target);
        if count < cap {
            self.set_neighbor(layer, target, count, new_id);
            self.set_neighbor_count(layer, target, (count + 1) as u8);
            return;
        }

        let new_dist = self.distance_between(target, new_id);
        let mut worst_index = 0usize;
        let mut worst_dist = f32::NEG_INFINITY;
        for index in 0..cap {
            let neighbor = self.neighbor_at(layer, target, index);
            if neighbor == new_id {
                return;
            }
            let dist = self.distance_between(target, neighbor);
            if dist > worst_dist {
                worst_dist = dist;
                worst_index = index;
            }
        }
        if force || new_dist < worst_dist {
            self.set_neighbor(layer, target, worst_index, new_id);
        }
    }

    fn set_neighbor(&mut self, layer: usize, id: NodeId, index: usize, value: NodeId) {
        let id = id as usize;
        if layer == 0 {
            self.links0[id * M0 + index] = value;
        } else {
            self.upper[id].links[(layer - 1) * M + index] = value;
        }
    }

    fn set_neighbor_count(&mut self, layer: usize, id: NodeId, count: u8) {
        let id = id as usize;
        if layer == 0 {
            self.counts0[id] = count;
        } else {
            self.upper[id].counts[layer - 1] = count;
        }
    }

    /// Search the zero layer and return up to `limit` valid nodes ordered by
    /// ascending distance.
    pub(crate) fn search(
        &self,
        query: &[f32],
        ef: usize,
        validity: &RowValidity,
        limit: usize,
    ) -> ILResult<Vec<(NodeId, f32)>> {
        if self.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dim {
            return Err(ILError::invalid_input(format!(
                "HNSW query dimension {} does not match index dimension {}",
                query.len(),
                self.dim
            )));
        }
        let ef = ef.max(1).min(self.len());

        let mut ep = self.entry.expect("non-empty graph has an entry point");
        let mut ep_dist = self.distance_to(query, ep);
        for layer in (1..=self.max_level).rev() {
            (ep, ep_dist) = self.greedy_layer(query, ep, ep_dist, layer);
        }

        let mut out = Vec::with_capacity(ef);
        with_search_scratch(self.len(), ef, |scratch| {
            self.search_layer(query, ep, ef, 0, Some(validity), scratch, &mut out)
        })?;
        out.truncate(limit);
        Ok(out.into_iter().map(|c| (c.id, c.dist_sq)).collect())
    }
}

fn deterministic_level(id: NodeId) -> usize {
    let mut z = LEVEL_SEED ^ (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Uniform in the open interval (0, 1).
    let uniform = ((z >> 11) as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64));
    let level = (-uniform.ln() * (1.0 / (M as f64).ln())).floor() as usize;
    level.min(MAX_LEVEL)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn build(points: &[Vec<f32>], dim: usize, ef_construction: usize) -> HnswGraph {
        let mut vectors = Vec::with_capacity(points.len() * dim);
        let mut row_ids = Vec::with_capacity(points.len());
        let mut row_offsets = Vec::with_capacity(points.len());
        for (index, point) in points.iter().enumerate() {
            assert_eq!(point.len(), dim);
            vectors.extend_from_slice(point);
            row_ids.push(Uuid::from_u128(index as u128));
            row_offsets.push(index as u32);
        }
        HnswGraph::build(vectors, row_ids, row_offsets, dim, ef_construction).unwrap()
    }

    /// Independent ground truth: a serial f64 fold and a real square root.
    /// Deliberately does not call [`l2_sq`] so a bug in the index kernel cannot
    /// make the oracle agree with it.
    fn brute_force_exact(points: &[Vec<f32>], query: &[f32], limit: usize) -> Vec<(usize, f64)> {
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
                (dist_sq.sqrt(), index)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        scored
            .into_iter()
            .take(limit)
            .map(|(dist, index)| (index, dist))
            .collect()
    }

    fn brute_force(points: &[Vec<f32>], query: &[f32], limit: usize) -> Vec<usize> {
        brute_force_exact(points, query, limit)
            .into_iter()
            .map(|(index, _)| index)
            .collect()
    }

    /// Points on a `1 / scale` grid. Sums of their squared differences are
    /// exactly representable in f32 for every dimension and accumulation order
    /// used by the tests, so rank comparisons cannot be perturbed by SIMD
    /// rounding; ties are resolved by id on both sides.
    fn quantized_points(
        rng: &mut SplitMix64,
        count: usize,
        dim: usize,
        scale: f32,
    ) -> Vec<Vec<f32>> {
        (0..count)
            .map(|_| {
                (0..dim)
                    .map(|_| (rng.next_f32() * scale).round() / scale)
                    .collect()
            })
            .collect()
    }

    /// Search with `ef = node_count` must reproduce the exact scan over the
    /// valid rows reachable from the entry point, including the result order.
    /// The graphs used here are connected; the duplicate-vector test below
    /// covers the disconnected degenerate case.
    fn assert_matches_exact_scan(
        graph: &HnswGraph,
        points: &[Vec<f32>],
        validity: &RowValidity,
        queries: &[Vec<f32>],
        limit: usize,
    ) {
        let count = points.len();
        for query in queries {
            let expected: Vec<(usize, f64)> = brute_force_exact(points, query, count)
                .into_iter()
                .filter(|(index, _)| validity.is_valid(*index).unwrap())
                .take(limit)
                .collect();
            let actual = graph.search(query, count, validity, limit).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (&(node_id, dist_sq), &(expected_id, expected_dist)) in actual.iter().zip(&expected)
            {
                assert_eq!(node_id as usize, expected_id);
                let dist = dist_sq.max(0.0).sqrt() as f64;
                let tolerance = expected_dist.abs().max(1.0) * 1e-5;
                assert!(
                    (dist - expected_dist).abs() <= tolerance,
                    "distance mismatch for node {node_id}: {dist} vs {expected_dist}"
                );
            }
        }
    }

    #[test]
    fn empty_graph() {
        let graph = build(&[], 0, 32);
        assert!(graph.is_empty());
        let result = graph.search(&[], 10, &RowValidity::new(0), 5).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn single_node() {
        let graph = build(&[vec![1.0, 2.0]], 2, 32);
        let result = graph
            .search(&[1.0, 2.0], 10, &RowValidity::new(1), 5)
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
        assert_eq!(result[0].1, 0.0);
    }

    #[test]
    fn exact_small_search() {
        let points = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![20.0, 0.0],
            vec![30.0, 0.0],
        ];
        let graph = build(&points, 2, 64);
        let result = graph
            .search(&[9.0, 0.0], 64, &RowValidity::new(4), 2)
            .unwrap();
        assert_eq!(
            result
                .iter()
                .map(|(id, _)| *id as usize)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
    }

    #[test]
    fn respects_row_offsets_and_validity() {
        let points = [vec![0.0], vec![10.0], vec![20.0]];
        let graph = HnswGraph::build(
            points.iter().flatten().copied().collect(),
            points
                .iter()
                .enumerate()
                .map(|(index, _)| Uuid::from_u128(index as u128))
                .collect(),
            vec![5, 7, 11],
            1,
            64,
        )
        .unwrap();

        let mut validity = RowValidity::new(12);
        validity.set(7, false);
        let result = graph.search(&[10.0], 16, &validity, 3).unwrap();
        assert_eq!(
            result
                .iter()
                .map(|(id, _)| *id as usize)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );

        let mut only_second = RowValidity::new(12);
        only_second.set(5, false);
        only_second.set(11, false);
        let result = graph.search(&[0.0], 16, &only_second, 3).unwrap();
        assert_eq!(
            result
                .iter()
                .map(|(id, _)| *id as usize)
                .collect::<Vec<_>>(),
            vec![1]
        );
    }

    #[test]
    fn duplicate_vectors() {
        let points = vec![vec![1.0, 1.0]; 5];
        let graph = build(&points, 2, 64);
        let result = graph
            .search(&[1.0, 1.0], 16, &RowValidity::new(5), 5)
            .unwrap();
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|(_, dist)| *dist == 0.0));
    }

    #[test]
    fn full_ef_search_matches_exact_scan() {
        let mut rng = SplitMix64(0x5EED_0001);
        let dim = 8;
        let count = 300;
        let points = quantized_points(&mut rng, count, dim, 64.0);
        let graph = build(&points, dim, 100);
        let queries = quantized_points(&mut rng, 20, dim, 64.0);

        let all_valid = RowValidity::new(count);
        assert_matches_exact_scan(&graph, &points, &all_valid, &queries, count);

        // Including row 0 (the entry point) forces traversal to start from an
        // invalid node while still returning every live row.
        let mut masked = RowValidity::new(count);
        let masked_rows = [0, 3, 17, 42, 99, 150, 250, 299];
        for row in masked_rows {
            masked.set(row, false);
        }
        assert_matches_exact_scan(&graph, &points, &masked, &queries, count);
        assert_matches_exact_scan(&graph, &points, &masked, &queries, 10);
    }

    #[test]
    fn full_ef_search_matches_exact_scan_across_shapes() {
        let mut rng = SplitMix64(0x5EED_0002);
        // Dimensions straddle the scalar/SIMD threshold (32) and both tail
        // classes; counts include degenerate and multi-block sizes.
        for dim in [1usize, 3, 7, 16, 31, 32, 33, 64, 128] {
            for count in [2usize, 17, 100, 300] {
                let points = quantized_points(&mut rng, count, dim, 64.0);
                let graph = build(&points, dim, 100);
                let queries = quantized_points(&mut rng, 4, dim, 64.0);
                let validity = RowValidity::new(count);
                assert_matches_exact_scan(&graph, &points, &validity, &queries, count);
            }
        }
    }

    /// Nodes reachable from the entry point over layer-0 directed links.
    fn reachable_from_entry(graph: &HnswGraph) -> Vec<bool> {
        let mut seen = vec![false; graph.len()];
        let Some(entry) = graph.entry else {
            return seen;
        };
        seen[entry as usize] = true;
        let mut stack = vec![entry];
        while let Some(node) = stack.pop() {
            for index in 0..graph.neighbor_count(0, node) {
                let neighbor = graph.neighbor_at(0, node, index);
                if !seen[neighbor as usize] {
                    seen[neighbor as usize] = true;
                    stack.push(neighbor);
                }
            }
        }
        seen
    }

    /// Exact duplicates tie every distance, so the id tie-break makes every
    /// node select the same low-id hubs and a full list never accepts an
    /// equal-distance reverse link. With 300 identical vectors only ~34 are
    /// reachable today. Search must still be exact over the reachable
    /// component, and it must stay usable for ordinary top-k limits.
    #[test]
    fn duplicate_vectors_are_exact_within_the_reachable_component() {
        let points = vec![vec![1.0f32, 1.0]; 300];
        let graph = build(&points, 2, 64);
        let validity = RowValidity::new(points.len());
        let reachable = reachable_from_entry(&graph);
        let reachable_count = reachable
            .iter()
            .filter(|is_reachable| **is_reachable)
            .count();
        assert!(
            reachable_count >= 10,
            "too few reachable duplicates: {reachable_count}"
        );

        let query = vec![1.0, 1.0];
        let actual = graph
            .search(&query, points.len(), &validity, points.len())
            .unwrap();
        let expected: Vec<usize> = (0..points.len()).filter(|id| reachable[*id]).collect();
        assert_eq!(
            actual
                .iter()
                .map(|(id, _)| *id as usize)
                .collect::<Vec<_>>(),
            expected
        );
        assert!(actual.iter().all(|(_, dist)| *dist == 0.0));

        // A typical top-k query must still be fully served.
        let result = graph.search(&query, 50, &validity, 10).unwrap();
        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|(_, dist)| *dist == 0.0));
    }

    #[test]
    fn decoded_graph_matches_exact_scan() {
        let mut rng = SplitMix64(0x5EED_0003);
        let dim = 16;
        let count = 200;
        let points = quantized_points(&mut rng, count, dim, 64.0);
        let graph = build(&points, dim, 100);
        let encoded = crate::serialization::encode_graph(&graph).unwrap();
        let decoded = crate::serialization::decode_graph(&encoded).unwrap();
        let queries = quantized_points(&mut rng, 10, dim, 64.0);
        let validity = RowValidity::new(count);
        assert_matches_exact_scan(&decoded, &points, &validity, &queries, count);
    }

    #[test]
    fn recall_matches_brute_force() {
        let mut rng = SplitMix64(0xDEAD_BEEF);
        let dim = 16;
        let count = 1000;
        let points: Vec<Vec<f32>> = (0..count)
            .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
            .collect();
        let graph = build(&points, dim, 200);
        let validity = RowValidity::new(count);

        let mut total_recall = 0.0f64;
        let queries = 32;
        for _ in 0..queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
            let expected = brute_force(&points, &query, 10);
            let actual: Vec<usize> = graph
                .search(&query, 100, &validity, 10)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id as usize)
                .collect();
            assert_eq!(actual.len(), 10);
            let hits = actual.iter().filter(|id| expected.contains(id)).count();
            total_recall += hits as f64 / 10.0;
        }
        let recall = total_recall / queries as f64;
        assert!(recall >= 0.95, "recall too low: {recall}");
    }

    #[test]
    #[ignore = "recall sweep"]
    fn recall_sweep() {
        let mut rng = SplitMix64(0xDEAD_BEEF);
        let dim = 64;
        let count = 5000;
        let points: Vec<Vec<f32>> = (0..count)
            .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
            .collect();
        let graph = build(&points, dim, 200);
        let validity = RowValidity::new(count);
        let queries = 50;
        let test_queries: Vec<Vec<f32>> = (0..queries)
            .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
            .collect();
        let expected: Vec<Vec<usize>> = test_queries
            .iter()
            .map(|query| brute_force(&points, query, 10))
            .collect();
        for ef in [10usize, 50, 100] {
            let mut recall = 0.0f64;
            for (query, expected) in test_queries.iter().zip(&expected) {
                let actual = graph.search(query, ef, &validity, 10).unwrap();
                let hits = actual
                    .iter()
                    .filter(|(id, _)| expected.contains(&(*id as usize)))
                    .count();
                recall += hits as f64 / 10.0;
            }
            eprintln!("ef={ef} recall={:.4}", recall / queries as f64);
        }
    }

    /// Clustered data exercises long-range links and cross-cluster routing;
    /// uniform random data does not. Selections that fill lists with short
    /// edges (for example `keepPrunedConnections`) pass on uniform data but
    /// collapse here.
    #[test]
    fn recall_on_clustered_data() {
        let mut rng = SplitMix64(0xABBA_1234);
        let dim = 16;
        let clusters = 20;
        let per_cluster = 100;
        let count = clusters * per_cluster;
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| (0..dim).map(|_| rng.next_f32() * 4.0).collect())
            .collect();
        let points: Vec<Vec<f32>> = (0..count)
            .map(|index| {
                let center = &centers[index % clusters];
                center
                    .iter()
                    .map(|value| value + rng.next_f32() * 0.2)
                    .collect()
            })
            .collect();
        let graph = build(&points, dim, 100);
        let validity = RowValidity::new(count);

        let queries = 20;
        let mut total_recall = 0.0f64;
        for index in 0..queries {
            let center = &centers[index % clusters];
            let query: Vec<f32> = center
                .iter()
                .map(|value| value + rng.next_f32() * 0.2)
                .collect();
            let expected = brute_force(&points, &query, 10);
            let actual: Vec<usize> = graph
                .search(&query, 50, &validity, 10)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id as usize)
                .collect();
            assert_eq!(actual.len(), 10);
            let hits = actual.iter().filter(|id| expected.contains(id)).count();
            total_recall += hits as f64 / 10.0;
        }
        let recall = total_recall / queries as f64;
        assert!(recall >= 0.98, "clustered recall too low: {recall}");
    }

    #[test]
    #[ignore = "performance smoke test"]
    fn bench_build_and_search() {
        use std::time::Instant;

        let mut rng = SplitMix64(0x1234_5678);
        let dim = 256;
        let count = 20_000;
        let points: Vec<Vec<f32>> = (0..count)
            .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
            .collect();

        let start = Instant::now();
        let graph = build(&points, dim, 200);
        eprintln!("built {count} x {dim} in {:?}", start.elapsed());

        let validity = RowValidity::new(count);
        let queries: Vec<Vec<f32>> = (0..200)
            .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
            .collect();
        let start = Instant::now();
        let mut total = 0usize;
        for query in &queries {
            total += graph.search(query, 100, &validity, 10).unwrap().len();
        }
        let elapsed = start.elapsed();
        eprintln!(
            "200 queries in {:?} ({:?} per query), {} results",
            elapsed,
            elapsed / queries.len() as u32,
            total
        );
        assert_eq!(total, queries.len() * 10);
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(
            HnswGraph::build(vec![0.0], vec![], vec![], 1, 16).is_err(),
            "row count mismatch must be rejected"
        );
        assert!(
            HnswGraph::build(vec![0.0], vec![Uuid::nil()], vec![0], 0, 16).is_err(),
            "zero dimension must be rejected"
        );
        assert!(
            HnswGraph::build(vec![f32::NAN], vec![Uuid::nil()], vec![0], 1, 16).is_err(),
            "non-finite vectors must be rejected"
        );
    }
}
