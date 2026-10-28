use indexlake::{ILError, ILResult};
use uuid::Uuid;

use crate::graph::{HnswGraph, M, M0, MAX_LEVEL, NONE, UpperNode};

const GRAPH_MAGIC: &[u8; 8] = b"ILHNSW01";
const GRAPH_VERSION: u32 = 1;

const PENDING_MAGIC: &[u8; 4] = b"IHR1";
const PENDING_VERSION: u32 = 1;

/// Rows buffered between a batch append and a build / inline serialization.
pub(crate) struct PendingRows {
    pub dim: Option<usize>,
    pub row_ids: Vec<Uuid>,
    pub present: Vec<bool>,
    /// Values of present rows only, concatenated.
    pub vectors: Vec<f32>,
}

fn corrupt(message: impl Into<String>) -> ILError {
    ILError::index(format!("Invalid HNSW index data: {}", message.into()))
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, len: usize) -> ILResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| corrupt("offset overflow"))?;
        if end > self.data.len() {
            return Err(corrupt(format!(
                "unexpected end of data: need {len} bytes at offset {}, have {}",
                self.pos,
                self.data.len()
            )));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> ILResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> ILResult<u32> {
        let bytes: [u8; 4] = self.take(4)?.try_into().expect("length checked");
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> ILResult<u64> {
        let bytes: [u8; 8] = self.take(8)?.try_into().expect("length checked");
        Ok(u64::from_le_bytes(bytes))
    }

    fn finish(&self) -> ILResult<()> {
        if self.pos == self.data.len() {
            Ok(())
        } else {
            Err(corrupt(format!(
                "{} trailing bytes after payload",
                self.data.len() - self.pos
            )))
        }
    }
}

#[inline]
fn push_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

#[inline]
fn push_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

#[inline]
fn push_f32_slice(buf: &mut Vec<u8>, values: &[f32]) {
    #[cfg(target_endian = "little")]
    {
        buf.extend_from_slice(bytemuck::cast_slice(values));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for value in values {
            buf.extend_from_slice(&value.to_le_bytes());
        }
    }
}

#[inline]
fn push_u32_slice(buf: &mut Vec<u8>, values: &[u32]) {
    #[cfg(target_endian = "little")]
    {
        buf.extend_from_slice(bytemuck::cast_slice(values));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for value in values {
            buf.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn read_f32_slice(bytes: &[u8]) -> ILResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(corrupt("float payload is not a multiple of 4 bytes"));
    }
    #[cfg(target_endian = "little")]
    {
        Ok(bytemuck::pod_collect_to_vec::<u8, f32>(bytes))
    }
    #[cfg(not(target_endian = "little"))]
    {
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk length")))
            .collect())
    }
}

fn read_u32_slice(bytes: &[u8]) -> ILResult<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(corrupt("u32 payload is not a multiple of 4 bytes"));
    }
    #[cfg(target_endian = "little")]
    {
        Ok(bytemuck::pod_collect_to_vec::<u8, u32>(bytes))
    }
    #[cfg(not(target_endian = "little"))]
    {
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunk length")))
            .collect())
    }
}

fn checked_mul(a: usize, b: usize) -> ILResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| corrupt("payload size overflow"))
}

pub(crate) fn encode_graph(graph: &HnswGraph) -> ILResult<Vec<u8>> {
    let node_count = graph.len();
    let upper_level_count: usize = graph.upper.iter().map(|node| node.counts.len()).sum();
    let upper_link_count: usize = graph
        .upper
        .iter()
        .map(|node| node.links.len())
        .sum::<usize>();
    let header_size = 8 + 4 * 4 + 8 + 4 + 4 + 8;
    let payload_size = checked_mul(node_count, graph.dim)? * 4
        + checked_mul(node_count, 16)?
        + checked_mul(node_count, 4)?
        + node_count
        + node_count
        + checked_mul(node_count, M0)? * 4
        + upper_level_count
        + checked_mul(upper_link_count, 4)?;
    let mut buf = Vec::with_capacity(header_size + payload_size);

    buf.extend_from_slice(GRAPH_MAGIC);
    push_u32(&mut buf, GRAPH_VERSION);
    push_u32(&mut buf, graph.dim as u32);
    push_u32(&mut buf, M0 as u32);
    push_u32(&mut buf, M as u32);
    push_u64(&mut buf, node_count as u64);
    push_u32(&mut buf, graph.entry.unwrap_or(NONE));
    push_u32(&mut buf, graph.max_level as u32);
    push_u64(&mut buf, 0); // reserved

    push_f32_slice(&mut buf, &graph.vectors);
    for row_id in &graph.row_ids {
        buf.extend_from_slice(row_id.as_bytes());
    }
    push_u32_slice(&mut buf, &graph.row_offsets);
    buf.extend_from_slice(&graph.levels);
    buf.extend_from_slice(&graph.counts0);
    push_u32_slice(&mut buf, &graph.links0);
    for node in &graph.upper {
        for (layer, &count) in node.counts.iter().enumerate() {
            buf.push(count);
            let base = layer * M;
            for &neighbor in &node.links[base..base + count as usize] {
                push_u32(&mut buf, neighbor);
            }
        }
    }

    Ok(buf)
}

pub(crate) fn decode_graph(data: &[u8]) -> ILResult<HnswGraph> {
    let mut reader = Reader::new(data);

    if reader.take(GRAPH_MAGIC.len())? != GRAPH_MAGIC.as_slice() {
        return Err(corrupt("bad magic"));
    }
    let version = reader.u32()?;
    if version != GRAPH_VERSION {
        return Err(corrupt(format!(
            "unsupported format version {version}, expected {GRAPH_VERSION}"
        )));
    }
    let dim = reader.u32()? as usize;
    let m0 = reader.u32()? as usize;
    let m = reader.u32()? as usize;
    if m0 != M0 || m != M {
        return Err(corrupt(format!(
            "graph was built with m0={m0}, m={m}, expected m0={M0}, m={M}"
        )));
    }
    let node_count = reader.u64()? as usize;
    if node_count >= NONE as usize {
        return Err(corrupt("node count exceeds addressable range"));
    }
    let entry_raw = reader.u32()?;
    let max_level = reader.u32()? as usize;
    let _reserved = reader.u64()?;

    if node_count > 0 && dim == 0 {
        return Err(corrupt("non-empty graph has zero dimension"));
    }
    if max_level > MAX_LEVEL {
        return Err(corrupt(format!(
            "max level {max_level} exceeds {MAX_LEVEL}"
        )));
    }
    let entry = if node_count == 0 {
        if entry_raw != NONE {
            return Err(corrupt("empty graph has an entry point"));
        }
        None
    } else {
        if entry_raw >= node_count as u32 {
            return Err(corrupt(format!("entry point {entry_raw} out of range")));
        }
        Some(entry_raw)
    };

    let vector_bytes = checked_mul(node_count, dim)? * 4;
    let vectors = read_f32_slice(reader.take(vector_bytes)?)?;
    if vectors.iter().any(|value| !value.is_finite()) {
        return Err(corrupt("vectors contain non-finite values"));
    }

    let mut row_ids = Vec::with_capacity(node_count);
    let row_id_bytes = reader.take(checked_mul(node_count, 16)?)?;
    let (row_id_chunks, _) = row_id_bytes.as_chunks::<16>();
    for chunk in row_id_chunks {
        row_ids.push(Uuid::from_bytes(*chunk));
    }

    let row_offsets = read_u32_slice(reader.take(checked_mul(node_count, 4)?)?)?;
    let levels = reader.take(node_count)?.to_vec();
    for &level in &levels {
        if level as usize > MAX_LEVEL {
            return Err(corrupt(format!("node level {level} exceeds {MAX_LEVEL}")));
        }
    }

    let counts0 = reader.take(node_count)?.to_vec();
    for &count in &counts0 {
        if count as usize > M0 {
            return Err(corrupt(format!("layer-0 degree {count} exceeds {M0}")));
        }
    }

    let links0 = read_u32_slice(reader.take(checked_mul(node_count, M0)? * 4)?)?;
    for (node, &count) in counts0.iter().enumerate() {
        let base = node * M0;
        for index in 0..M0 {
            let link = links0[base + index];
            if index < count as usize {
                if link == NONE || link as usize >= node_count {
                    return Err(corrupt(format!(
                        "invalid layer-0 link {link} at node {node}"
                    )));
                }
            } else if link != NONE {
                return Err(corrupt(format!(
                    "unused layer-0 slot at node {node} is not empty"
                )));
            }
        }
    }

    let mut upper = Vec::with_capacity(node_count);
    for (node, &level) in levels.iter().enumerate() {
        let level = level as usize;
        let mut links = vec![NONE; level * M];
        let mut counts = vec![0u8; level];
        for (layer_index, count_slot) in counts.iter_mut().enumerate() {
            let count = reader.u8()? as usize;
            if count > M {
                return Err(corrupt(format!(
                    "layer {} degree {count} exceeds {M}",
                    layer_index + 1
                )));
            }
            *count_slot = count as u8;
            let base = layer_index * M;
            for index in 0..count {
                let link = reader.u32()?;
                if link as usize >= node_count {
                    return Err(corrupt(format!(
                        "layer {} link {link} at node {node} out of range",
                        layer_index + 1
                    )));
                }
                if (levels[link as usize] as usize) < layer_index + 1 {
                    return Err(corrupt(format!(
                        "layer {} link {link} at node {node} has a lower level",
                        layer_index + 1
                    )));
                }
                links[base + index] = link;
            }
        }
        upper.push(UpperNode { links, counts });
    }

    reader.finish()?;

    if let Some(entry) = entry {
        if levels[entry as usize] as usize != max_level {
            return Err(corrupt("entry point level does not match max level"));
        }
        let actual_max = levels.iter().copied().max().unwrap_or(0) as usize;
        if actual_max != max_level {
            return Err(corrupt("stored max level does not match node levels"));
        }
    } else if max_level != 0 {
        return Err(corrupt("empty graph has a non-zero max level"));
    }

    Ok(HnswGraph {
        dim,
        vectors,
        row_ids,
        row_offsets,
        levels,
        links0,
        counts0,
        upper,
        entry,
        max_level,
    })
}

#[cfg(test)]
pub(crate) fn encode_pending(rows: &PendingRows) -> ILResult<Vec<u8>> {
    let mut buf = Vec::new();
    write_pending(
        &mut buf,
        rows.dim,
        &rows.row_ids,
        &rows.present,
        &rows.vectors,
    )?;
    Ok(buf)
}

pub(crate) fn write_pending(
    buf: &mut Vec<u8>,
    dim: Option<usize>,
    row_ids: &[Uuid],
    present: &[bool],
    vectors: &[f32],
) -> ILResult<()> {
    if row_ids.len() != present.len() {
        return Err(ILError::index(
            "HNSW row id count does not match presence flags",
        ));
    }
    let dim_raw = dim.unwrap_or(0);
    let present_count = present.iter().filter(|is_present| **is_present).count();
    if vectors.len() != checked_mul(present_count, dim_raw)? {
        return Err(ILError::index(
            "HNSW pending vector buffer does not match row metadata",
        ));
    }

    let row_count = row_ids.len();
    let bitset_len = row_count.div_ceil(8);
    buf.reserve(4 + 4 + 4 + 8 + row_count * 16 + bitset_len + vectors.len() * 4);
    buf.extend_from_slice(PENDING_MAGIC);
    push_u32(buf, PENDING_VERSION);
    push_u32(buf, dim_raw as u32);
    push_u64(buf, row_count as u64);
    for row_id in row_ids {
        buf.extend_from_slice(row_id.as_bytes());
    }

    let mut bitset = vec![0u8; bitset_len];
    for (index, is_present) in present.iter().enumerate() {
        if *is_present {
            bitset[index / 8] |= 1 << (index % 8);
        }
    }
    buf.extend_from_slice(&bitset);
    push_f32_slice(buf, vectors);
    Ok(())
}

pub(crate) fn decode_pending(data: &[u8]) -> ILResult<PendingRows> {
    let mut reader = Reader::new(data);
    if reader.take(PENDING_MAGIC.len())? != PENDING_MAGIC.as_slice() {
        return Err(corrupt("bad pending-rows magic"));
    }
    let version = reader.u32()?;
    if version != PENDING_VERSION {
        return Err(corrupt(format!(
            "unsupported pending-rows version {version}"
        )));
    }
    let dim_raw = reader.u32()? as usize;
    let row_count = reader.u64()? as usize;
    if row_count > NONE as usize {
        return Err(corrupt("pending row count exceeds addressable range"));
    }

    let row_id_bytes = reader.take(checked_mul(row_count, 16)?)?;
    let mut row_ids = Vec::with_capacity(row_count);
    let (row_id_chunks, _) = row_id_bytes.as_chunks::<16>();
    for chunk in row_id_chunks {
        row_ids.push(Uuid::from_bytes(*chunk));
    }

    let bitset = reader.take(row_count.div_ceil(8))?;
    let mut present = Vec::with_capacity(row_count);
    let mut present_count = 0usize;
    for index in 0..row_count {
        let is_present = bitset[index / 8] & (1 << (index % 8)) != 0;
        if is_present {
            present_count += 1;
        }
        present.push(is_present);
    }

    let dim = if dim_raw == 0 { None } else { Some(dim_raw) };
    if dim.is_none() && present_count > 0 {
        return Err(corrupt("pending rows contain vectors without a dimension"));
    }
    let vector_bytes = checked_mul(present_count, dim_raw)? * 4;
    let vectors = read_f32_slice(reader.take(vector_bytes)?)?;
    reader.finish()?;

    Ok(PendingRows {
        dim,
        row_ids,
        present,
        vectors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::HnswGraph;

    fn sample_graph() -> HnswGraph {
        let points = vec![
            vec![0.0f32, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
        ];
        let n = points.len();
        HnswGraph::build(
            points.into_iter().flatten().collect(),
            (0..n).map(|index| Uuid::from_u128(index as u128)).collect(),
            (0..n as u32).collect(),
            2,
            32,
        )
        .unwrap()
    }

    #[test]
    fn graph_roundtrip() {
        let graph = sample_graph();
        let encoded = encode_graph(&graph).unwrap();
        let decoded = decode_graph(&encoded).unwrap();
        assert_eq!(decoded.dim, graph.dim);
        assert_eq!(decoded.row_ids, graph.row_ids);
        assert_eq!(decoded.row_offsets, graph.row_offsets);
        assert_eq!(decoded.vectors, graph.vectors);
        assert_eq!(decoded.links0, graph.links0);
        assert_eq!(decoded.counts0, graph.counts0);
        assert_eq!(decoded.entry, graph.entry);
        assert_eq!(decoded.max_level, graph.max_level);

        let query = [0.5f32, 0.5];
        let validity = indexlake::index::RowValidity::new(4);
        let before = graph.search(&query, 16, &validity, 4).unwrap();
        let after = decoded.search(&query, 16, &validity, 4).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn rejects_truncated_graph() {
        let graph = sample_graph();
        let encoded = encode_graph(&graph).unwrap();
        assert!(decode_graph(&encoded[..encoded.len() - 3]).is_err());
        assert!(decode_graph(b"not a graph").is_err());
    }

    #[test]
    fn empty_graph_roundtrip() {
        let graph = HnswGraph::build(Vec::new(), Vec::new(), Vec::new(), 0, 16).unwrap();
        let encoded = encode_graph(&graph).unwrap();
        let decoded = decode_graph(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn pending_roundtrip() {
        let rows = PendingRows {
            dim: Some(3),
            row_ids: vec![Uuid::from_u128(1), Uuid::from_u128(2), Uuid::from_u128(3)],
            present: vec![true, false, true],
            vectors: vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
        };
        let encoded = encode_pending(&rows).unwrap();
        let decoded = decode_pending(&encoded).unwrap();
        assert_eq!(decoded.dim, Some(3));
        assert_eq!(decoded.row_ids, rows.row_ids);
        assert_eq!(decoded.present, rows.present);
        assert_eq!(decoded.vectors, rows.vectors);
    }

    #[test]
    fn pending_all_null_roundtrip() {
        let rows = PendingRows {
            dim: None,
            row_ids: vec![Uuid::from_u128(1), Uuid::from_u128(2)],
            present: vec![false, false],
            vectors: Vec::new(),
        };
        let encoded = encode_pending(&rows).unwrap();
        let decoded = decode_pending(&encoded).unwrap();
        assert_eq!(decoded.dim, None);
        assert_eq!(decoded.present, vec![false, false]);
        assert!(decoded.vectors.is_empty());
    }
}
