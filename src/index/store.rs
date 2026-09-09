//! Core index storage.
//!
//! Manages a vector index (HNSW + fallback brute-force KNN) for semantic search
//! and an inverted text index for keyword search. Provides build, query, save,
//! and load operations.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::INDEX_VERSION;
use crate::error::IndexError;
use crate::index::{IndexEntry, SearchHit};
use crate::parser::{CallSite, CodeChunk};
use crate::search::references::{CallerHit, REFERENCES_DISCLAIMER, ReferenceHit, score_reference};
use crate::search::symbol::{SymbolSummary, is_indexable_symbol, normalize_symbol_name};

// ============================================================
// HNSW Point wrapper
// ============================================================

/// Wrapper around `Vec<f32>` implementing `instant_distance::Point` for HNSW.
///
/// Uses cosine distance (1 - cosine_similarity) as the distance metric,
/// which is appropriate for L2-normalized embedding vectors.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EmbeddingPoint(Vec<f32>);

#[derive(serde::Serialize, serde::Deserialize)]
struct HnswSidecar {
    version: u32,
    vector_fingerprint: [u8; 32],
    hnsw: instant_distance::HnswMap<EmbeddingPoint, u64>,
}

#[derive(serde::Serialize)]
struct HnswSidecarRef<'a> {
    version: u32,
    vector_fingerprint: [u8; 32],
    hnsw: &'a instant_distance::HnswMap<EmbeddingPoint, u64>,
}

const HNSW_SIDECAR_VERSION: u32 = 1;
const HNSW_FILENAME: &str = "hnsw.bin";
const MENTIONS_SIDECAR_VERSION: u32 = 1;
const MENTIONS_FILENAME: &str = "mentions.bin";
const MENTIONS_MAGIC: &[u8; 8] = b"SEEKRREF";
const INDEX_FILENAME: &str = "index.bin";
const INDEX_MAGIC: &[u8; 8] = b"SEEKRIDX";

#[derive(serde::Serialize, serde::Deserialize)]
struct MentionsSidecar {
    version: u32,
    chunk_fingerprint: [u8; 32],
    call_sites: HashMap<u64, CallSite>,
    file_to_call_sites: HashMap<PathBuf, Vec<u64>>,
}
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

struct Bm25Stats {
    document_lengths: HashMap<u64, usize>,
    average_document_length: f32,
}

impl instant_distance::Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // For L2-normalized vectors, cosine_similarity = dot product.
        // Distance = 1 - similarity (lower is closer).
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        1.0 - dot
    }
}

/// Contiguous row-major embedding storage with O(1) id lookup.
#[derive(Debug, Clone, Default)]
struct VectorStore {
    dim: usize,
    data: Vec<f32>,
    row_ids: Vec<u64>,
    id_to_row: HashMap<u64, u32>,
}

impl VectorStore {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            data: Vec::new(),
            row_ids: Vec::new(),
            id_to_row: HashMap::new(),
        }
    }

    fn from_parts(dim: usize, data: Vec<f32>, row_ids: Vec<u64>) -> Result<Self, IndexError> {
        if dim == 0 {
            return Err(IndexError::Corrupted(
                "embedding dimension must be non-zero".to_string(),
            ));
        }
        if data.len() != row_ids.len().saturating_mul(dim) {
            return Err(IndexError::Corrupted(format!(
                "vector payload length mismatch: data={}, rows={}, dim={}",
                data.len(),
                row_ids.len(),
                dim
            )));
        }
        let mut id_to_row = HashMap::with_capacity(row_ids.len());
        for (row, &chunk_id) in row_ids.iter().enumerate() {
            if id_to_row.insert(chunk_id, row as u32).is_some() {
                return Err(IndexError::Corrupted(format!(
                    "duplicate vector row for chunk id {chunk_id}"
                )));
            }
        }
        Ok(Self {
            dim,
            data,
            row_ids,
            id_to_row,
        })
    }

    fn len(&self) -> usize {
        self.row_ids.len()
    }

    fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }

    fn get(&self, chunk_id: u64) -> Option<&[f32]> {
        let row = *self.id_to_row.get(&chunk_id)? as usize;
        let start = row * self.dim;
        Some(&self.data[start..start + self.dim])
    }

    fn insert(&mut self, chunk_id: u64, embedding: &[f32]) -> Result<(), IndexError> {
        if embedding.len() != self.dim {
            return Err(IndexError::Corrupted(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dim,
                embedding.len()
            )));
        }
        if let Some(&row) = self.id_to_row.get(&chunk_id) {
            let start = row as usize * self.dim;
            self.data[start..start + self.dim].copy_from_slice(embedding);
            return Ok(());
        }
        let row = self.row_ids.len() as u32;
        self.data.extend_from_slice(embedding);
        self.row_ids.push(chunk_id);
        self.id_to_row.insert(chunk_id, row);
        Ok(())
    }

    fn remove(&mut self, chunk_id: u64) {
        let Some(row) = self.id_to_row.remove(&chunk_id) else {
            return;
        };
        let row = row as usize;
        let last = self.row_ids.len() - 1;
        if row != last {
            let moved_id = self.row_ids[last];
            let src = last * self.dim;
            let dst = row * self.dim;
            self.data.copy_within(src..src + self.dim, dst);
            self.row_ids[row] = moved_id;
            self.id_to_row.insert(moved_id, row as u32);
        }
        self.row_ids.pop();
        self.data.truncate(self.row_ids.len() * self.dim);
    }

    fn sorted_entries(&self) -> Vec<(u64, &[f32])> {
        let mut rows: Vec<usize> = (0..self.row_ids.len()).collect();
        rows.sort_unstable_by_key(|&row| self.row_ids[row]);
        rows.into_iter()
            .map(|row| {
                let start = row * self.dim;
                (self.row_ids[row], &self.data[start..start + self.dim])
            })
            .collect()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct IndexPayloadV4 {
    embedding_dim: usize,
    chunk_count: usize,
    vector_data: Vec<f32>,
    vector_row_ids: Vec<u64>,
    inverted_index: HashMap<String, Vec<(u64, u32)>>,
    chunks: HashMap<u64, CodeChunk>,
}

/// The main index structure holding both vector and text indices.
///
/// Uses HNSW (Hierarchical Navigable Small Worlds) for fast approximate
/// nearest neighbor search, with brute-force KNN as fallback.
pub struct SeekrIndex {
    version: u32,
    vectors: VectorStore,
    inverted_index: HashMap<String, Vec<(u64, u32)>>,
    chunks: HashMap<u64, CodeChunk>,
    call_sites: HashMap<u64, CallSite>,
    file_to_call_sites: HashMap<PathBuf, Vec<u64>>,
    embedding_dim: usize,
    chunk_count: usize,
    hnsw: OnceLock<instant_distance::HnswMap<EmbeddingPoint, u64>>,
    index_dir: Option<PathBuf>,
    bm25_stats: OnceLock<Bm25Stats>,
    symbol_index: OnceLock<HashMap<String, Vec<u64>>>,
    mention_index: OnceLock<HashMap<String, Vec<u64>>>,
}

impl std::fmt::Debug for SeekrIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeekrIndex")
            .field("version", &self.version)
            .field("embedding_dim", &self.embedding_dim)
            .field("chunk_count", &self.chunk_count)
            .field("vectors_len", &self.vectors.len())
            .field("hnsw", &self.hnsw.get().map(|_| "Some(<HnswMap>)"))
            .finish()
    }
}

impl SeekrIndex {
    /// Create a new empty index.
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            version: INDEX_VERSION,
            vectors: VectorStore::new(embedding_dim),
            inverted_index: HashMap::new(),
            chunks: HashMap::new(),
            call_sites: HashMap::new(),
            file_to_call_sites: HashMap::new(),
            embedding_dim,
            chunk_count: 0,
            hnsw: OnceLock::new(),
            index_dir: None,
            bm25_stats: OnceLock::new(),
            symbol_index: OnceLock::new(),
            mention_index: OnceLock::new(),
        }
    }

    pub fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn format_version(&self) -> u32 {
        self.version
    }

    pub fn iter_chunks(&self) -> impl Iterator<Item = (&u64, &CodeChunk)> {
        self.chunks.iter()
    }

    pub fn embedding(&self, chunk_id: u64) -> Option<&[f32]> {
        self.vectors.get(chunk_id)
    }

    pub fn set_format_version(&mut self, version: u32) {
        self.version = version;
    }

    /// Add an entry to the index.
    pub fn add_entry(&mut self, entry: IndexEntry, chunk: CodeChunk) {
        self.try_add_entry(entry, chunk)
            .expect("content-addressed chunk ID collision");
    }

    /// Add an entry while reporting the vanishingly rare hash collision.
    pub fn try_add_entry(&mut self, entry: IndexEntry, chunk: CodeChunk) -> Result<(), IndexError> {
        self.hnsw.take();
        self.bm25_stats.take();
        self.symbol_index.take();
        self.mention_index.take();
        let chunk_id = entry.chunk_id;

        if let Some(existing) = self.chunks.get(&chunk_id) {
            let same_identity = existing.file_path == chunk.file_path
                && existing.kind == chunk.kind
                && existing.name == chunk.name
                && existing.body == chunk.body;
            if !same_identity {
                return Err(IndexError::ChunkIdCollision {
                    chunk_id,
                    existing_path: existing.file_path.clone(),
                    incoming_path: chunk.file_path.clone(),
                });
            }
            self.remove_chunk(chunk_id);
        }

        self.vectors.insert(chunk_id, &entry.embedding)?;

        for token in &entry.text_tokens {
            let posting_list = self.inverted_index.entry(token.clone()).or_default();
            if let Some(existing) = posting_list.iter_mut().find(|(id, _)| *id == chunk_id) {
                existing.1 += 1;
            } else {
                posting_list.push((chunk_id, 1));
            }
        }

        self.chunks.insert(chunk_id, chunk);
        self.chunk_count = self.chunks.len();
        Ok(())
    }

    /// Remove a chunk from the index by ID.
    pub fn remove_chunk(&mut self, chunk_id: u64) {
        self.hnsw.take();
        self.bm25_stats.take();
        self.symbol_index.take();
        self.mention_index.take();
        self.vectors.remove(chunk_id);

        self.inverted_index.retain(|_token, posting_list| {
            posting_list.retain(|(id, _)| *id != chunk_id);
            !posting_list.is_empty()
        });

        self.chunks.remove(&chunk_id);
        self.chunk_count = self.chunks.len();
        self.detach_call_sites_for_chunk(chunk_id);
    }

    /// Replace all call-site mentions for a file (incremental update).
    pub fn replace_file_call_sites(&mut self, file_path: &Path, sites: Vec<CallSite>) {
        self.mention_index.take();
        self.remove_file_call_sites(file_path);
        if sites.is_empty() {
            return;
        }
        let mut ids = Vec::with_capacity(sites.len());
        for site in sites {
            ids.push(site.id);
            self.call_sites.insert(site.id, site);
        }
        self.file_to_call_sites.insert(file_path.to_path_buf(), ids);
    }

    /// Remove all call-site mentions for a file.
    pub fn remove_file_call_sites(&mut self, file_path: &Path) {
        self.mention_index.take();
        if let Some(ids) = self.file_to_call_sites.remove(file_path) {
            for id in ids {
                self.call_sites.remove(&id);
            }
        }
    }

    fn detach_call_sites_for_chunk(&mut self, chunk_id: u64) {
        let affected: Vec<u64> = self
            .call_sites
            .iter()
            .filter_map(|(id, site)| {
                if site.enclosing_chunk_id == Some(chunk_id) {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        if affected.is_empty() {
            return;
        }
        for id in &affected {
            if let Some(site) = self.call_sites.get_mut(id) {
                site.enclosing_chunk_id = None;
            }
        }
    }

    /// Remove multiple chunks by their IDs.
    pub fn remove_chunks(&mut self, chunk_ids: &[u64]) {
        for &chunk_id in chunk_ids {
            self.remove_chunk(chunk_id);
        }
    }

    /// Build the index from chunks and their embeddings.
    pub fn build_from(chunks: &[CodeChunk], embeddings: &[Vec<f32>], embedding_dim: usize) -> Self {
        Self::try_build_from(chunks, embeddings, embedding_dim)
            .expect("content-addressed chunk ID collision")
    }

    /// Build an index and return a structured error if chunk IDs collide.
    pub fn try_build_from(
        chunks: &[CodeChunk],
        embeddings: &[Vec<f32>],
        embedding_dim: usize,
    ) -> Result<Self, IndexError> {
        let mut index = Self::new(embedding_dim);

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            let text_tokens = tokenize_for_index(&chunk.body);

            let entry = IndexEntry {
                chunk_id: chunk.id,
                embedding: embedding.clone(),
                text_tokens,
            };

            index.try_add_entry(entry, chunk.clone())?;
        }

        index.rebuild_hnsw();
        Ok(index)
    }

    /// Rebuild the HNSW graph from the current vectors.
    pub fn rebuild_hnsw(&mut self) {
        self.hnsw.take();
        if self.vectors.is_empty() {
            return;
        }

        let hnsw_map = self.build_hnsw();
        let _ = self.hnsw.set(hnsw_map);

        tracing::debug!(chunks = self.vectors.len(), "HNSW graph built");
    }

    /// Perform a vector similarity search.
    pub fn search_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        score_threshold: f32,
    ) -> Vec<SearchHit> {
        if let Some(hnsw) = self.ensure_hnsw() {
            self.search_vector_hnsw(hnsw, query_embedding, top_k, score_threshold)
        } else {
            self.search_vector_brute_force(query_embedding, top_k, score_threshold)
        }
    }

    fn ensure_hnsw(&self) -> Option<&instant_distance::HnswMap<EmbeddingPoint, u64>> {
        if self.vectors.is_empty() {
            return None;
        }
        Some(self.hnsw.get_or_init(|| {
            if let Some(index_dir) = &self.index_dir {
                match self.load_hnsw_sidecar(index_dir) {
                    Ok(Some(hnsw)) => {
                        tracing::debug!("HNSW sidecar loaded");
                        return hnsw;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        tracing::warn!(%error, "Ignoring invalid HNSW sidecar");
                    }
                }
            }

            let hnsw = self.build_hnsw();
            if let Some(index_dir) = &self.index_dir {
                if let Err(error) = self.save_hnsw_sidecar(index_dir, &hnsw) {
                    tracing::warn!(%error, "Failed to persist rebuilt HNSW sidecar");
                }
            }
            hnsw
        }))
    }

    fn build_hnsw(&self) -> instant_distance::HnswMap<EmbeddingPoint, u64> {
        let entries = self.vectors.sorted_entries();
        let points = entries
            .iter()
            .map(|(_, embedding)| EmbeddingPoint(embedding.to_vec()))
            .collect();
        let values = entries.iter().map(|(chunk_id, _)| *chunk_id).collect();
        instant_distance::Builder::default().build(points, values)
    }

    fn search_vector_hnsw(
        &self,
        hnsw: &instant_distance::HnswMap<EmbeddingPoint, u64>,
        query_embedding: &[f32],
        top_k: usize,
        score_threshold: f32,
    ) -> Vec<SearchHit> {
        let query_point = EmbeddingPoint(query_embedding.to_vec());
        let mut search = instant_distance::Search::default();

        hnsw.search(&query_point, &mut search)
            .take(top_k)
            .filter_map(|item| {
                let chunk_id = *item.value;
                let score = 1.0 - item.distance;
                if score >= score_threshold {
                    Some(SearchHit { chunk_id, score })
                } else {
                    None
                }
            })
            .collect()
    }

    fn search_vector_brute_force(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        score_threshold: f32,
    ) -> Vec<SearchHit> {
        let mut scores: Vec<(u64, f32)> = self
            .vectors
            .sorted_entries()
            .into_iter()
            .map(|(chunk_id, embedding)| {
                let score = cosine_similarity(query_embedding, embedding);
                (chunk_id, score)
            })
            .filter(|(_, score)| *score >= score_threshold)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .take(top_k)
            .map(|(chunk_id, score)| SearchHit { chunk_id, score })
            .collect()
    }

    /// Perform a text search using the inverted index.
    pub fn search_text(&self, query: &str, top_k: usize) -> Vec<SearchHit> {
        let query_tokens = tokenize_for_index(query);

        if query_tokens.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<u64, f32> = HashMap::new();

        for token in &query_tokens {
            if let Some(posting_list) = self.inverted_index.get(token) {
                for &(chunk_id, frequency) in posting_list {
                    *scores.entry(chunk_id).or_default() += frequency as f32;
                }
            }
        }

        let num_tokens = query_tokens.len() as f32;
        let mut results: Vec<(u64, f32)> = scores
            .into_iter()
            .map(|(id, score)| (id, score / num_tokens))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
            .into_iter()
            .take(top_k)
            .map(|(chunk_id, score)| SearchHit { chunk_id, score })
            .collect()
    }

    /// Rank lexical matches with BM25 for hybrid retrieval.
    pub fn search_bm25(&self, query: &str, top_k: usize) -> Vec<SearchHit> {
        let query_tokens: HashSet<String> = tokenize_for_index(query).into_iter().collect();
        if query_tokens.is_empty() || self.chunks.is_empty() {
            return Vec::new();
        }
        let stats = self.bm25_stats.get_or_init(|| {
            let document_lengths: HashMap<u64, usize> = self
                .chunks
                .iter()
                .map(|(&chunk_id, chunk)| (chunk_id, tokenize_for_index(&chunk.body).len()))
                .collect();
            let average_document_length = document_lengths.values().sum::<usize>() as f32
                / document_lengths.len().max(1) as f32;
            Bm25Stats {
                document_lengths,
                average_document_length,
            }
        });
        let document_count = self.chunks.len() as f32;
        let average_length = stats.average_document_length.max(1.0);
        let mut scores: HashMap<u64, f32> = HashMap::new();

        for token in query_tokens {
            let Some(postings) = self.inverted_index.get(&token) else {
                continue;
            };
            let document_frequency = postings.len() as f32;
            let inverse_document_frequency = (1.0
                + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                .ln();
            for &(chunk_id, frequency) in postings {
                let term_frequency = frequency as f32;
                let document_length =
                    stats.document_lengths.get(&chunk_id).copied().unwrap_or(0) as f32;
                let denominator = term_frequency
                    + BM25_K1 * (1.0 - BM25_B + BM25_B * document_length / average_length);
                *scores.entry(chunk_id).or_default() +=
                    inverse_document_frequency * term_frequency * (BM25_K1 + 1.0) / denominator;
            }
        }

        let mut results: Vec<SearchHit> = scores
            .into_iter()
            .map(|(chunk_id, score)| SearchHit { chunk_id, score })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk_id.cmp(&b.chunk_id))
        });
        results.truncate(top_k);
        results
    }

    /// Return all name-based symbol definition candidates.
    pub fn symbol_definitions(&self, name: &str) -> Vec<CodeChunk> {
        let normalized = normalize_symbol_name(name);
        if normalized.is_empty() {
            return Vec::new();
        }
        let Some(chunk_ids) = self.ensure_symbol_index().get(&normalized) else {
            return Vec::new();
        };
        let mut definitions: Vec<CodeChunk> = chunk_ids
            .iter()
            .filter_map(|chunk_id| self.chunks.get(chunk_id).cloned())
            .collect();
        definitions.sort_by(|left, right| {
            left.file_path
                .cmp(&right.file_path)
                .then_with(|| left.line_range.start.cmp(&right.line_range.start))
                .then_with(|| left.id.cmp(&right.id))
        });
        definitions
    }

    /// List normalized symbols, optionally constrained by a name prefix.
    pub fn symbols(&self, prefix: Option<&str>, limit: usize) -> Vec<SymbolSummary> {
        let normalized_prefix = prefix.map(normalize_symbol_name).unwrap_or_default();
        let mut symbols = Vec::new();
        for (normalized_name, chunk_ids) in self.ensure_symbol_index() {
            if !normalized_name.starts_with(&normalized_prefix) {
                continue;
            }
            let mut names = BTreeSet::new();
            let mut kinds = BTreeSet::new();
            let mut languages = BTreeSet::new();
            for chunk_id in chunk_ids {
                if let Some(chunk) = self.chunks.get(chunk_id) {
                    if let Some(name) = &chunk.name {
                        names.insert(name.clone());
                    }
                    kinds.insert(chunk.kind.to_string());
                    languages.insert(chunk.language.clone());
                }
            }
            symbols.push(SymbolSummary {
                normalized_name: normalized_name.clone(),
                names: names.into_iter().collect(),
                kinds: kinds.into_iter().collect(),
                languages: languages.into_iter().collect(),
                definition_count: chunk_ids.len(),
            });
        }
        symbols.sort_by(|left, right| left.normalized_name.cmp(&right.normalized_name));
        symbols.truncate(limit);
        symbols
    }

    /// Mentions of `name` joined to matching definition chunks (name-level only).
    pub fn references(&self, name: &str) -> Vec<ReferenceHit> {
        let normalized = normalize_symbol_name(name);
        if normalized.is_empty() {
            return Vec::new();
        }
        let definitions = self.symbol_definitions(name);
        if definitions.is_empty() {
            return Vec::new();
        }
        let definition_count = definitions.len();
        let Some(mention_ids) = self.ensure_mention_index().get(&normalized) else {
            return Vec::new();
        };

        let mut hits = Vec::new();
        for mention_id in mention_ids {
            let Some(mention) = self.call_sites.get(mention_id) else {
                continue;
            };
            for definition in &definitions {
                let (confidence, reasons) = score_reference(definition, mention, definition_count);
                hits.push(ReferenceHit {
                    definition: definition.clone(),
                    mention: mention.clone(),
                    confidence,
                    reasons,
                    disclaimer: REFERENCES_DISCLAIMER,
                });
            }
        }
        hits.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.mention.file_path.cmp(&right.mention.file_path))
                .then_with(|| {
                    left.mention
                        .line_range
                        .start
                        .cmp(&right.mention.line_range.start)
                })
                .then_with(|| left.mention.id.cmp(&right.mention.id))
        });
        hits
    }

    /// Call sites whose callee matches `name` (callers of that symbol).
    pub fn callers(&self, name: &str) -> Vec<CallerHit> {
        let normalized = normalize_symbol_name(name);
        if normalized.is_empty() {
            return Vec::new();
        }
        let definitions = self.symbol_definitions(name);
        let definition_count = definitions.len().max(1);
        let Some(mention_ids) = self.ensure_mention_index().get(&normalized) else {
            return Vec::new();
        };

        let mut hits = Vec::new();
        for mention_id in mention_ids {
            let Some(mention) = self.call_sites.get(mention_id) else {
                continue;
            };
            let caller_chunk = mention
                .enclosing_chunk_id
                .and_then(|chunk_id| self.chunks.get(&chunk_id).cloned());
            let matched_definitions: Vec<CodeChunk> = definitions
                .iter()
                .filter(|definition| definition.language == mention.language)
                .cloned()
                .collect();
            let sample_definition = matched_definitions.first().or_else(|| definitions.first());
            let (confidence, reasons) = if let Some(definition) = sample_definition {
                score_reference(definition, mention, definition_count)
            } else {
                (
                    0.2,
                    vec![
                        "normalized_name_match".to_string(),
                        "definition_not_indexed".to_string(),
                    ],
                )
            };
            hits.push(CallerHit {
                caller_chunk,
                mention: mention.clone(),
                matched_definitions,
                confidence,
                reasons,
                disclaimer: REFERENCES_DISCLAIMER,
            });
        }
        hits.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.mention.file_path.cmp(&right.mention.file_path))
                .then_with(|| {
                    left.mention
                        .line_range
                        .start
                        .cmp(&right.mention.line_range.start)
                })
                .then_with(|| left.mention.id.cmp(&right.mention.id))
        });
        hits
    }

    fn ensure_symbol_index(&self) -> &HashMap<String, Vec<u64>> {
        self.symbol_index.get_or_init(|| {
            let mut symbol_index: HashMap<String, Vec<u64>> = HashMap::new();
            for (&chunk_id, chunk) in &self.chunks {
                if !is_indexable_symbol(chunk) {
                    continue;
                }
                let normalized = normalize_symbol_name(
                    chunk.name.as_deref().expect("indexable symbols have names"),
                );
                symbol_index.entry(normalized).or_default().push(chunk_id);
            }
            for chunk_ids in symbol_index.values_mut() {
                chunk_ids.sort_unstable();
            }
            symbol_index
        })
    }

    fn ensure_mention_index(&self) -> &HashMap<String, Vec<u64>> {
        self.mention_index.get_or_init(|| {
            let mut mention_index: HashMap<String, Vec<u64>> = HashMap::new();
            for (id, site) in &self.call_sites {
                let normalized = normalize_symbol_name(&site.callee_name);
                if normalized.is_empty() {
                    continue;
                }
                mention_index.entry(normalized).or_default().push(*id);
            }
            for ids in mention_index.values_mut() {
                ids.sort_unstable();
            }
            mention_index
        })
    }

    /// Get a chunk by its ID.
    pub fn get_chunk(&self, chunk_id: u64) -> Option<&CodeChunk> {
        self.chunks.get(&chunk_id)
    }

    /// Save the index to a directory using the v4 framed format.
    pub fn save(&self, dir: &Path) -> Result<(), IndexError> {
        std::fs::create_dir_all(dir)?;

        let payload = IndexPayloadV4 {
            embedding_dim: self.embedding_dim,
            chunk_count: self.chunk_count,
            vector_data: self.vectors.data.clone(),
            vector_row_ids: self.vectors.row_ids.clone(),
            inverted_index: self.inverted_index.clone(),
            chunks: self.chunks.clone(),
        };
        let payload_bytes =
            bincode::serialize(&payload).map_err(|e| IndexError::Serialization(e.to_string()))?;
        let mut framed = Vec::with_capacity(20 + payload_bytes.len());
        framed.extend_from_slice(INDEX_MAGIC);
        framed.extend_from_slice(&INDEX_VERSION.to_le_bytes());
        framed.extend_from_slice(&(payload_bytes.len() as u64).to_le_bytes());
        framed.extend_from_slice(&payload_bytes);
        crate::index::atomic::atomic_write(&dir.join(INDEX_FILENAME), &framed)?;
        if let Some(hnsw) = self.hnsw.get() {
            self.save_hnsw_sidecar(dir, hnsw)?;
        }
        self.save_mentions_sidecar(dir)?;

        let old_json_path = dir.join("index.json");
        if old_json_path.exists() {
            let _ = std::fs::remove_file(&old_json_path);
        }

        tracing::info!(
            chunks = self.chunk_count,
            call_sites = self.call_sites.len(),
            path = %dir.display(),
            "Index saved (framed v4)"
        );

        Ok(())
    }

    fn save_mentions_sidecar(&self, dir: &Path) -> Result<(), IndexError> {
        let sidecar = MentionsSidecar {
            version: MENTIONS_SIDECAR_VERSION,
            chunk_fingerprint: self.chunk_fingerprint(),
            call_sites: self.call_sites.clone(),
            file_to_call_sites: self.file_to_call_sites.clone(),
        };
        let payload =
            bincode::serialize(&sidecar).map_err(|e| IndexError::Serialization(e.to_string()))?;
        let mut framed = Vec::with_capacity(20 + payload.len());
        framed.extend_from_slice(MENTIONS_MAGIC);
        framed.extend_from_slice(&MENTIONS_SIDECAR_VERSION.to_le_bytes());
        framed.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        framed.extend_from_slice(&payload);
        crate::index::atomic::atomic_write(&dir.join(MENTIONS_FILENAME), &framed)
    }

    fn load_mentions_sidecar(&mut self, dir: &Path) -> Result<(), IndexError> {
        let path = dir.join(MENTIONS_FILENAME);
        if !path.exists() {
            return Ok(());
        }
        let data = std::fs::read(&path)?;
        if data.len() < 20 || &data[..8] != MENTIONS_MAGIC {
            tracing::warn!(path = %path.display(), "Ignoring corrupt mentions sidecar");
            return Ok(());
        }
        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        if version != MENTIONS_SIDECAR_VERSION {
            tracing::warn!(
                path = %path.display(),
                version,
                "Ignoring mentions sidecar with unsupported version"
            );
            return Ok(());
        }
        let payload_len = u64::from_le_bytes(data[12..20].try_into().unwrap()) as usize;
        if data.len() != 20 + payload_len {
            tracing::warn!(path = %path.display(), "Ignoring truncated mentions sidecar");
            return Ok(());
        }
        let sidecar: MentionsSidecar = bincode::deserialize(&data[20..])
            .map_err(|e| IndexError::Serialization(e.to_string()))?;
        if sidecar.chunk_fingerprint != self.chunk_fingerprint() {
            tracing::warn!(
                path = %path.display(),
                "Mentions sidecar fingerprint mismatch; dropping stale edges"
            );
            return Ok(());
        }
        self.call_sites = sidecar.call_sites;
        self.file_to_call_sites = sidecar.file_to_call_sites;
        self.mention_index.take();
        Ok(())
    }

    fn chunk_fingerprint(&self) -> [u8; 32] {
        let mut ids: Vec<u64> = self.chunks.keys().copied().collect();
        ids.sort_unstable();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(ids.len() as u64).to_le_bytes());
        for id in ids {
            hasher.update(&id.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }

    fn save_hnsw_sidecar(
        &self,
        dir: &Path,
        hnsw: &instant_distance::HnswMap<EmbeddingPoint, u64>,
    ) -> Result<(), IndexError> {
        let sidecar = HnswSidecarRef {
            version: HNSW_SIDECAR_VERSION,
            vector_fingerprint: self.vector_fingerprint(),
            hnsw,
        };
        let data =
            bincode::serialize(&sidecar).map_err(|e| IndexError::Serialization(e.to_string()))?;
        crate::index::atomic::atomic_write(&dir.join(HNSW_FILENAME), &data)
    }

    fn load_hnsw_sidecar(
        &self,
        dir: &Path,
    ) -> Result<Option<instant_distance::HnswMap<EmbeddingPoint, u64>>, IndexError> {
        let path = dir.join(HNSW_FILENAME);
        if !path.exists() {
            return Ok(None);
        }
        let data = std::fs::read(path)?;
        let sidecar: HnswSidecar =
            bincode::deserialize(&data).map_err(|e| IndexError::Serialization(e.to_string()))?;
        if sidecar.version != HNSW_SIDECAR_VERSION
            || sidecar.vector_fingerprint != self.vector_fingerprint()
        {
            return Ok(None);
        }
        Ok(Some(sidecar.hnsw))
    }

    fn vector_fingerprint(&self) -> [u8; 32] {
        let entries = self.vectors.sorted_entries();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(self.embedding_dim as u64).to_le_bytes());
        hasher.update(&(entries.len() as u64).to_le_bytes());
        for (chunk_id, embedding) in entries {
            hasher.update(&chunk_id.to_le_bytes());
            hasher.update(&(embedding.len() as u64).to_le_bytes());
            for value in embedding {
                hasher.update(&value.to_bits().to_le_bytes());
            }
        }
        *hasher.finalize().as_bytes()
    }

    /// Load an index from a directory.
    ///
    /// Reads the framed v4 header first. Legacy v3 bincode indexes are
    /// rejected with a rebuild prompt and are never auto-migrated.
    pub fn load(dir: &Path) -> Result<Self, IndexError> {
        let bin_path = dir.join(INDEX_FILENAME);
        if !bin_path.exists() {
            if dir.join("index.json").exists() {
                return Err(IndexError::VersionMismatch {
                    file_version: 1,
                    expected_version: INDEX_VERSION,
                });
            }
            return Err(IndexError::NotFound(bin_path));
        }

        let data = std::fs::read(&bin_path)?;
        let mut index = if data.len() >= 20 && data[..8] == *INDEX_MAGIC {
            let file_version = u32::from_le_bytes(data[8..12].try_into().unwrap());
            if file_version != INDEX_VERSION {
                return Err(IndexError::VersionMismatch {
                    file_version,
                    expected_version: INDEX_VERSION,
                });
            }
            let payload_len = u64::from_le_bytes(data[12..20].try_into().unwrap()) as usize;
            if data.len() != 20 + payload_len {
                return Err(IndexError::Corrupted(format!(
                    "index payload length mismatch: header={payload_len}, actual={}",
                    data.len().saturating_sub(20)
                )));
            }
            let payload: IndexPayloadV4 = bincode::deserialize(&data[20..])
                .map_err(|e| IndexError::Serialization(e.to_string()))?;
            Self::from_payload(payload)?
        } else if data.len() >= 4 {
            let file_version = u32::from_le_bytes(data[..4].try_into().unwrap());
            return Err(IndexError::VersionMismatch {
                file_version,
                expected_version: INDEX_VERSION,
            });
        } else {
            return Err(IndexError::Corrupted(
                "index.bin is too small to contain a valid header".to_string(),
            ));
        };

        index.index_dir = Some(dir.to_path_buf());
        if let Err(error) = index.load_mentions_sidecar(dir) {
            tracing::warn!(%error, "Failed to load mentions sidecar");
        }

        tracing::info!(
            chunks = index.chunk_count,
            call_sites = index.call_sites.len(),
            path = %dir.display(),
            "Index loaded (HNSW deferred)"
        );

        Ok(index)
    }

    fn from_payload(payload: IndexPayloadV4) -> Result<Self, IndexError> {
        if payload.chunk_count != payload.chunks.len() {
            return Err(IndexError::Corrupted(format!(
                "chunk_count {} does not match chunks map size {}",
                payload.chunk_count,
                payload.chunks.len()
            )));
        }
        let vectors = VectorStore::from_parts(
            payload.embedding_dim,
            payload.vector_data,
            payload.vector_row_ids,
        )?;
        Ok(Self {
            version: INDEX_VERSION,
            vectors,
            inverted_index: payload.inverted_index,
            chunks: payload.chunks,
            call_sites: HashMap::new(),
            file_to_call_sites: HashMap::new(),
            embedding_dim: payload.embedding_dim,
            chunk_count: payload.chunk_count,
            hnsw: OnceLock::new(),
            index_dir: None,
            bm25_stats: OnceLock::new(),
            symbol_index: OnceLock::new(),
            mention_index: OnceLock::new(),
        })
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Simple tokenization for the inverted text index.
fn tokenize_for_index(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for identifier in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if identifier.is_empty() {
            continue;
        }
        let normalized = identifier.to_lowercase();
        if normalized.len() >= 2 {
            tokens.push(normalized.clone());
        }
        for component in split_identifier(identifier) {
            let component = component.to_lowercase();
            if component.len() >= 2 && component != normalized {
                tokens.push(component);
            }
        }
    }
    tokens
}

fn split_identifier(identifier: &str) -> Vec<&str> {
    let mut components = Vec::new();
    for word in identifier.split('_').filter(|word| !word.is_empty()) {
        let mut start = 0;
        let mut previous_lowercase = false;
        for (offset, character) in word.char_indices() {
            if offset > 0 && character.is_uppercase() && previous_lowercase {
                components.push(&word[start..offset]);
                start = offset;
            }
            previous_lowercase = character.is_lowercase();
        }
        components.push(&word[start..]);
    }
    components
}

/// Public wrapper for `tokenize_for_index` — used by incremental indexing.
pub fn tokenize_for_index_pub(text: &str) -> Vec<String> {
    tokenize_for_index(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ChunkKind;
    use std::path::PathBuf;

    fn make_test_chunk(id: u64, name: &str, body: &str) -> CodeChunk {
        CodeChunk {
            id,
            file_path: PathBuf::from("test.rs"),
            language: "rust".to_string(),
            kind: ChunkKind::Function,
            name: Some(name.to_string()),
            signature: None,
            doc_comment: None,
            body: body.to_string(),
            byte_range: 0..body.len(),
            line_range: 0..1,
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b)).abs() < 0.01);

        let c = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_build_and_search_text() {
        let chunks = vec![
            make_test_chunk(
                1,
                "authenticate",
                "fn authenticate(user: &str, password: &str) -> Result<Token, Error>",
            ),
            make_test_chunk(2, "calculate", "fn calculate_total(items: &[Item]) -> f64"),
        ];
        let embeddings = vec![vec![0.1; 8], vec![0.2; 8]];

        let index = SeekrIndex::build_from(&chunks, &embeddings, 8);

        assert_eq!(index.chunk_count(), 2);

        let results = index.search_text("authenticate user password", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, 1);
    }

    #[test]
    fn test_build_and_search_vector() {
        let chunks = vec![
            make_test_chunk(1, "foo", "fn foo()"),
            make_test_chunk(2, "bar", "fn bar()"),
        ];
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let index = SeekrIndex::build_from(&chunks, &embeddings, 3);

        let query = vec![0.9, 0.1, 0.0];
        let results = index.search_vector(&query, 2, 0.0);
        assert!(!results.is_empty());
        assert_eq!(
            results[0].chunk_id, 1,
            "Should find the most similar chunk first"
        );
    }

    #[test]
    fn test_save_and_load() {
        let chunks = vec![make_test_chunk(1, "test", "fn test() {}")];
        let embeddings = vec![vec![0.5; 4]];
        let index = SeekrIndex::build_from(&chunks, &embeddings, 4);

        let dir = tempfile::tempdir().unwrap();
        index.save(dir.path()).unwrap();
        index.save(dir.path()).unwrap();
        assert!(dir.path().join(HNSW_FILENAME).exists());

        let data = std::fs::read(dir.path().join(INDEX_FILENAME)).unwrap();
        assert_eq!(&data[..8], INDEX_MAGIC);

        let loaded = SeekrIndex::load(dir.path()).unwrap();
        assert_eq!(loaded.chunk_count(), 1);
        assert_eq!(loaded.format_version(), INDEX_VERSION);
        assert!(loaded.hnsw.get().is_none());
        assert!(!loaded.search_text("test", 10).is_empty());
        assert!(
            loaded.hnsw.get().is_none(),
            "text search must not load HNSW"
        );
        assert!(!loaded.search_vector(&[0.5; 4], 1, 0.0).is_empty());
        assert!(loaded.hnsw.get().is_some());
        assert_eq!(loaded.embedding(1), Some(&[0.5; 4][..]));
        assert!(std::fs::read_dir(dir.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp-")
        }));
    }

    #[test]
    fn rejects_legacy_v3_index_without_deserializing_payload() {
        let dir = tempfile::tempdir().unwrap();
        // Minimal legacy-looking payload: version u32 = 3 at the front.
        std::fs::write(dir.path().join(INDEX_FILENAME), 3u32.to_le_bytes()).unwrap();

        let error = SeekrIndex::load(dir.path()).unwrap_err();
        assert!(matches!(
            error,
            IndexError::VersionMismatch {
                file_version: 3,
                expected_version: INDEX_VERSION
            }
        ));
        let message = error.to_string();
        assert!(
            message.contains("index --force")
                || message.contains("VersionMismatch")
                || message.contains("version")
        );
    }

    #[test]
    fn vector_store_swap_remove_preserves_remaining_embeddings() {
        let mut store = VectorStore::new(2);
        store.insert(1, &[1.0, 0.0]).unwrap();
        store.insert(2, &[0.0, 1.0]).unwrap();
        store.insert(3, &[0.5, 0.5]).unwrap();
        store.remove(1);
        assert_eq!(store.get(2), Some(&[0.0, 1.0][..]));
        assert_eq!(store.get(3), Some(&[0.5, 0.5][..]));
        assert!(store.get(1).is_none());
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_tokenize_for_index() {
        let tokens =
            tokenize_for_index("fn authenticate_user(username: &str) -> HttpResponse<String>");
        assert!(tokens.contains(&"fn".to_string()));
        assert!(tokens.contains(&"authenticate_user".to_string()));
        assert!(tokens.contains(&"authenticate".to_string()));
        assert!(tokens.contains(&"user".to_string()));
        assert!(tokens.contains(&"username".to_string()));
        assert!(tokens.contains(&"http".to_string()));
        assert!(tokens.contains(&"response".to_string()));
        assert!(tokens.contains(&"string".to_string()));
    }

    #[test]
    fn test_add_entry_rejects_chunk_id_collision_without_overwrite() {
        let mut index = SeekrIndex::new(2);
        let first = make_test_chunk(7, "first", "fn first() {}");
        index.add_entry(
            IndexEntry {
                chunk_id: 7,
                embedding: vec![1.0, 0.0],
                text_tokens: vec!["first".to_string()],
            },
            first.clone(),
        );

        let mut collision = make_test_chunk(7, "second", "fn second() {}");
        collision.file_path = PathBuf::from("other.rs");
        let result = index.try_add_entry(
            IndexEntry {
                chunk_id: 7,
                embedding: vec![0.0, 1.0],
                text_tokens: vec!["second".to_string()],
            },
            collision,
        );

        assert!(matches!(result, Err(IndexError::ChunkIdCollision { .. })));
        assert_eq!(index.get_chunk(7).unwrap().name.as_deref(), Some("first"));
        assert_eq!(index.embedding(7), Some(&[1.0, 0.0][..]));
    }

    #[test]
    fn test_bm25_prefers_shorter_document_with_same_term_frequency() {
        let chunks = vec![
            make_test_chunk(1, "short", "target helper"),
            make_test_chunk(
                2,
                "long",
                "target helper filler filler filler filler filler filler",
            ),
        ];
        let index = SeekrIndex::build_from(&chunks, &[vec![0.0; 2], vec![0.0; 2]], 2);

        let results = index.search_bm25("target", 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_id, 1);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_bm25_stats_are_invalidated_after_mutation() {
        let mut index = SeekrIndex::build_from(
            &[make_test_chunk(1, "first", "existing token")],
            &[vec![0.0; 2]],
            2,
        );
        assert!(!index.search_bm25("existing", 10).is_empty());

        let chunk = make_test_chunk(2, "second", "newly added");
        index.add_entry(
            IndexEntry {
                chunk_id: 2,
                embedding: vec![0.0; 2],
                text_tokens: tokenize_for_index(&chunk.body),
            },
            chunk,
        );
        assert_eq!(index.search_bm25("newly", 10)[0].chunk_id, 2);
    }

    #[test]
    fn symbol_lookup_returns_all_candidates_in_stable_order() {
        let mut second = make_test_chunk(2, "Shared", "fn Shared() {}");
        second.file_path = PathBuf::from("src/b.rs");
        let mut first = make_test_chunk(1, "shared", "fn shared() {}");
        first.file_path = PathBuf::from("src/a.rs");
        let mut index = SeekrIndex::build_from(&[second, first], &[vec![0.0; 2], vec![0.0; 2]], 2);

        let definitions = index.symbol_definitions(" SHARED ");
        assert_eq!(
            definitions
                .iter()
                .map(|chunk| chunk.file_path.clone())
                .collect::<Vec<_>>(),
            vec![PathBuf::from("src/a.rs"), PathBuf::from("src/b.rs")]
        );

        index.remove_chunk(1);
        assert_eq!(index.symbol_definitions("shared").len(), 1);
        assert_eq!(index.symbols(Some("sha"), 10)[0].definition_count, 1);
    }

    #[test]
    fn fallback_blocks_are_not_symbols() {
        let mut chunk = make_test_chunk(1, "file.rs:L1-L20", "unparsed content");
        chunk.kind = ChunkKind::Block;
        let index = SeekrIndex::build_from(&[chunk], &[vec![0.0; 2]], 2);

        assert!(index.symbol_definitions("file.rs:L1-L20").is_empty());
    }
}
