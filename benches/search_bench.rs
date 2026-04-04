//! Performance benchmarks for Seekr search engine.
//!
//! Run with: `cargo bench --bench search_bench`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::path::PathBuf;

use seekr_code::index::store::{cosine_similarity, SeekrIndex};
use seekr_code::parser::{CodeChunk, ChunkKind};

/// Create a test chunk with a given ID and body.
fn make_chunk(id: u64, body: &str) -> CodeChunk {
    CodeChunk {
        id,
        file_path: PathBuf::from(format!("bench/file_{}.rs", id)),
        language: "rust".to_string(),
        kind: ChunkKind::Function,
        name: Some(format!("func_{}", id)),
        signature: None,
        doc_comment: None,
        body: body.to_string(),
        byte_range: 0..body.len(),
        line_range: 0..1,
    }
}

/// Generate a pseudo-random f32 vector of a given dimension.
fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = seed;
    for _ in 0..dim {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
        v.push(val);
    }
    // L2 normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Benchmark index build from chunks.
fn bench_index_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_build");

    for &n in &[100, 500, 1000] {
        let chunks: Vec<CodeChunk> = (0..n)
            .map(|i| make_chunk(i, &format!("fn func_{}() {{ let x = {}; }}", i, i * 2)))
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..n).map(|i| random_vec(384, i)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                SeekrIndex::build_from(&chunks, &embeddings, 384)
            });
        });
    }

    group.finish();
}

/// Benchmark vector search (brute-force KNN).
fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");

    for &n in &[500, 1000, 5000] {
        let chunks: Vec<CodeChunk> = (0..n)
            .map(|i| make_chunk(i, &format!("fn search_target_{}() {{}}", i)))
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..n).map(|i| random_vec(384, i)).collect();
        let index = SeekrIndex::build_from(&chunks, &embeddings, 384);

        let query = random_vec(384, 99999);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                index.search_vector(&query, 10, 0.0)
            });
        });
    }

    group.finish();
}

/// Benchmark text search (inverted index).
fn bench_text_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_search");

    for &n in &[500, 1000, 5000] {
        let chunks: Vec<CodeChunk> = (0..n)
            .map(|i| make_chunk(i, &format!("fn authenticate_user_{i}(username: &str, password: &str) -> Result<Token, Error> {{ validate(username); hash(password); }}")))
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..n).map(|i| random_vec(384, i)).collect();
        let index = SeekrIndex::build_from(&chunks, &embeddings, 384);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                index.search_text("authenticate user password", 10)
            });
        });
    }

    group.finish();
}

/// Benchmark cosine similarity computation.
fn bench_cosine_similarity(c: &mut Criterion) {
    let a = random_vec(384, 42);
    let b = random_vec(384, 84);

    c.bench_function("cosine_similarity_384d", |bench| {
        bench.iter(|| cosine_similarity(&a, &b));
    });
}

/// Benchmark index save/load (bincode).
fn bench_save_load(c: &mut Criterion) {
    let n = 1000;
    let chunks: Vec<CodeChunk> = (0..n)
        .map(|i| make_chunk(i, &format!("fn bench_file_{i}() {{ let data = vec![{i}]; }}")))
        .collect();
    let embeddings: Vec<Vec<f32>> = (0..n).map(|i| random_vec(384, i)).collect();
    let index = SeekrIndex::build_from(&chunks, &embeddings, 384);

    let dir = tempfile::tempdir().unwrap();

    c.bench_function("index_save_1k", |b| {
        b.iter(|| {
            index.save(dir.path()).unwrap();
        });
    });

    // Save once so we can benchmark load
    index.save(dir.path()).unwrap();

    c.bench_function("index_load_1k", |b| {
        b.iter(|| {
            SeekrIndex::load(dir.path()).unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_index_build,
    bench_vector_search,
    bench_text_search,
    bench_cosine_similarity,
    bench_save_load,
);
criterion_main!(benches);
