//! Versioned retrieval quality corpus and gates.
//!
//! - Offline lexical (BM25) judgments always run.
//! - Hybrid Recall@5 / MRR against a real `OnnxEmbedder` runs when
//!   `SEEKR_QUALITY_GATE=1` (CI quality job) or when explicitly requested.
//!
//! Corpus version: 1

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use seekr_code::config::{SearchConfig, SeekrConfig};
use seekr_code::embedder::onnx::OnnxEmbedder;
use seekr_code::embedder::traits::Embedder;
use seekr_code::error::EmbedderError;
use seekr_code::index::builder::IndexBuilder;
use seekr_code::index::store::SeekrIndex;
use seekr_code::parser::{ChunkKind, CodeChunk};
use seekr_code::search::SearchMode;
use seekr_code::search::engine::{SearchEngine, SearchOptions};

const CORPUS_VERSION: u32 = 1;

/// First-round Hybrid baseline measured 2026-09-10 (macOS arm64, MiniLM):
/// Recall@5 = 1.000, MRR = 1.000. Floors keep a small regression margin.
const HYBRID_RECALL_AT_5_BASELINE: f32 = 1.0;
const HYBRID_MRR_BASELINE: f32 = 1.0;
const BASELINE_MARGIN: f32 = 0.05;

struct Judgment {
    query: &'static str,
    /// Stable keys: `language|relative/path|name`
    relevant_keys: &'static [&'static str],
}

fn relative_path(chunk: &CodeChunk) -> String {
    let path = chunk.file_path.to_string_lossy().replace('\\', "/");
    // Strip temp project prefix by keeping the corpus-relative tail we authored.
    for marker in ["corpus/", "project/"] {
        if let Some(idx) = path.rfind(marker) {
            return path[idx + marker.len()..].to_string();
        }
    }
    path
}

fn chunk_key(chunk: &CodeChunk) -> String {
    format!(
        "{}|{}|{}",
        chunk.language,
        relative_path(chunk),
        chunk.name.as_deref().unwrap_or("")
    )
}

fn bm25_chunk(id: u64, language: &str, path: &str, name: &str, body: &str) -> CodeChunk {
    CodeChunk {
        id,
        file_path: PathBuf::from(path),
        language: language.to_string(),
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
fn lexical_multilingual_bm25_quality_corpus_v1() {
    assert_eq!(CORPUS_VERSION, 1);
    let chunks = vec![
        bm25_chunk(
            1,
            "rust",
            "src/auth.rs",
            "verify_password",
            "verify password hash and reject invalid credentials",
        ),
        bm25_chunk(
            2,
            "rust",
            "src/cache.rs",
            "invalidate_cache",
            "invalidate stale cache entries after configuration changes",
        ),
        bm25_chunk(
            3,
            "python",
            "auth/service.py",
            "authenticate",
            "authenticate user credentials and issue session token",
        ),
        bm25_chunk(
            4,
            "typescript",
            "src/billing.ts",
            "formatCurrency",
            "format currency amount using Intl NumberFormat",
        ),
        bm25_chunk(
            5,
            "python",
            "pricing.py",
            "calculate_discount",
            "calculate discounted price from percent off",
        ),
        bm25_chunk(
            6,
            "rust",
            "src/db.rs",
            "rollback_transaction",
            "rollback database transaction when persistence fails",
        ),
    ];
    let embeddings = vec![vec![0.0; 4]; chunks.len()];
    let index = SeekrIndex::build_from(&chunks, &embeddings, 4);
    let judgments = [
        ("verify password hash", "rust|src/auth.rs|verify_password"),
        (
            "invalidate cache entries",
            "rust|src/cache.rs|invalidate_cache",
        ),
        (
            "authenticate user credentials",
            "python|auth/service.py|authenticate",
        ),
        (
            "format currency amount",
            "typescript|src/billing.ts|formatCurrency",
        ),
        (
            "calculate discounted price",
            "python|pricing.py|calculate_discount",
        ),
        (
            "database transaction rollback",
            "rust|src/db.rs|rollback_transaction",
        ),
    ];

    let mut recall_hits = 0usize;
    let reciprocal_rank_sum: f32 = judgments
        .iter()
        .map(|(query, relevant_key)| {
            let hits = index.search_bm25(query, 5);
            let rank = hits.iter().position(|hit| {
                index
                    .get_chunk(hit.chunk_id)
                    .is_some_and(|chunk| chunk_key(chunk) == *relevant_key)
            });
            if rank.is_some() {
                recall_hits += 1;
            }
            rank.map(|rank| 1.0 / (rank + 1) as f32).unwrap_or(0.0)
        })
        .sum();
    let recall_at_5 = recall_hits as f32 / judgments.len() as f32;
    let mrr = reciprocal_rank_sum / judgments.len() as f32;
    assert!(
        recall_at_5 >= 0.95,
        "lexical Recall@5={recall_at_5:.3}, expected >= 0.95"
    );
    assert!(mrr >= 0.95, "lexical MRR={mrr:.3}, expected >= 0.95");
}

fn quality_corpus_sources() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "src/auth.rs",
            r#"
/// Validate credentials and return a session token.
pub fn authenticate_user(username: &str, password: &str) -> Result<String, String> {
    let user = find_user(username)?;
    if verify_password(password, &user.hash) {
        Ok(issue_token(&user))
    } else {
        Err("invalid".into())
    }
}

fn find_user(_name: &str) -> Result<User, String> {
    Err("missing".into())
}

fn verify_password(_password: &str, _hash: &str) -> bool {
    true
}

fn issue_token(_user: &User) -> String {
    "tok".into()
}

struct User {
    hash: String,
}
"#,
        ),
        (
            "src/cache.rs",
            r#"
/// Drop stale entries from the in-memory cache.
pub fn invalidate_cache(keys: &[String]) {
    for key in keys {
        let _ = key;
    }
}

/// Rebuild the cache after configuration reloads.
pub fn warm_cache() {
    invalidate_cache(&[]);
}
"#,
        ),
        (
            "billing.py",
            r#"
def calculate_discount(price: float, discount_percent: float) -> float:
    """Apply a percentage discount to a price."""
    return price * (1 - discount_percent / 100)


def format_invoice_total(items: list[float]) -> float:
    """Sum invoice line items."""
    return sum(items)
"#,
        ),
        (
            "src/users.ts",
            r#"
export async function fetchUserProfile(userId: number): Promise<object> {
  const response = await fetch(`/api/users/${userId}`);
  return response.json();
}

export function formatCurrency(amount: number, currency: string): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency }).format(amount);
}
"#,
        ),
    ]
}

fn hybrid_judgments() -> Vec<Judgment> {
    vec![
        Judgment {
            query: "authenticate user with password and return session token",
            relevant_keys: &["rust|src/auth.rs|authenticate_user"],
        },
        Judgment {
            query: "invalidate stale cache entries",
            relevant_keys: &["rust|src/cache.rs|invalidate_cache"],
        },
        Judgment {
            query: "calculate discounted price from percent",
            relevant_keys: &["python|billing.py|calculate_discount"],
        },
        Judgment {
            query: "format money amount as currency string",
            relevant_keys: &["typescript|src/users.ts|formatCurrency"],
        },
        Judgment {
            query: "fetch user profile from HTTP API",
            relevant_keys: &["typescript|src/users.ts|fetchUserProfile"],
        },
    ]
}

fn evaluate_hybrid(index: &SeekrIndex, embedder: Arc<dyn Embedder>) -> (f32, f32) {
    let engine = SearchEngine::new(SearchConfig::default(), Some(embedder));
    let judgments = hybrid_judgments();
    let mut recall_hits = 0usize;
    let mut mrr_sum = 0.0f32;

    for judgment in &judgments {
        let results = engine
            .search(
                index,
                judgment.query,
                SearchMode::Hybrid,
                &SearchOptions {
                    top_k: 5,
                    ..SearchOptions::default()
                },
            )
            .expect("hybrid search");
        let relevant: HashSet<&str> = judgment.relevant_keys.iter().copied().collect();
        let rank = results
            .iter()
            .position(|result| relevant.contains(chunk_key(&result.chunk).as_str()));
        if rank.is_some() {
            recall_hits += 1;
        }
        mrr_sum += rank.map(|rank| 1.0 / (rank + 1) as f32).unwrap_or(0.0);
    }

    let n = judgments.len() as f32;
    (recall_hits as f32 / n, mrr_sum / n)
}

fn build_quality_index(embedder: Arc<dyn Embedder>) -> (tempfile::TempDir, SeekrIndex) {
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path().join("corpus");
    std::fs::create_dir_all(&project).unwrap();
    for (relative, source) in quality_corpus_sources() {
        let path = project.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, source).unwrap();
    }
    let config = SeekrConfig {
        index_dir: dir.path().join("indexes"),
        model_dir: dir.path().join("unused-models"),
        ..SeekrConfig::default()
    };
    let report = IndexBuilder::new(config, embedder)
        .build(&project, true)
        .expect("index quality corpus");
    let index = SeekrIndex::load(&report.index_dir).expect("load quality index");
    (dir, index)
}

/// Offline Hybrid smoke using a deterministic dummy embedder.
/// This does **not** gate release quality — semantic ranking is meaningless here.
#[test]
fn hybrid_dummy_smoke_returns_results() {
    struct DummyEmbedder {
        dim: usize,
    }
    impl Embedder for DummyEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            let mut embedding = vec![0.0; self.dim];
            for (index, byte) in text.bytes().enumerate() {
                embedding[index % self.dim] += byte as f32;
            }
            let norm = embedding
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .max(1e-6);
            for value in &mut embedding {
                *value /= norm;
            }
            Ok(embedding)
        }
        fn dimension(&self) -> usize {
            self.dim
        }
    }

    let embedder: Arc<dyn Embedder> = Arc::new(DummyEmbedder { dim: 32 });
    let (_guard, index) = build_quality_index(Arc::clone(&embedder));
    let engine = SearchEngine::new(SearchConfig::default(), Some(embedder));
    let results = engine
        .search(
            &index,
            "authenticate user",
            SearchMode::Hybrid,
            &SearchOptions {
                top_k: 5,
                ..SearchOptions::default()
            },
        )
        .unwrap();
    assert!(!results.is_empty());
}

fn quality_gate_enabled() -> bool {
    matches!(
        std::env::var("SEEKR_QUALITY_GATE").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

fn resolve_model_dir() -> PathBuf {
    if let Ok(path) = std::env::var("SEEKR_MODEL_DIR") {
        return PathBuf::from(path);
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".seekr").join("models")
}

#[test]
fn onnx_hybrid_quality_gate_v1() {
    if !quality_gate_enabled() {
        eprintln!(
            "skipping onnx hybrid quality gate (set SEEKR_QUALITY_GATE=1 to enable); corpus_v{CORPUS_VERSION}"
        );
        return;
    }

    let model_dir = resolve_model_dir();
    let embedder: Arc<dyn Embedder> =
        Arc::new(OnnxEmbedder::new(&model_dir).unwrap_or_else(|error| {
            panic!(
                "failed to load OnnxEmbedder from {}: {error}",
                model_dir.display()
            )
        }));
    let (_guard, index) = build_quality_index(Arc::clone(&embedder));
    let (recall_at_5, mrr) = evaluate_hybrid(&index, embedder);

    let recall_floor = (HYBRID_RECALL_AT_5_BASELINE - BASELINE_MARGIN).max(0.0);
    let mrr_floor = (HYBRID_MRR_BASELINE - BASELINE_MARGIN).max(0.0);
    eprintln!(
        "onnx hybrid quality corpus_v{CORPUS_VERSION}: Recall@5={recall_at_5:.3} (floor {recall_floor:.3}), MRR={mrr:.3} (floor {mrr_floor:.3})"
    );
    assert!(
        recall_at_5 >= recall_floor,
        "Hybrid Recall@5={recall_at_5:.3} below floor {recall_floor:.3}"
    );
    assert!(
        mrr >= mrr_floor,
        "Hybrid MRR={mrr:.3} below floor {mrr_floor:.3}"
    );
}
