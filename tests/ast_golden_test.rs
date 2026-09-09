//! AST DSL golden suite for Rust / TypeScript / Python.
//!
//! Measures per-language and overall F1 for the existing Seekr signature DSL.
//! Only strong matches (score >= [`STRONG_MATCH_THRESHOLD`]) count toward F1.
//! Structural false positives (wrong kind among strong matches) must be zero.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use seekr_code::index::store::SeekrIndex;
use seekr_code::parser::chunker::chunk_file_from_path;
use seekr_code::parser::summary::generate_summary;
use seekr_code::parser::{ChunkKind, CodeChunk};
use seekr_code::search::ast_pattern::{parse_pattern, search_ast_pattern};

/// Kind-only DSL matches score 0.5; require a stronger multi-field hit.
const STRONG_MATCH_THRESHOLD: f32 = 0.8;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn build_index(paths: &[&str]) -> SeekrIndex {
    let mut chunks: Vec<CodeChunk> = Vec::new();
    for relative in paths {
        let path = fixtures_dir().join(relative);
        let parsed = chunk_file_from_path(&path)
            .expect("parse fixture")
            .expect("fixture should produce chunks");
        chunks.extend(parsed.chunks);
    }

    let dim = 8;
    let embeddings: Vec<Vec<f32>> = chunks
        .iter()
        .map(|chunk| {
            let summary = generate_summary(chunk);
            let mut embedding = vec![0.0; dim];
            for (idx, byte) in summary.bytes().take(dim).enumerate() {
                embedding[idx] = byte as f32 / 255.0;
            }
            embedding
        })
        .collect();

    SeekrIndex::build_from(&chunks, &embeddings, dim)
}

#[derive(Clone)]
struct Judgment {
    query: &'static str,
    /// Expected chunk names (stable across content-addressed IDs).
    relevant_names: &'static [&'static str],
    /// Kinds that must never appear among strong matches.
    forbidden_kinds: &'static [ChunkKind],
}

fn evaluate(index: &SeekrIndex, judgments: &[Judgment]) -> (f32, f32, usize) {
    let mut precision_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut structural_fp = 0usize;

    for judgment in judgments {
        let hits = search_ast_pattern(index, judgment.query, 32).expect("ast search");
        let retrieved: Vec<&CodeChunk> = hits
            .iter()
            .filter(|hit| hit.score >= STRONG_MATCH_THRESHOLD)
            .filter_map(|hit| index.get_chunk(hit.chunk_id))
            .collect();

        let relevant: HashSet<&str> = judgment.relevant_names.iter().copied().collect();
        let retrieved_names: HashSet<&str> = retrieved
            .iter()
            .filter_map(|chunk| chunk.name.as_deref())
            .collect();

        let tp = retrieved_names.intersection(&relevant).count() as f32;
        let precision = if retrieved_names.is_empty() {
            if relevant.is_empty() { 1.0 } else { 0.0 }
        } else {
            tp / retrieved_names.len() as f32
        };
        let recall = if relevant.is_empty() {
            1.0
        } else {
            tp / relevant.len() as f32
        };

        precision_sum += precision;
        recall_sum += recall;

        for chunk in &retrieved {
            if judgment.forbidden_kinds.contains(&chunk.kind) {
                structural_fp += 1;
            }
        }

        parse_pattern(judgment.query).expect("pattern should parse");
    }

    let n = judgments.len() as f32;
    let precision = precision_sum / n;
    let recall = recall_sum / n;
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    (f1, recall, structural_fp)
}

#[test]
fn rust_ast_dsl_golden_f1() {
    let index = build_index(&["sample.rs"]);
    let judgments = [
        Judgment {
            query: "fn authenticate*",
            relevant_names: &["authenticate_user"],
            forbidden_kinds: &[ChunkKind::Struct, ChunkKind::Enum],
        },
        Judgment {
            query: "fn(*) -> Result",
            relevant_names: &["authenticate_user", "find_user_by_name"],
            forbidden_kinds: &[ChunkKind::Struct, ChunkKind::Enum],
        },
        Judgment {
            query: "fn(*) -> bool",
            relevant_names: &["verify_password"],
            forbidden_kinds: &[ChunkKind::Struct, ChunkKind::Enum],
        },
        Judgment {
            query: "struct User",
            relevant_names: &["User"],
            forbidden_kinds: &[ChunkKind::Function, ChunkKind::Enum],
        },
        Judgment {
            query: "enum AuthError",
            relevant_names: &["AuthError"],
            forbidden_kinds: &[ChunkKind::Function, ChunkKind::Struct],
        },
        Judgment {
            query: "fn calculate*",
            relevant_names: &["calculate_total"],
            forbidden_kinds: &[ChunkKind::Struct],
        },
    ];

    let (f1, _recall, structural_fp) = evaluate(&index, &judgments);
    assert_eq!(
        structural_fp, 0,
        "Rust structural false positives must be 0"
    );
    assert!(f1 >= 0.90, "Rust AST DSL F1 must be >= 0.90, got {f1:.3}");
}

#[test]
fn typescript_ast_dsl_golden_f1() {
    let index = build_index(&["sample.ts"]);
    let judgments = [
        Judgment {
            query: "async fn fetch*",
            relevant_names: &["fetchUserProfile"],
            forbidden_kinds: &[ChunkKind::Class, ChunkKind::Interface],
        },
        Judgment {
            query: "fn formatCurrency(*)",
            relevant_names: &["formatCurrency"],
            forbidden_kinds: &[ChunkKind::Class],
        },
        Judgment {
            query: "class TokenBucket",
            relevant_names: &["TokenBucket"],
            forbidden_kinds: &[ChunkKind::Function],
        },
        Judgment {
            query: "fn tryAcquire(*)",
            relevant_names: &["tryAcquire"],
            forbidden_kinds: &[ChunkKind::Class],
        },
    ];

    let (f1, _recall, structural_fp) = evaluate(&index, &judgments);
    assert_eq!(
        structural_fp, 0,
        "TypeScript structural false positives must be 0"
    );
    assert!(
        f1 >= 0.90,
        "TypeScript AST DSL F1 must be >= 0.90, got {f1:.3}"
    );
}

#[test]
fn python_ast_dsl_golden_f1() {
    let index = build_index(&["sample.py"]);
    let judgments = [
        Judgment {
            query: "fn authenticate(*)",
            relevant_names: &["authenticate"],
            forbidden_kinds: &[ChunkKind::Class],
        },
        Judgment {
            query: "fn calculate_discount(*)",
            relevant_names: &["calculate_discount"],
            forbidden_kinds: &[ChunkKind::Class],
        },
        Judgment {
            query: "class UserService",
            relevant_names: &["UserService"],
            forbidden_kinds: &[ChunkKind::Function],
        },
        Judgment {
            query: "fn get_user_profile(*)",
            relevant_names: &["get_user_profile"],
            forbidden_kinds: &[ChunkKind::Class],
        },
    ];

    let (f1, _recall, structural_fp) = evaluate(&index, &judgments);
    assert_eq!(
        structural_fp, 0,
        "Python structural false positives must be 0"
    );
    assert!(f1 >= 0.90, "Python AST DSL F1 must be >= 0.90, got {f1:.3}");
}

#[test]
fn overall_ast_dsl_golden_f1() {
    let index = build_index(&["sample.rs", "sample.ts", "sample.py"]);
    let judgments = [
        Judgment {
            query: "fn authenticate*",
            relevant_names: &["authenticate_user", "authenticate"],
            forbidden_kinds: &[ChunkKind::Struct, ChunkKind::Class, ChunkKind::Enum],
        },
        Judgment {
            query: "class *",
            relevant_names: &[
                "UserService",
                "TokenBucket",
                "AuthenticationError",
                "UserNotFoundError",
            ],
            forbidden_kinds: &[ChunkKind::Function],
        },
        Judgment {
            query: "fn calculate*",
            relevant_names: &["calculate_total", "calculate_discount"],
            forbidden_kinds: &[ChunkKind::Class, ChunkKind::Struct],
        },
        Judgment {
            query: "async fn fetch*",
            relevant_names: &["fetchUserProfile"],
            forbidden_kinds: &[ChunkKind::Class, ChunkKind::Struct],
        },
        Judgment {
            query: "struct User",
            relevant_names: &["User"],
            forbidden_kinds: &[ChunkKind::Function, ChunkKind::Class],
        },
    ];

    let (f1, _recall, structural_fp) = evaluate(&index, &judgments);
    assert_eq!(
        structural_fp, 0,
        "Overall structural false positives must be 0"
    );
    assert!(
        f1 >= 0.95,
        "Overall AST DSL F1 must be >= 0.95, got {f1:.3}"
    );
}
