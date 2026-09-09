use std::path::PathBuf;

use seekr_code::index::store::SeekrIndex;
use seekr_code::parser::{ChunkKind, CodeChunk};

fn chunk(id: u64, name: &str, body: &str) -> CodeChunk {
    CodeChunk {
        id,
        file_path: PathBuf::from(format!("src/{name}.rs")),
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
fn bm25_retrieval_quality_fixture_has_perfect_mrr() {
    let chunks = vec![
        chunk(
            1,
            "verify_password",
            "verify password hash and reject invalid credentials",
        ),
        chunk(
            2,
            "invalidate_cache",
            "invalidate stale cache entries after configuration changes",
        ),
        chunk(
            3,
            "rollback_transaction",
            "rollback database transaction when persistence fails",
        ),
        chunk(
            4,
            "generic_handler",
            "handle request response data value result state context",
        ),
    ];
    let embeddings = vec![vec![0.0; 4]; chunks.len()];
    let index = SeekrIndex::build_from(&chunks, &embeddings, 4);
    let judgments = [
        ("verify password hash", 1),
        ("invalidate cache entries", 2),
        ("database transaction rollback", 3),
    ];

    let reciprocal_rank_sum: f32 = judgments
        .iter()
        .map(|(query, relevant_id)| {
            index
                .search_bm25(query, 10)
                .iter()
                .position(|hit| hit.chunk_id == *relevant_id)
                .map(|rank| 1.0 / (rank + 1) as f32)
                .unwrap_or(0.0)
        })
        .sum();
    let mean_reciprocal_rank = reciprocal_rank_sum / judgments.len() as f32;

    assert_eq!(mean_reciprocal_rank, 1.0);
}
