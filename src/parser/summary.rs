//! Code summary generator.
//!
//! Generates structured summaries for code chunks, including function
//! signatures, documentation comments, parameter types, and return types.
//! These summaries improve embedding quality for semantic search.

use crate::parser::CodeChunk;

/// Generate a text summary for a code chunk, suitable for embedding.
///
/// The summary combines:
/// - The chunk kind and name
/// - The function/method signature
/// - Documentation comments
/// - Key structural information
///
/// This produces a more semantically meaningful text than just the raw
/// source code, improving embedding quality for semantic search.
pub fn generate_summary(chunk: &CodeChunk) -> String {
    let mut parts = Vec::new();

    // Add kind + name
    if let Some(ref name) = chunk.name {
        parts.push(format!("{} {}", chunk.kind, name));
    }

    // Add language context
    parts.push(format!("language: {}", chunk.language));

    // Add signature if different from name
    if let Some(ref sig) = chunk.signature {
        let sig_trimmed = sig.trim();
        if !sig_trimmed.is_empty() {
            parts.push(format!("signature: {}", sig_trimmed));
        }
    }

    // Add documentation
    if let Some(ref doc) = chunk.doc_comment {
        let doc_trimmed = doc.trim();
        if !doc_trimmed.is_empty() {
            // Limit doc comment length to avoid overwhelming the embedding
            let truncated = if doc_trimmed.len() > 500 {
                format!("{}...", &doc_trimmed[..500])
            } else {
                doc_trimmed.to_string()
            };
            parts.push(truncated);
        }
    }

    // Add a snippet of the body (first few meaningful lines)
    let body_snippet = extract_body_snippet(&chunk.body, 5);
    if !body_snippet.is_empty() {
        parts.push(body_snippet);
    }

    parts.join("\n")
}

/// Extract a brief snippet from the body, skipping boilerplate.
fn extract_body_snippet(body: &str, max_lines: usize) -> String {
    let meaningful_lines: Vec<&str> = body
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty()
                && !trimmed.starts_with("//")
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("///")
                && !trimmed.starts_with("/*")
                && !trimmed.starts_with('*')
                && trimmed != "{"
                && trimmed != "}"
                && trimmed != "("
                && trimmed != ")"
        })
        .take(max_lines)
        .collect();

    meaningful_lines.join("\n")
}

/// Generate summaries for a batch of chunks, returning (chunk_id, summary) pairs.
pub fn generate_summaries(chunks: &[CodeChunk]) -> Vec<(u64, String)> {
    chunks
        .iter()
        .map(|chunk| (chunk.id, generate_summary(chunk)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ChunkKind;
    use std::path::PathBuf;

    fn make_chunk(
        kind: ChunkKind,
        name: &str,
        signature: Option<&str>,
        doc: Option<&str>,
        body: &str,
    ) -> CodeChunk {
        CodeChunk {
            id: 1,
            file_path: PathBuf::from("test.rs"),
            language: "rust".to_string(),
            kind,
            name: Some(name.to_string()),
            signature: signature.map(String::from),
            doc_comment: doc.map(String::from),
            body: body.to_string(),
            byte_range: 0..body.len(),
            line_range: 0..body.lines().count(),
        }
    }

    #[test]
    fn test_generate_summary() {
        let chunk = make_chunk(
            ChunkKind::Function,
            "authenticate_user",
            Some("pub fn authenticate_user(username: &str, password: &str) -> Result<String, AuthError>"),
            Some("Validates the provided credentials against the database."),
            "pub fn authenticate_user(username: &str, password: &str) -> Result<String, AuthError> {\n    let user = find_user(username)?;\n    verify(password, &user.hash)\n}",
        );

        let summary = generate_summary(&chunk);
        assert!(summary.contains("function authenticate_user"));
        assert!(summary.contains("language: rust"));
        assert!(summary.contains("Validates"));
        assert!(summary.contains("signature:"));
    }

    #[test]
    fn test_generate_summaries_batch() {
        let chunks = vec![
            make_chunk(ChunkKind::Function, "foo", None, None, "fn foo() {}"),
            make_chunk(ChunkKind::Struct, "Bar", None, None, "struct Bar {}"),
        ];

        let summaries = generate_summaries(&chunks);
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].0, 1); // chunk id
    }
}
