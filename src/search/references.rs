//! Name-level references / callers over Tree-sitter call-site mentions.
//!
//! Explicitly **not** compiler- or LSP-grade resolution. Results join
//! definition chunks and call-site mentions by normalized identifier only.

use std::path::PathBuf;

use crate::parser::{CallKind, CallSite, ChunkKind, CodeChunk};

pub const REFERENCES_DISCLAIMER: &str =
    "Tree-sitter name-level inference only — not compiler/LSP resolution.";

pub const DEFAULT_REFERENCE_LIMIT: usize = 100;
pub const MAX_REFERENCE_LIMIT: usize = 1_000;

#[derive(Debug, Clone)]
pub struct ReferenceOptions {
    pub path_prefix: Option<PathBuf>,
    pub languages: Vec<String>,
    pub limit: Option<usize>,
}

impl Default for ReferenceOptions {
    fn default() -> Self {
        Self {
            path_prefix: None,
            languages: Vec::new(),
            limit: Some(DEFAULT_REFERENCE_LIMIT),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ReferenceHit {
    pub definition: CodeChunk,
    pub mention: CallSite,
    pub confidence: f32,
    pub reasons: Vec<String>,
    pub disclaimer: &'static str,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CallerHit {
    pub caller_chunk: Option<CodeChunk>,
    pub mention: CallSite,
    pub matched_definitions: Vec<CodeChunk>,
    pub confidence: f32,
    pub reasons: Vec<String>,
    pub disclaimer: &'static str,
}

pub fn score_reference(
    definition: &CodeChunk,
    mention: &CallSite,
    definition_count: usize,
) -> (f32, Vec<String>) {
    if definition.language != mention.language
        || definition.name.as_deref() != Some(mention.callee_name.as_str())
    {
        return (0.05, vec!["incompatible_name_or_language".to_string()]);
    }

    let mut score: f32 = 0.45;
    let mut reasons = vec!["exact_name_and_language".to_string()];

    if is_simple_identifier(&mention.callee_name) && !mention.is_member {
        score += 0.1;
        reasons.push("simple_identifier".to_string());
    }

    if definition_count == 1 {
        score += 0.2;
        reasons.push("unique_definition".to_string());
    } else if definition_count > 1 {
        score -= 0.15;
        reasons.push("ambiguous_definitions".to_string());
    }

    if definition.file_path == mention.file_path {
        score += 0.15;
        reasons.push("same_file".to_string());
    }

    let member_call = mention.is_member || mention.call_kind == CallKind::Method;
    if member_call {
        score -= 0.15;
        reasons.push("member_or_method_call".to_string());
    }

    if mention.call_kind == CallKind::Constructor {
        score += 0.05;
        reasons.push("constructor_call".to_string());
    } else if mention.call_kind == CallKind::Function
        && matches!(definition.kind, ChunkKind::Class | ChunkKind::Struct)
    {
        score -= 0.1;
        reasons.push("possible_constructor_call".to_string());
    }

    if mention.call_kind == CallKind::Macro {
        score -= 0.1;
        reasons.push("macro_invocation".to_string());
    }

    let mut confidence = score.clamp(0.05, 0.95);
    if member_call {
        confidence = confidence.min(0.55);
    }
    (confidence, reasons)
}

fn is_simple_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_alphabetic() || first == '_')
        && chars.all(|c| c.is_alphanumeric() || c == '_')
        && !name.contains('.')
        && !name.contains("::")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ChunkKind;
    use std::path::PathBuf;

    fn chunk(name: &str, path: &str) -> CodeChunk {
        CodeChunk {
            id: 1,
            file_path: PathBuf::from(path),
            language: "rust".into(),
            kind: ChunkKind::Function,
            name: Some(name.into()),
            signature: None,
            doc_comment: None,
            body: format!("fn {name}() {{}}"),
            byte_range: 0..10,
            line_range: 0..1,
        }
    }

    fn mention(name: &str, path: &str, member: bool) -> CallSite {
        CallSite {
            id: 2,
            file_path: PathBuf::from(path),
            language: "rust".into(),
            callee_name: name.into(),
            call_kind: if member {
                CallKind::Method
            } else {
                CallKind::Function
            },
            is_member: member,
            byte_range: 0..5,
            line_range: 0..1,
            enclosing_chunk_id: Some(9),
            body: format!("{name}()"),
        }
    }

    #[test]
    fn unique_same_file_simple_call_is_high_confidence() {
        let (score, reasons) =
            score_reference(&chunk("foo", "a.rs"), &mention("foo", "a.rs", false), 1);
        assert!(score >= 0.8, "score={score}");
        assert!(reasons.iter().any(|r| r == "unique_definition"));
        assert!(reasons.iter().any(|r| r == "same_file"));
    }

    #[test]
    fn member_and_ambiguity_reduce_confidence() {
        let (score, reasons) =
            score_reference(&chunk("foo", "a.rs"), &mention("foo", "b.rs", true), 3);
        assert!(score < 0.5, "score={score}");
        assert!(reasons.iter().any(|r| r == "ambiguous_definitions"));
        assert!(reasons.iter().any(|r| r == "member_or_method_call"));
    }
}
