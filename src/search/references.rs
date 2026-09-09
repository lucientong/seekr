//! Name-level references / callers over Tree-sitter call-site mentions.
//!
//! Explicitly **not** compiler- or LSP-grade resolution. Results join
//! definition chunks and call-site mentions by normalized identifier only.

use std::path::PathBuf;

use crate::parser::{CallKind, CallSite, CodeChunk};
use crate::search::symbol::normalize_symbol_name;

pub const REFERENCES_DISCLAIMER: &str =
    "Tree-sitter name-level inference only — not compiler/LSP resolution.";

#[derive(Debug, Clone, Default)]
pub struct ReferenceOptions {
    pub path_prefix: Option<PathBuf>,
    pub languages: Vec<String>,
    pub limit: Option<usize>,
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
    let mut score: f32 = 0.35;
    let mut reasons = vec!["normalized_name_match".to_string()];

    let def_name = definition
        .name
        .as_deref()
        .map(normalize_symbol_name)
        .unwrap_or_default();
    let mention_name = normalize_symbol_name(&mention.callee_name);
    if def_name == mention_name && is_simple_identifier(&mention.callee_name) && !mention.is_member
    {
        score += 0.25;
        reasons.push("simple_identifier".to_string());
    }

    if definition_count == 1 {
        score += 0.25;
        reasons.push("unique_definition".to_string());
    } else if definition_count > 1 {
        score -= 0.15;
        reasons.push("ambiguous_definitions".to_string());
    }

    if definition.file_path == mention.file_path {
        score += 0.15;
        reasons.push("same_file".to_string());
    }

    if mention.is_member || mention.call_kind == CallKind::Method {
        score -= 0.15;
        reasons.push("member_or_method_call".to_string());
    }

    if mention.call_kind == CallKind::Constructor {
        // Constructors often share type names with structs/classes — mild demotion.
        score -= 0.05;
        reasons.push("constructor_call".to_string());
    }

    if mention.call_kind == CallKind::Macro {
        score -= 0.1;
        reasons.push("macro_invocation".to_string());
    }

    if definition.language != mention.language {
        score -= 0.2;
        reasons.push("cross_language".to_string());
    }

    let confidence = score.clamp(0.05, 0.95);
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
