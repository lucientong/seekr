//! Call-site extraction for name-level references / callers.
//!
//! Reuses the same Tree-sitter tree as chunking. This is **not**
//! compiler- or LSP-grade resolution — only surface syntax mentions.

use std::path::Path;

use tree_sitter::Node;

use crate::parser::treesitter::SupportedLanguage;
use crate::parser::{CallKind, CallSite, CodeChunk};

/// Extract call sites from an already-parsed syntax tree root.
pub fn extract_call_sites(
    root: &Node,
    source: &str,
    path: &Path,
    lang: SupportedLanguage,
    chunks: &[CodeChunk],
) -> Vec<CallSite> {
    if !matches!(
        lang,
        SupportedLanguage::Rust
            | SupportedLanguage::Python
            | SupportedLanguage::TypeScript
            | SupportedLanguage::Tsx
            | SupportedLanguage::JavaScript
    ) {
        return Vec::new();
    }

    let mut sites = Vec::new();
    walk_calls(root, source, path, lang, chunks, &mut sites);
    sites
}

fn walk_calls(
    node: &Node,
    source: &str,
    path: &Path,
    lang: SupportedLanguage,
    chunks: &[CodeChunk],
    out: &mut Vec<CallSite>,
) {
    if let Some(site) = node_to_call_site(node, source, path, lang, chunks) {
        out.push(site);
    }

    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            let child = cursor.node();
            walk_calls(&child, source, path, lang, chunks, out);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
}

fn node_to_call_site(
    node: &Node,
    source: &str,
    path: &Path,
    lang: SupportedLanguage,
    chunks: &[CodeChunk],
) -> Option<CallSite> {
    let kind = node.kind();
    let (call_kind, callee_node) = match lang {
        SupportedLanguage::Rust => match kind {
            "call_expression" => (CallKind::Function, node.child_by_field_name("function")?),
            "method_call_expression" => (CallKind::Method, node.child_by_field_name("name")?),
            "macro_invocation" => (CallKind::Macro, node.child_by_field_name("macro")?),
            _ => return None,
        },
        SupportedLanguage::Python => {
            if kind != "call" {
                return None;
            }
            (CallKind::Function, node.child_by_field_name("function")?)
        }
        SupportedLanguage::TypeScript | SupportedLanguage::Tsx | SupportedLanguage::JavaScript => {
            match kind {
                "call_expression" => (CallKind::Function, node.child_by_field_name("function")?),
                "new_expression" => {
                    let ctor = node.child_by_field_name("constructor").or_else(|| {
                        // Some grammars expose the type as a child after `new`.
                        (0..node.child_count())
                            .filter_map(|i| node.child(i))
                            .find(|child| !matches!(child.kind(), "new" | "(" | ")" | "arguments"))
                    })?;
                    (CallKind::Constructor, ctor)
                }
                _ => return None,
            }
        }
        _ => return None,
    };

    let (callee_name, is_member) = callee_tail(callee_node, source)?;
    if callee_name.is_empty() || !is_identifier_like(&callee_name) {
        return None;
    }

    let call_kind = if is_member && call_kind == CallKind::Function {
        CallKind::Method
    } else {
        call_kind
    };

    let byte_range = node.start_byte()..node.end_byte();
    let line_range = node.start_position().row..node.end_position().row + 1;
    let body = source
        .get(byte_range.clone())
        .unwrap_or("")
        .trim()
        .to_string();
    if body.is_empty() {
        return None;
    }

    let enclosing_chunk = find_enclosing_chunk(chunks, &byte_range);
    let enclosing_chunk_id = enclosing_chunk.map(|chunk| chunk.id);
    let relative_start = enclosing_chunk
        .map(|chunk| byte_range.start.saturating_sub(chunk.byte_range.start))
        .unwrap_or(byte_range.start);
    let id = stable_call_site_id(
        path,
        &call_kind,
        &callee_name,
        enclosing_chunk_id,
        relative_start,
        &body,
    );

    Some(CallSite {
        id,
        file_path: path.to_path_buf(),
        language: lang.name().to_string(),
        callee_name,
        call_kind,
        is_member,
        byte_range,
        line_range,
        enclosing_chunk_id,
        body,
    })
}

fn callee_tail(node: Node, source: &str) -> Option<(String, bool)> {
    match node.kind() {
        "identifier" | "property_identifier" | "field_identifier" | "type_identifier" => {
            let text = node.utf8_text(source.as_bytes()).ok()?.to_string();
            Some((text, false))
        }
        "scoped_identifier" | "member_expression" | "field_expression" | "attribute" => {
            // Prefer the right-most identifier segment.
            let mut last = None;
            let mut cursor = node.walk();
            if cursor.goto_first_child() {
                loop {
                    let child = cursor.node();
                    if matches!(
                        child.kind(),
                        "identifier"
                            | "property_identifier"
                            | "field_identifier"
                            | "type_identifier"
                    ) {
                        last = child.utf8_text(source.as_bytes()).ok().map(str::to_string);
                    } else if matches!(
                        child.kind(),
                        "scoped_identifier"
                            | "member_expression"
                            | "field_expression"
                            | "attribute"
                    ) {
                        if let Some((name, _)) = callee_tail(child, source) {
                            last = Some(name);
                        }
                    }
                    if !cursor.goto_next_sibling() {
                        break;
                    }
                }
            }
            last.map(|name| (name, true))
        }
        _ => {
            // Fallback: take trailing identifier-like token from node text.
            let text = node.utf8_text(source.as_bytes()).ok()?.trim();
            let tail = text.rsplit(['.', ':']).next().unwrap_or(text).trim();
            if is_identifier_like(tail) {
                Some((tail.to_string(), text.contains('.') || text.contains("::")))
            } else {
                None
            }
        }
    }
}

fn is_identifier_like(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_alphabetic() || first == '_') && chars.all(|c| c.is_alphanumeric() || c == '_')
}

fn find_enclosing_chunk<'a>(
    chunks: &'a [CodeChunk],
    call_range: &std::ops::Range<usize>,
) -> Option<&'a CodeChunk> {
    chunks
        .iter()
        .filter(|chunk| {
            chunk.byte_range.start <= call_range.start && chunk.byte_range.end >= call_range.end
        })
        .min_by_key(|chunk| chunk.byte_range.end - chunk.byte_range.start)
}

fn stable_call_site_id(
    path: &Path,
    kind: &CallKind,
    callee: &str,
    enclosing_chunk_id: Option<u64>,
    relative_start: usize,
    body: &str,
) -> u64 {
    let mut hasher = blake3::Hasher::new();
    let path = path.to_string_lossy();
    let kind = kind.as_str();
    for field in [
        path.as_bytes(),
        kind.as_bytes(),
        callee.as_bytes(),
        body.as_bytes(),
    ] {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    hasher.update(&enclosing_chunk_id.unwrap_or_default().to_le_bytes());
    hasher.update(&(relative_start as u64).to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(
        digest.as_bytes()[..8]
            .try_into()
            .expect("fixed digest size"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::chunker::chunk_file;
    use crate::parser::treesitter::SupportedLanguage;

    #[test]
    fn extracts_rust_function_and_method_calls() {
        let source = r#"
fn caller() {
    authenticate_user("a", "b");
    user.verify_password("x");
    println!("hi");
}
"#;
        let path = Path::new("lib.rs");
        let parsed = chunk_file(path, source, SupportedLanguage::Rust).unwrap();
        let names: Vec<_> = parsed
            .call_sites
            .iter()
            .map(|site| site.callee_name.as_str())
            .collect();
        assert!(names.contains(&"authenticate_user"));
        assert!(names.contains(&"verify_password"));
        assert!(names.contains(&"println"));
        assert!(
            parsed
                .call_sites
                .iter()
                .any(|site| site.callee_name == "verify_password" && site.is_member)
        );
    }

    #[test]
    fn extracts_python_and_typescript_calls() {
        let py = "def run():\n    authenticate(user, password)\n    svc.get_user_profile(1)\n";
        let py_parsed = chunk_file(Path::new("a.py"), py, SupportedLanguage::Python).unwrap();
        let py_names: Vec<_> = py_parsed
            .call_sites
            .iter()
            .map(|s| s.callee_name.as_str())
            .collect();
        assert!(py_names.contains(&"authenticate"));
        assert!(py_names.contains(&"get_user_profile"));

        let ts = "export function run() { fetchUserProfile(1); new TokenBucket(10); }";
        let ts_parsed = chunk_file(Path::new("a.ts"), ts, SupportedLanguage::TypeScript).unwrap();
        let ts_names: Vec<_> = ts_parsed
            .call_sites
            .iter()
            .map(|s| s.callee_name.as_str())
            .collect();
        assert!(ts_names.contains(&"fetchUserProfile"));
        assert!(ts_names.contains(&"TokenBucket"));
        assert!(
            ts_parsed
                .call_sites
                .iter()
                .any(|s| s.callee_name == "TokenBucket" && s.call_kind == CallKind::Constructor)
        );
    }

    #[test]
    fn call_site_ids_are_stable() {
        let source = "fn a() { foo(); }\n";
        let path = Path::new("stable.rs");
        let first = chunk_file(path, source, SupportedLanguage::Rust).unwrap();
        let second = chunk_file(path, source, SupportedLanguage::Rust).unwrap();
        assert_eq!(first.call_sites.len(), second.call_sites.len());
        assert_eq!(first.call_sites[0].id, second.call_sites[0].id);
    }
}
