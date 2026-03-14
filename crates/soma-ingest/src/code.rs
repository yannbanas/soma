//! Code graph ingestion — parses Rust source files into SOMA graph triplets.
//!
//! Inspired by CodeGraphContext's multi-phase approach:
//! 1. Pre-scan: build global symbol → location map
//! 2. Parse: extract nodes (functions, structs, traits, etc.) per file
//! 3. Edge resolution: create CALLS/IMPORTS edges using symbol map

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use soma_core::{Channel, NodeKind};
use soma_graph::StigreGraph;

/// Result of code ingestion.
#[derive(Debug, Default)]
pub struct CodeIngestResult {
    pub files_processed: usize,
    pub functions_found: usize,
    pub structs_found: usize,
    pub traits_found: usize,
    pub impls_found: usize,
    pub edges_created: usize,
}

/// A code symbol extracted from source.
#[derive(Debug, Clone)]
struct CodeSymbol {
    name: String,
    kind: SymbolKind,
    file: String,
    line: usize,
    /// Functions this symbol calls (by name)
    calls: Vec<String>,
    /// Types this symbol uses/references
    uses_types: Vec<String>,
    /// Trait being implemented (for impl blocks)
    implements: Option<String>,
    /// Cyclomatic complexity estimate
    #[allow(dead_code)]
    complexity: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Mod,
}

/// Ingest all Rust source files from a directory into a SOMA graph.
pub fn ingest_rust_directory(
    graph: &mut StigreGraph,
    dir: &Path,
    source_prefix: &str,
) -> CodeIngestResult {
    let mut result = CodeIngestResult::default();

    // Phase 1: Collect all .rs files
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "rs")
            && !path.to_string_lossy().contains("/target/")
        {
            files.push(path.to_path_buf());
        }
    }

    // Phase 2: Parse all files, collect symbols
    let mut all_symbols: Vec<CodeSymbol> = Vec::new();
    let mut symbol_map: HashMap<String, Vec<(String, usize)>> = HashMap::new(); // name → [(file, line)]

    for file in &files {
        let relative = file
            .strip_prefix(dir)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();

        let Ok(content) = std::fs::read_to_string(file) else {
            continue;
        };

        let symbols = parse_rust_file(&content, &relative);

        for sym in &symbols {
            symbol_map
                .entry(sym.name.clone())
                .or_default()
                .push((sym.file.clone(), sym.line));
        }

        result.files_processed += 1;
        all_symbols.extend(symbols);
    }

    // Phase 3: Create nodes in SOMA graph
    for sym in &all_symbols {
        let (tags, kind) = match sym.kind {
            SymbolKind::Function => {
                result.functions_found += 1;
                (vec!["function".into(), "rust".into()], NodeKind::Procedure)
            }
            SymbolKind::Struct => {
                result.structs_found += 1;
                (vec!["struct".into(), "rust".into()], NodeKind::Entity)
            }
            SymbolKind::Enum => {
                result.structs_found += 1;
                (vec!["enum".into(), "rust".into()], NodeKind::Entity)
            }
            SymbolKind::Trait => {
                result.traits_found += 1;
                (vec!["trait".into(), "rust".into()], NodeKind::Concept)
            }
            SymbolKind::Impl => {
                result.impls_found += 1;
                continue; // Impl blocks create edges, not nodes
            }
            SymbolKind::Mod => (vec!["module".into(), "rust".into()], NodeKind::Entity),
        };

        let node_id = graph.upsert_node_with_tags(&sym.name, kind, tags);

        // Set metadata (file, line, complexity)
        // Meta would require mutable node access — tracked via tags/label for now
        let _ = node_id;

        // Create File→Symbol CONTAINS edge (permanent)
        let file_label = &sym.file;
        let file_id = graph.upsert_node_with_tags(
            file_label,
            NodeKind::Entity,
            vec!["file".into(), "rust".into()],
        );
        graph.upsert_edge_labeled(
            file_id,
            node_id,
            Channel::DerivesDe,
            0.95,
            source_prefix,
            Some("contains"),
        );
        result.edges_created += 1;
    }

    // Phase 4: Create edges (CALLS, IMPLEMENTS)
    for sym in &all_symbols {
        let Some(from_id) = graph.node_id_by_label(&sym.name) else {
            continue;
        };

        // CALLS edges
        for call_name in &sym.calls {
            // Resolve: does this call target a known symbol?
            let confidence = if symbol_map.contains_key(call_name) {
                // Check if in same file (high confidence) or cross-file
                let locations = &symbol_map[call_name];
                if locations.iter().any(|(f, _)| f == &sym.file) {
                    0.95 // Same file
                } else {
                    0.8 // Cross-file via import
                }
            } else {
                0.4 // Unknown — might be external crate
            };

            let to_id = graph.upsert_node_with_tags(
                call_name,
                NodeKind::Procedure,
                vec!["function".into(), "rust".into()],
            );

            graph.upsert_edge_labeled(
                from_id,
                to_id,
                Channel::Trail,
                confidence,
                source_prefix,
                Some("calls"),
            );
            result.edges_created += 1;
        }

        // IMPLEMENTS edges (trait implementations)
        if let Some(ref trait_name) = sym.implements {
            let trait_id = graph.upsert_node_with_tags(
                trait_name,
                NodeKind::Concept,
                vec!["trait".into(), "rust".into()],
            );
            graph.upsert_edge_labeled(
                from_id,
                trait_id,
                Channel::Causal,
                0.95,
                source_prefix,
                Some("implements"),
            );
            result.edges_created += 1;
        }

        // TYPE USAGE edges
        for type_name in &sym.uses_types {
            if let Some(to_id) = graph.node_id_by_label(type_name) {
                graph.upsert_edge_labeled(
                    from_id,
                    to_id,
                    Channel::Trail,
                    0.6,
                    source_prefix,
                    Some("uses_type"),
                );
                result.edges_created += 1;
            }
        }
    }

    result
}

/// Parse a single Rust file and extract code symbols.
fn parse_rust_file(content: &str, file_path: &str) -> Vec<CodeSymbol> {
    let Ok(syntax) = syn::parse_file(content) else {
        return Vec::new();
    };

    let mut symbols = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    for item in &syntax.items {
        extract_item(item, file_path, &lines, &mut symbols, None);
    }

    symbols
}

fn extract_item(
    item: &syn::Item,
    file_path: &str,
    lines: &[&str],
    symbols: &mut Vec<CodeSymbol>,
    parent_name: Option<&str>,
) {
    match item {
        syn::Item::Fn(f) => {
            let name = if let Some(parent) = parent_name {
                format!("{}::{}", parent, f.sig.ident)
            } else {
                f.sig.ident.to_string()
            };

            let line = span_to_line(&f.sig.ident.span(), lines);
            let calls = extract_function_calls(&f.block);
            let uses_types = extract_type_references(&f.sig);
            let complexity = estimate_complexity(&f.block);

            symbols.push(CodeSymbol {
                name,
                kind: SymbolKind::Function,
                file: file_path.to_string(),
                line,
                calls,
                uses_types,
                implements: None,
                complexity,
            });
        }
        syn::Item::Struct(s) => {
            let name = s.ident.to_string();
            let line = span_to_line(&s.ident.span(), lines);
            symbols.push(CodeSymbol {
                name,
                kind: SymbolKind::Struct,
                file: file_path.to_string(),
                line,
                calls: Vec::new(),
                uses_types: Vec::new(),
                implements: None,
                complexity: 0,
            });
        }
        syn::Item::Enum(e) => {
            let name = e.ident.to_string();
            let line = span_to_line(&e.ident.span(), lines);
            symbols.push(CodeSymbol {
                name,
                kind: SymbolKind::Enum,
                file: file_path.to_string(),
                line,
                calls: Vec::new(),
                uses_types: Vec::new(),
                implements: None,
                complexity: 0,
            });
        }
        syn::Item::Trait(t) => {
            let name = t.ident.to_string();
            let line = span_to_line(&t.ident.span(), lines);
            symbols.push(CodeSymbol {
                name,
                kind: SymbolKind::Trait,
                file: file_path.to_string(),
                line,
                calls: Vec::new(),
                uses_types: Vec::new(),
                implements: None,
                complexity: 0,
            });
        }
        syn::Item::Impl(imp) => {
            // Get the type being implemented
            let self_type = type_to_string(&imp.self_ty);

            // Get trait name if this is a trait impl
            let trait_name = imp.trait_.as_ref().map(|(_, path, _)| {
                path.segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default()
            });

            // If implementing a trait, create an impl symbol for the edge
            if let Some(ref tname) = trait_name {
                symbols.push(CodeSymbol {
                    name: self_type.clone(),
                    kind: SymbolKind::Impl,
                    file: file_path.to_string(),
                    line: 0,
                    calls: Vec::new(),
                    uses_types: Vec::new(),
                    implements: Some(tname.clone()),
                    complexity: 0,
                });
            }

            // Extract methods
            for impl_item in &imp.items {
                if let syn::ImplItem::Fn(method) = impl_item {
                    let method_name = format!("{}::{}", self_type, method.sig.ident);
                    let line = span_to_line(&method.sig.ident.span(), lines);
                    let calls = extract_function_calls(&method.block);
                    let uses_types = extract_type_references(&method.sig);
                    let complexity = estimate_complexity(&method.block);

                    symbols.push(CodeSymbol {
                        name: method_name,
                        kind: SymbolKind::Function,
                        file: file_path.to_string(),
                        line,
                        calls,
                        uses_types,
                        implements: None,
                        complexity,
                    });
                }
            }
        }
        syn::Item::Mod(m) => {
            let name = m.ident.to_string();
            symbols.push(CodeSymbol {
                name: name.clone(),
                kind: SymbolKind::Mod,
                file: file_path.to_string(),
                line: span_to_line(&m.ident.span(), lines),
                calls: Vec::new(),
                uses_types: Vec::new(),
                implements: None,
                complexity: 0,
            });

            // Recurse into inline modules
            if let Some((_, items)) = &m.content {
                for sub in items {
                    extract_item(sub, file_path, lines, symbols, Some(&name));
                }
            }
        }
        _ => {}
    }
}

/// Extract function call names from a block of code.
fn extract_function_calls(block: &syn::Block) -> Vec<String> {
    let mut calls = Vec::new();
    let source: String = quote::quote!(#block).to_string();

    // Simple heuristic extraction: find patterns like `ident(`
    let tokens: Vec<&str> = source.split_whitespace().collect();
    for token in &tokens {
        if let Some(name) = (*token).strip_suffix('(') {
            let clean: &str = name.trim_start_matches('&').trim_start_matches("mut ");
            let clean: &str = clean.rsplit("::").next().unwrap_or(clean);
            if !clean.is_empty()
                && clean.chars().next().is_some_and(|c: char| c.is_lowercase())
                && !is_rust_keyword(clean)
            {
                let owned = clean.to_string();
                if !calls.contains(&owned) {
                    calls.push(owned);
                }
            }
        }
    }

    calls
}

/// Extract type references from a function signature.
fn extract_type_references(sig: &syn::Signature) -> Vec<String> {
    let mut types = Vec::new();

    // Return type
    if let syn::ReturnType::Type(_, ty) = &sig.output {
        collect_type_names(ty, &mut types);
    }

    // Parameter types
    for arg in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            collect_type_names(&pat_type.ty, &mut types);
        }
    }

    types
}

fn collect_type_names(ty: &syn::Type, names: &mut Vec<String>) {
    match ty {
        syn::Type::Path(tp) => {
            if let Some(seg) = tp.path.segments.last() {
                let name = seg.ident.to_string();
                if name.chars().next().is_some_and(|c| c.is_uppercase())
                    && !is_std_type(&name)
                    && !names.contains(&name)
                {
                    names.push(name);
                }
            }
        }
        syn::Type::Reference(r) => collect_type_names(&r.elem, names),
        syn::Type::Slice(s) => collect_type_names(&s.elem, names),
        syn::Type::Array(a) => collect_type_names(&a.elem, names),
        syn::Type::Tuple(t) => {
            for elem in &t.elems {
                collect_type_names(elem, names);
            }
        }
        _ => {}
    }
}

/// Estimate cyclomatic complexity of a function body.
fn estimate_complexity(block: &syn::Block) -> u32 {
    let source: String = quote::quote!(#block).to_string();
    let mut complexity = 1u32;

    for keyword in [
        "if ", "else if", "match ", "for ", "while ", "loop ", "&&", "||", "?",
    ] {
        complexity += source.matches(keyword).count() as u32;
    }

    complexity
}

fn type_to_string(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(tp) => tp
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_else(|| "Unknown".into()),
        _ => "Unknown".into(),
    }
}

fn span_to_line(span: &proc_macro2::Span, _lines: &[&str]) -> usize {
    span.start().line
}

fn is_rust_keyword(s: &str) -> bool {
    matches!(
        s,
        "if" | "else"
            | "match"
            | "for"
            | "while"
            | "loop"
            | "let"
            | "mut"
            | "ref"
            | "return"
            | "break"
            | "continue"
            | "fn"
            | "pub"
            | "use"
            | "mod"
            | "struct"
            | "enum"
            | "trait"
            | "impl"
            | "where"
            | "as"
            | "in"
            | "self"
            | "super"
            | "crate"
            | "type"
            | "const"
            | "static"
            | "async"
            | "await"
            | "move"
            | "dyn"
            | "unsafe"
            | "extern"
            | "true"
            | "false"
            | "some"
            | "none"
            | "ok"
            | "err"
    )
}

fn is_std_type(s: &str) -> bool {
    matches!(
        s,
        "String"
            | "Vec"
            | "HashMap"
            | "HashSet"
            | "Option"
            | "Result"
            | "Box"
            | "Arc"
            | "Rc"
            | "Mutex"
            | "RwLock"
            | "Cell"
            | "RefCell"
            | "Cow"
            | "Pin"
            | "Future"
            | "Iterator"
            | "Display"
            | "Debug"
            | "Clone"
            | "Copy"
            | "Default"
            | "Send"
            | "Sync"
            | "Sized"
            | "Drop"
            | "From"
            | "Into"
            | "AsRef"
            | "AsMut"
            | "Deref"
            | "DerefMut"
            | "Fn"
            | "FnMut"
            | "FnOnce"
            | "Serialize"
            | "Deserialize"
            | "Self"
            | "PathBuf"
            | "Path"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_function() {
        let code = r#"
            pub fn hello(name: &str) -> String {
                format!("Hello {}", name)
            }
        "#;
        let symbols = parse_rust_file(code, "test.rs");
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].name, "hello");
        assert_eq!(symbols[0].kind, SymbolKind::Function);
    }

    #[test]
    fn parse_struct_and_impl() {
        let code = r#"
            pub struct MyGraph {
                nodes: Vec<Node>,
            }

            impl MyGraph {
                pub fn new() -> Self {
                    MyGraph { nodes: Vec::new() }
                }

                pub fn add_node(&mut self, node: Node) {
                    self.nodes.push(node);
                }
            }
        "#;
        let symbols = parse_rust_file(code, "test.rs");
        // Should find: MyGraph (struct) + MyGraph::new (fn) + MyGraph::add_node (fn)
        assert!(symbols.len() >= 3);
        assert!(symbols
            .iter()
            .any(|s| s.name == "MyGraph" && s.kind == SymbolKind::Struct));
        assert!(symbols
            .iter()
            .any(|s| s.name == "MyGraph::new" && s.kind == SymbolKind::Function));
    }

    #[test]
    fn parse_trait_impl() {
        let code = r#"
            pub trait Searchable {
                fn search(&self, query: &str) -> Vec<String>;
            }

            pub struct Engine;

            impl Searchable for Engine {
                fn search(&self, query: &str) -> Vec<String> {
                    vec![]
                }
            }
        "#;
        let symbols = parse_rust_file(code, "test.rs");
        assert!(symbols
            .iter()
            .any(|s| s.name == "Searchable" && s.kind == SymbolKind::Trait));
        assert!(symbols
            .iter()
            .any(|s| s.kind == SymbolKind::Impl && s.implements == Some("Searchable".into())));
    }

    #[test]
    fn parse_enum() {
        let code = r#"
            pub enum Color {
                Red,
                Green,
                Blue,
            }
        "#;
        let symbols = parse_rust_file(code, "test.rs");
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].name, "Color");
        assert_eq!(symbols[0].kind, SymbolKind::Enum);
    }

    #[test]
    fn complexity_estimation() {
        let code = r#"
            fn complex(x: i32) -> i32 {
                if x > 0 {
                    if x > 10 {
                        match x {
                            1 => 1,
                            2 => 2,
                            _ => 3,
                        }
                    } else {
                        for i in 0..x {
                            println!("{}", i);
                        }
                        0
                    }
                } else {
                    -1
                }
            }
        "#;
        let symbols = parse_rust_file(code, "test.rs");
        assert!(!symbols.is_empty());
        // Should have complexity > 1 (has if, match, for)
        assert!(
            symbols[0].complexity > 3,
            "complexity={}",
            symbols[0].complexity
        );
    }

    #[test]
    fn ingest_real_directory() {
        // Ingest the soma-core crate as a test
        let soma_core = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("soma-core/src");
        if soma_core.exists() {
            let mut graph = StigreGraph::new("test-code", 0.05);
            let result = ingest_rust_directory(&mut graph, &soma_core, "test");
            assert!(result.files_processed > 0, "Should process at least 1 file");
            assert!(result.functions_found > 0, "Should find functions");
            assert!(result.structs_found > 0, "Should find structs");
            assert!(result.edges_created > 0, "Should create edges");
        }
    }
}
