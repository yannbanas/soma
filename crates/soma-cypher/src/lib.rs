//! # soma-cypher — Cypher query language for SOMA
//!
//! Parses a subset of openCypher and executes queries against StigreGraph.
//! Supports MATCH, WHERE, RETURN, CREATE, DELETE with SOMA's bio-inspired edges.

mod ast;
mod lexer;
mod parser;
mod executor;

pub use ast::*;
pub use lexer::{Token, Lexer};
pub use parser::Parser;
pub use executor::{CypherExecutor, CypherResult, CypherValue};
