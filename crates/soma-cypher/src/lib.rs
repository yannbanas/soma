//! # soma-cypher — Cypher query language for SOMA
//!
//! Parses a subset of openCypher and executes queries against StigreGraph.
//! Supports MATCH, WHERE, RETURN, CREATE, DELETE with SOMA's bio-inspired edges.

mod ast;
mod executor;
mod lexer;
mod parser;

pub use ast::*;
pub use executor::{CypherExecutor, CypherResult, CypherValue};
pub use lexer::{Lexer, Token};
pub use parser::Parser;
