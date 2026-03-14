//! Cypher AST types.

use std::collections::HashMap;

/// Top-level Cypher statement.
#[derive(Debug, Clone)]
pub enum CypherStatement {
    /// MATCH ... WHERE ... RETURN ...
    Query(CypherQuery),
    /// CREATE (a)-[:Rel]->(b)
    Create(CreateClause),
    /// MATCH ... DELETE n
    Delete(DeleteClause),
    /// MATCH ... SET n.prop = val
    Set(SetClause),
}

/// A read query: MATCH + optional WHERE + RETURN.
#[derive(Debug, Clone)]
pub struct CypherQuery {
    pub match_clause: MatchClause,
    pub where_clause: Option<WhereExpr>,
    pub return_clause: ReturnClause,
    pub order_by: Option<OrderBy>,
    pub limit: Option<usize>,
}

/// MATCH clause — one or more graph patterns.
#[derive(Debug, Clone)]
pub struct MatchClause {
    pub patterns: Vec<Pattern>,
}

/// A graph pattern: chain of nodes and relationships.
/// Example: (a:Entity)-[:Trail*1..3]->(b:Concept)
#[derive(Debug, Clone)]
pub struct Pattern {
    pub elements: Vec<PatternElement>,
}

/// Element of a pattern: either a node or a relationship.
#[derive(Debug, Clone)]
pub enum PatternElement {
    Node(NodePattern),
    Relationship(RelPattern),
}

/// (var:Label {prop: val, ...})
#[derive(Debug, Clone)]
pub struct NodePattern {
    pub variable: Option<String>,
    pub label: Option<String>,
    pub properties: HashMap<String, PropValue>,
}

/// -[:Type*min..max {props}]->
#[derive(Debug, Clone)]
pub struct RelPattern {
    pub variable: Option<String>,
    pub rel_type: Option<String>,
    pub direction: Direction,
    pub min_hops: Option<u8>,
    pub max_hops: Option<u8>,
    pub properties: HashMap<String, PropValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// ->
    Outgoing,
    /// <-
    Incoming,
    /// --
    Both,
}

/// Property value in a pattern.
#[derive(Debug, Clone)]
pub enum PropValue {
    String(String),
    Float(f64),
    Int(i64),
    Bool(bool),
}

/// WHERE clause — boolean expression tree.
#[derive(Debug, Clone)]
pub enum WhereExpr {
    /// var.prop op value
    Comparison(Comparison),
    /// expr AND expr
    And(Box<WhereExpr>, Box<WhereExpr>),
    /// expr OR expr
    Or(Box<WhereExpr>, Box<WhereExpr>),
    /// NOT expr
    Not(Box<WhereExpr>),
    /// var.prop CONTAINS "str"
    Contains(PropertyAccess, String),
}

#[derive(Debug, Clone)]
pub struct Comparison {
    pub left: PropertyAccess,
    pub op: CompOp,
    pub right: PropValue,
}

#[derive(Debug, Clone)]
pub struct PropertyAccess {
    pub variable: String,
    pub property: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompOp {
    Eq,      // =
    Neq,     // <>
    Lt,      // <
    Gt,      // >
    Lte,     // <=
    Gte,     // >=
}

/// RETURN clause.
#[derive(Debug, Clone)]
pub struct ReturnClause {
    pub items: Vec<ReturnItem>,
}

#[derive(Debug, Clone)]
pub enum ReturnItem {
    /// var.prop or var
    Property(PropertyAccess),
    /// var (return full node)
    Variable(String),
    /// COUNT(var), etc.
    Function(String, String),
}

/// ORDER BY clause.
#[derive(Debug, Clone)]
pub struct OrderBy {
    pub property: PropertyAccess,
    pub descending: bool,
}

/// CREATE clause.
#[derive(Debug, Clone)]
pub struct CreateClause {
    /// Patterns to match first (nodes that must exist)
    pub match_clause: Option<MatchClause>,
    /// The relationship to create
    pub from_var: String,
    pub to_var: String,
    pub rel_type: String,
    pub properties: HashMap<String, PropValue>,
}

/// DELETE clause.
#[derive(Debug, Clone)]
pub struct DeleteClause {
    pub match_clause: MatchClause,
    pub where_clause: Option<WhereExpr>,
    pub variables: Vec<String>,
}

/// SET clause.
#[derive(Debug, Clone)]
pub struct SetClause {
    pub match_clause: MatchClause,
    pub where_clause: Option<WhereExpr>,
    pub assignments: Vec<(PropertyAccess, PropValue)>,
}
