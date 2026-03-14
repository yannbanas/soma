//! Cypher parser — transforms tokens into AST.

use std::collections::HashMap;

use crate::ast::*;
use crate::lexer::Token;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    /// Parse a complete Cypher statement.
    pub fn parse(&mut self) -> Result<CypherStatement, String> {
        match self.peek() {
            Token::Match => {
                let match_clause = self.parse_match()?;

                // Check what follows: WHERE, RETURN, DELETE, SET, CREATE
                match self.peek() {
                    Token::Where => {
                        let where_clause = Some(self.parse_where()?);
                        match self.peek() {
                            Token::Return => {
                                let return_clause = self.parse_return()?;
                                let order_by = self.parse_optional_order_by()?;
                                let limit = self.parse_optional_limit()?;
                                Ok(CypherStatement::Query(CypherQuery {
                                    match_clause,
                                    where_clause,
                                    return_clause,
                                    order_by,
                                    limit,
                                }))
                            }
                            Token::Delete => {
                                self.advance();
                                let variables = self.parse_ident_list()?;
                                Ok(CypherStatement::Delete(DeleteClause {
                                    match_clause,
                                    where_clause,
                                    variables,
                                }))
                            }
                            Token::Set => {
                                let assignments = self.parse_set_assignments()?;
                                Ok(CypherStatement::Set(SetClause {
                                    match_clause,
                                    where_clause,
                                    assignments,
                                }))
                            }
                            _ => Err("Expected RETURN, DELETE, or SET after WHERE".into()),
                        }
                    }
                    Token::Return => {
                        let return_clause = self.parse_return()?;
                        let order_by = self.parse_optional_order_by()?;
                        let limit = self.parse_optional_limit()?;
                        Ok(CypherStatement::Query(CypherQuery {
                            match_clause,
                            where_clause: None,
                            return_clause,
                            order_by,
                            limit,
                        }))
                    }
                    Token::Delete => {
                        self.advance();
                        let variables = self.parse_ident_list()?;
                        Ok(CypherStatement::Delete(DeleteClause {
                            match_clause,
                            where_clause: None,
                            variables,
                        }))
                    }
                    Token::Set => {
                        let assignments = self.parse_set_assignments()?;
                        Ok(CypherStatement::Set(SetClause {
                            match_clause,
                            where_clause: None,
                            assignments,
                        }))
                    }
                    Token::Create => {
                        // MATCH ... CREATE (a)-[:Rel]->(b)
                        self.advance();
                        self.parse_create_rel(Some(match_clause))
                    }
                    _ => Err(format!("Unexpected token after MATCH: {:?}", self.peek())),
                }
            }
            Token::Create => {
                self.advance();
                self.parse_create_rel(None)
            }
            _ => Err(format!("Expected MATCH or CREATE, got {:?}", self.peek())),
        }
    }

    // --- helpers ---

    fn peek(&self) -> Token {
        self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.peek();
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        let tok = self.advance();
        if &tok == expected {
            Ok(())
        } else {
            Err(format!("Expected {:?}, got {:?}", expected, tok))
        }
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        match self.advance() {
            Token::Ident(s) => Ok(s),
            other => Err(format!("Expected identifier, got {:?}", other)),
        }
    }

    // --- MATCH ---

    fn parse_match(&mut self) -> Result<MatchClause, String> {
        self.expect(&Token::Match)?;
        let mut patterns = Vec::new();
        patterns.push(self.parse_pattern()?);

        // Multiple patterns separated by commas
        while self.peek() == Token::Comma {
            self.advance();
            patterns.push(self.parse_pattern()?);
        }

        Ok(MatchClause { patterns })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, String> {
        let mut elements = Vec::new();

        // Must start with a node
        elements.push(PatternElement::Node(self.parse_node_pattern()?));

        // Optional chain of -[rel]->(node) or <-[rel]-(node)
        loop {
            match self.peek() {
                Token::Dash => {
                    let rel = self.parse_rel_pattern()?;
                    elements.push(PatternElement::Relationship(rel));
                    elements.push(PatternElement::Node(self.parse_node_pattern()?));
                }
                Token::LeftArrow => {
                    let rel = self.parse_rel_pattern()?;
                    elements.push(PatternElement::Relationship(rel));
                    elements.push(PatternElement::Node(self.parse_node_pattern()?));
                }
                _ => break,
            }
        }

        Ok(Pattern { elements })
    }

    fn parse_node_pattern(&mut self) -> Result<NodePattern, String> {
        self.expect(&Token::LParen)?;

        let mut variable = None;
        let mut label = None;
        let mut properties = HashMap::new();

        // Optional variable name
        if let Token::Ident(_) = self.peek() {
            variable = Some(self.expect_ident()?);
        }

        // Optional :Label
        if self.peek() == Token::Colon {
            self.advance();
            label = Some(self.expect_ident()?);
        }

        // Optional {prop: val, ...}
        if self.peek() == Token::LBrace {
            properties = self.parse_properties()?;
        }

        self.expect(&Token::RParen)?;

        Ok(NodePattern {
            variable,
            label,
            properties,
        })
    }

    fn parse_rel_pattern(&mut self) -> Result<RelPattern, String> {
        // Determine direction start
        let left_arrow = self.peek() == Token::LeftArrow;
        if left_arrow {
            self.advance(); // consume <-
        } else {
            self.expect(&Token::Dash)?; // consume -
        }

        let mut variable = None;
        let mut rel_type = None;
        let mut min_hops = None;
        let mut max_hops = None;
        let mut properties = HashMap::new();

        // Optional [...]
        if self.peek() == Token::LBracket {
            self.advance();

            // Optional variable
            if let Token::Ident(_) = self.peek() {
                variable = Some(self.expect_ident()?);
            }

            // Optional :Type
            if self.peek() == Token::Colon {
                self.advance();
                rel_type = Some(self.expect_ident()?);
            }

            // Optional *min..max
            if self.peek() == Token::Star {
                self.advance();
                if let Token::IntLit(n) = self.peek() {
                    min_hops = Some(n as u8);
                    self.advance();
                }
                if self.peek() == Token::DotDot {
                    self.advance();
                    if let Token::IntLit(n) = self.peek() {
                        max_hops = Some(n as u8);
                        self.advance();
                    }
                } else if min_hops.is_some() {
                    // *3 means exactly 3 hops
                    max_hops = min_hops;
                }
            }

            // Optional {props}
            if self.peek() == Token::LBrace {
                properties = self.parse_properties()?;
            }

            self.expect(&Token::RBracket)?;
        }

        // Direction end
        let direction = if left_arrow {
            // <-[...]-
            if self.peek() == Token::Dash {
                self.advance();
            }
            Direction::Incoming
        } else {
            // -[...]-> or -[...]-
            if self.peek() == Token::Arrow {
                self.advance();
                Direction::Outgoing
            } else if self.peek() == Token::Dash {
                self.advance();
                Direction::Both
            } else {
                Direction::Outgoing
            }
        };

        Ok(RelPattern {
            variable,
            rel_type,
            direction,
            min_hops,
            max_hops,
            properties,
        })
    }

    fn parse_properties(&mut self) -> Result<HashMap<String, PropValue>, String> {
        self.expect(&Token::LBrace)?;
        let mut props = HashMap::new();

        if self.peek() != Token::RBrace {
            loop {
                let key = self.expect_ident()?;
                self.expect(&Token::Colon)?;
                let val = self.parse_prop_value()?;
                props.insert(key, val);

                if self.peek() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        self.expect(&Token::RBrace)?;
        Ok(props)
    }

    fn parse_prop_value(&mut self) -> Result<PropValue, String> {
        match self.advance() {
            Token::StringLit(s) => Ok(PropValue::String(s)),
            Token::IntLit(n) => Ok(PropValue::Int(n)),
            Token::FloatLit(f) => Ok(PropValue::Float(f)),
            Token::BoolLit(b) => Ok(PropValue::Bool(b)),
            other => Err(format!("Expected property value, got {:?}", other)),
        }
    }

    // --- WHERE ---

    fn parse_where(&mut self) -> Result<WhereExpr, String> {
        self.expect(&Token::Where)?;
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> Result<WhereExpr, String> {
        let mut left = self.parse_and_expr()?;
        while self.peek() == Token::Or {
            self.advance();
            let right = self.parse_and_expr()?;
            left = WhereExpr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and_expr(&mut self) -> Result<WhereExpr, String> {
        let mut left = self.parse_not_expr()?;
        while self.peek() == Token::And {
            self.advance();
            let right = self.parse_not_expr()?;
            left = WhereExpr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not_expr(&mut self) -> Result<WhereExpr, String> {
        if self.peek() == Token::Not {
            self.advance();
            let expr = self.parse_primary_expr()?;
            return Ok(WhereExpr::Not(Box::new(expr)));
        }
        self.parse_primary_expr()
    }

    fn parse_primary_expr(&mut self) -> Result<WhereExpr, String> {
        // Parse property access: var.prop
        let prop = self.parse_property_access()?;

        // Check for CONTAINS
        if self.peek() == Token::Contains {
            self.advance();
            match self.advance() {
                Token::StringLit(s) => return Ok(WhereExpr::Contains(prop, s)),
                other => return Err(format!("Expected string after CONTAINS, got {:?}", other)),
            }
        }

        // Parse comparison operator
        let op = match self.advance() {
            Token::Eq => CompOp::Eq,
            Token::Neq => CompOp::Neq,
            Token::Lt => CompOp::Lt,
            Token::Gt => CompOp::Gt,
            Token::Lte => CompOp::Lte,
            Token::Gte => CompOp::Gte,
            other => return Err(format!("Expected comparison operator, got {:?}", other)),
        };

        let right = self.parse_prop_value()?;

        Ok(WhereExpr::Comparison(Comparison {
            left: prop,
            op,
            right,
        }))
    }

    fn parse_property_access(&mut self) -> Result<PropertyAccess, String> {
        let variable = self.expect_ident()?;
        self.expect(&Token::Dot)?;
        let property = self.expect_ident()?;
        Ok(PropertyAccess { variable, property })
    }

    // --- RETURN ---

    fn parse_return(&mut self) -> Result<ReturnClause, String> {
        self.expect(&Token::Return)?;
        let mut items = Vec::new();

        loop {
            let item = self.parse_return_item()?;
            items.push(item);

            if self.peek() == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }

        Ok(ReturnClause { items })
    }

    fn parse_return_item(&mut self) -> Result<ReturnItem, String> {
        let ident = self.expect_ident()?;

        // Check for function call: COUNT(var)
        if self.peek() == Token::LParen {
            self.advance();
            let arg = self.expect_ident()?;
            self.expect(&Token::RParen)?;
            return Ok(ReturnItem::Function(ident, arg));
        }

        // Check for property access: var.prop
        if self.peek() == Token::Dot {
            self.advance();
            let property = self.expect_ident()?;
            return Ok(ReturnItem::Property(PropertyAccess {
                variable: ident,
                property,
            }));
        }

        // Just a variable
        Ok(ReturnItem::Variable(ident))
    }

    fn parse_optional_order_by(&mut self) -> Result<Option<OrderBy>, String> {
        if self.peek() != Token::OrderBy {
            return Ok(None);
        }
        self.advance();

        let property = self.parse_property_access()?;
        let descending = match self.peek() {
            Token::Desc => {
                self.advance();
                true
            }
            Token::Asc => {
                self.advance();
                false
            }
            _ => false,
        };

        Ok(Some(OrderBy {
            property,
            descending,
        }))
    }

    fn parse_optional_limit(&mut self) -> Result<Option<usize>, String> {
        if self.peek() != Token::Limit {
            return Ok(None);
        }
        self.advance();

        match self.advance() {
            Token::IntLit(n) if n > 0 => Ok(Some(n as usize)),
            other => Err(format!(
                "Expected positive integer after LIMIT, got {:?}",
                other
            )),
        }
    }

    // --- CREATE ---

    fn parse_create_rel(
        &mut self,
        match_clause: Option<MatchClause>,
    ) -> Result<CypherStatement, String> {
        // (from_var)-[:RelType {props}]->(to_var)
        self.expect(&Token::LParen)?;
        let from_var = self.expect_ident()?;
        self.expect(&Token::RParen)?;

        self.expect(&Token::Dash)?;
        self.expect(&Token::LBracket)?;
        self.expect(&Token::Colon)?;
        let rel_type = self.expect_ident()?;

        let mut properties = HashMap::new();
        if self.peek() == Token::LBrace {
            properties = self.parse_properties()?;
        }

        self.expect(&Token::RBracket)?;
        self.expect(&Token::Arrow)?;
        self.expect(&Token::LParen)?;
        let to_var = self.expect_ident()?;
        self.expect(&Token::RParen)?;

        Ok(CypherStatement::Create(CreateClause {
            match_clause,
            from_var,
            to_var,
            rel_type,
            properties,
        }))
    }

    // --- SET ---

    fn parse_set_assignments(&mut self) -> Result<Vec<(PropertyAccess, PropValue)>, String> {
        self.expect(&Token::Set)?;
        let mut assignments = Vec::new();

        loop {
            let prop = self.parse_property_access()?;
            self.expect(&Token::Eq)?;
            let val = self.parse_prop_value()?;
            assignments.push((prop, val));

            if self.peek() == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }

        Ok(assignments)
    }

    // --- DELETE helpers ---

    fn parse_ident_list(&mut self) -> Result<Vec<String>, String> {
        let mut names = Vec::new();
        names.push(self.expect_ident()?);
        while self.peek() == Token::Comma {
            self.advance();
            names.push(self.expect_ident()?);
        }
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse(input: &str) -> Result<CypherStatement, String> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens);
        parser.parse()
    }

    #[test]
    fn parse_simple_match() {
        let stmt = parse("MATCH (n:Entity) RETURN n").unwrap();
        if let CypherStatement::Query(q) = stmt {
            assert_eq!(q.match_clause.patterns.len(), 1);
            assert!(q.where_clause.is_none());
            assert_eq!(q.return_clause.items.len(), 1);
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn parse_match_with_props() {
        let stmt = parse(r#"MATCH (a:Entity {label: "ChromoQ"}) RETURN a.label"#).unwrap();
        if let CypherStatement::Query(q) = stmt {
            let pat = &q.match_clause.patterns[0];
            if let PatternElement::Node(n) = &pat.elements[0] {
                assert_eq!(n.label.as_deref(), Some("Entity"));
                assert!(n.properties.contains_key("label"));
            }
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn parse_relationship_pattern() {
        let stmt = parse("MATCH (a:Entity)-[:Trail*1..3]->(b:Concept) RETURN b.label").unwrap();
        if let CypherStatement::Query(q) = stmt {
            let pat = &q.match_clause.patterns[0];
            assert_eq!(pat.elements.len(), 3); // node, rel, node
            if let PatternElement::Relationship(r) = &pat.elements[1] {
                assert_eq!(r.rel_type.as_deref(), Some("Trail"));
                assert_eq!(r.min_hops, Some(1));
                assert_eq!(r.max_hops, Some(3));
                assert_eq!(r.direction, Direction::Outgoing);
            }
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn parse_where_and() {
        let stmt =
            parse(r#"MATCH (n) WHERE n.intensity > 0.5 AND n.label CONTAINS "Rust" RETURN n"#)
                .unwrap();
        if let CypherStatement::Query(q) = stmt {
            assert!(q.where_clause.is_some());
            if let Some(WhereExpr::And(_, _)) = q.where_clause {
                // OK
            } else {
                panic!("Expected AND");
            }
        }
    }

    #[test]
    fn parse_order_by_limit() {
        let stmt = parse("MATCH (n:Entity) RETURN n.label ORDER BY n.label DESC LIMIT 10").unwrap();
        if let CypherStatement::Query(q) = stmt {
            assert!(q.order_by.is_some());
            assert!(q.order_by.as_ref().unwrap().descending);
            assert_eq!(q.limit, Some(10));
        }
    }

    #[test]
    fn parse_create() {
        let stmt = parse(
            r#"MATCH (a {label: "ChromoQ"}), (b {label: "fluor"}) CREATE (a)-[:Causal {confidence: 0.85}]->(b)"#
        ).unwrap();
        if let CypherStatement::Create(c) = stmt {
            assert!(c.match_clause.is_some());
            assert_eq!(c.rel_type, "Causal");
            assert_eq!(c.from_var, "a");
            assert_eq!(c.to_var, "b");
        } else {
            panic!("Expected Create");
        }
    }

    #[test]
    fn parse_delete() {
        let stmt = parse(r#"MATCH (n {label: "old"}) DELETE n"#).unwrap();
        if let CypherStatement::Delete(d) = stmt {
            assert_eq!(d.variables, vec!["n"]);
        } else {
            panic!("Expected Delete");
        }
    }
}
