//! Cypher lexer — tokenizes Cypher query strings.

/// Token types for the Cypher subset.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Match,
    Where,
    Return,
    Create,
    Delete,
    Set,
    OrderBy,
    Limit,
    And,
    Or,
    Not,
    Contains,
    Asc,
    Desc,
    // Literals
    Ident(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    // Symbols
    LParen,     // (
    RParen,     // )
    LBracket,   // [
    RBracket,   // ]
    LBrace,     // {
    RBrace,     // }
    Colon,      // :
    Comma,      // ,
    Dot,        // .
    Arrow,      // ->
    LeftArrow,  // <-
    Dash,       // -
    Star,       // *
    DotDot,     // ..
    Eq,         // =
    Neq,        // <>
    Lt,         // <
    Gt,         // >
    Lte,        // <=
    Gte,        // >=
    Eof,
}

pub struct Lexer {
    chars: Vec<char>,
    pos: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Lexer {
            chars: input.chars().collect(),
            pos: 0,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            if tok == Token::Eof {
                tokens.push(tok);
                break;
            }
            tokens.push(tok);
        }
        Ok(tokens)
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied();
        self.pos += 1;
        ch
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn next_token(&mut self) -> Result<Token, String> {
        self.skip_whitespace();

        let Some(ch) = self.peek() else {
            return Ok(Token::Eof);
        };

        match ch {
            '(' => { self.advance(); Ok(Token::LParen) }
            ')' => { self.advance(); Ok(Token::RParen) }
            '[' => { self.advance(); Ok(Token::LBracket) }
            ']' => { self.advance(); Ok(Token::RBracket) }
            '{' => { self.advance(); Ok(Token::LBrace) }
            '}' => { self.advance(); Ok(Token::RBrace) }
            ':' => { self.advance(); Ok(Token::Colon) }
            ',' => { self.advance(); Ok(Token::Comma) }
            '*' => { self.advance(); Ok(Token::Star) }
            '=' => { self.advance(); Ok(Token::Eq) }

            '.' => {
                self.advance();
                if self.peek() == Some('.') {
                    self.advance();
                    Ok(Token::DotDot)
                } else {
                    Ok(Token::Dot)
                }
            }

            '-' => {
                self.advance();
                if self.peek() == Some('>') {
                    self.advance();
                    Ok(Token::Arrow)
                } else {
                    Ok(Token::Dash)
                }
            }

            '<' => {
                self.advance();
                match self.peek() {
                    Some('-') => { self.advance(); Ok(Token::LeftArrow) }
                    Some('=') => { self.advance(); Ok(Token::Lte) }
                    Some('>') => { self.advance(); Ok(Token::Neq) }
                    _ => Ok(Token::Lt),
                }
            }

            '>' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::Gte)
                } else {
                    Ok(Token::Gt)
                }
            }

            '"' | '\'' => self.read_string(ch),

            c if c.is_ascii_digit() => self.read_number(),

            c if c.is_alphabetic() || c == '_' => self.read_ident(),

            _ => Err(format!("Unexpected character: '{}'", ch)),
        }
    }

    fn read_string(&mut self, quote: char) -> Result<Token, String> {
        self.advance(); // skip opening quote
        let mut s = String::new();
        loop {
            match self.advance() {
                Some(c) if c == quote => return Ok(Token::StringLit(s)),
                Some('\\') => {
                    match self.advance() {
                        Some('n') => s.push('\n'),
                        Some('t') => s.push('\t'),
                        Some('\\') => s.push('\\'),
                        Some(c) if c == quote => s.push(c),
                        Some(c) => { s.push('\\'); s.push(c); }
                        None => return Err("Unterminated string escape".into()),
                    }
                }
                Some(c) => s.push(c),
                None => return Err("Unterminated string literal".into()),
            }
        }
    }

    fn read_number(&mut self) -> Result<Token, String> {
        let mut s = String::new();
        let mut is_float = false;

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                s.push(ch);
                self.advance();
            } else if ch == '.' && !is_float {
                // Check it's not ".." (range)
                if self.chars.get(self.pos + 1) == Some(&'.') {
                    break;
                }
                is_float = true;
                s.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if is_float {
            s.parse::<f64>()
                .map(Token::FloatLit)
                .map_err(|e| format!("Invalid float: {}", e))
        } else {
            s.parse::<i64>()
                .map(Token::IntLit)
                .map_err(|e| format!("Invalid integer: {}", e))
        }
    }

    fn read_ident(&mut self) -> Result<Token, String> {
        let mut s = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                s.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        // Check for two-word keywords: ORDER BY
        let upper = s.to_uppercase();
        if upper == "ORDER" {
            let saved_pos = self.pos;
            self.skip_whitespace();
            let mut peek_word = String::new();
            let _peek_start = self.pos;
            while let Some(ch) = self.peek() {
                if ch.is_alphabetic() {
                    peek_word.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
            if peek_word.to_uppercase() == "BY" {
                return Ok(Token::OrderBy);
            }
            // Not ORDER BY — rewind
            self.pos = saved_pos;
        }

        match upper.as_str() {
            "MATCH" => Ok(Token::Match),
            "WHERE" => Ok(Token::Where),
            "RETURN" => Ok(Token::Return),
            "CREATE" => Ok(Token::Create),
            "DELETE" => Ok(Token::Delete),
            "SET" => Ok(Token::Set),
            "LIMIT" => Ok(Token::Limit),
            "AND" => Ok(Token::And),
            "OR" => Ok(Token::Or),
            "NOT" => Ok(Token::Not),
            "CONTAINS" => Ok(Token::Contains),
            "ASC" => Ok(Token::Asc),
            "DESC" => Ok(Token::Desc),
            "TRUE" => Ok(Token::BoolLit(true)),
            "FALSE" => Ok(Token::BoolLit(false)),
            _ => Ok(Token::Ident(s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_simple_match() {
        let mut lexer = Lexer::new("MATCH (n) RETURN n");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Match);
        assert_eq!(tokens[1], Token::LParen);
        assert_eq!(tokens[2], Token::Ident("n".into()));
        assert_eq!(tokens[3], Token::RParen);
        assert_eq!(tokens[4], Token::Return);
        assert_eq!(tokens[5], Token::Ident("n".into()));
        assert_eq!(tokens[6], Token::Eof);
    }

    #[test]
    fn tokenize_arrow() {
        let mut lexer = Lexer::new("-[:Trail]->");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Dash);
        assert_eq!(tokens[1], Token::LBracket);
        assert_eq!(tokens[2], Token::Colon);
        assert_eq!(tokens[3], Token::Ident("Trail".into()));
        assert_eq!(tokens[4], Token::RBracket);
        assert_eq!(tokens[5], Token::Arrow);
    }

    #[test]
    fn tokenize_string_and_number() {
        let mut lexer = Lexer::new(r#"{label: "ChromoQ", intensity: 0.85}"#);
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0], Token::LBrace);
        assert_eq!(tokens[1], Token::Ident("label".into()));
        assert_eq!(tokens[2], Token::Colon);
        assert_eq!(tokens[3], Token::StringLit("ChromoQ".into()));
        assert_eq!(tokens[4], Token::Comma);
        assert_eq!(tokens[5], Token::Ident("intensity".into()));
        assert_eq!(tokens[6], Token::Colon);
        assert_eq!(tokens[7], Token::FloatLit(0.85));
        assert_eq!(tokens[8], Token::RBrace);
    }

    #[test]
    fn tokenize_order_by() {
        let mut lexer = Lexer::new("ORDER BY n.label DESC LIMIT 10");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0], Token::OrderBy);
        assert_eq!(tokens[1], Token::Ident("n".into()));
        assert_eq!(tokens[2], Token::Dot);
        assert_eq!(tokens[3], Token::Ident("label".into()));
        assert_eq!(tokens[4], Token::Desc);
        assert_eq!(tokens[5], Token::Limit);
        assert_eq!(tokens[6], Token::IntLit(10));
    }

    #[test]
    fn tokenize_variable_length() {
        let mut lexer = Lexer::new("*1..3");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Star);
        assert_eq!(tokens[1], Token::IntLit(1));
        assert_eq!(tokens[2], Token::DotDot);
        assert_eq!(tokens[3], Token::IntLit(3));
    }
}
