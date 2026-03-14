use once_cell::sync::Lazy;
use regex::Regex;

use soma_core::Channel;

/// A compiled regex pattern for triplet extraction.
struct PatternDef {
    regex: Regex,
    channel: Channel,
    /// If true, swap subject/object
    swap: bool,
}

/// Pre-compiled L1 patterns. Compiled once, reused across all calls.
/// Performance: once_cell ensures zero overhead after first use.
static PATTERNS: Lazy<Vec<PatternDef>> = Lazy::new(|| {
    vec![
        // "X is a Y" / "X est un Y" — allows multi-word subject (up to 3 words)
        PatternDef {
            regex: Regex::new(r"(?i)(\S+(?:\s\S+){0,2})\s+(?:est un|est une|is a|is an)\s+(.+?)(?:\.|$)").unwrap(),
            channel: Channel::Trail,
            swap: false,
        },
        // "X derives from Y" / "X dérive de Y"
        PatternDef {
            regex: Regex::new(r"(?i)(\S+(?:\s\S+)?)\s+(?:dérive de|derives? from|derived from|based on)\s+(\S+(?:\s\S+)?)").unwrap(),
            channel: Channel::DerivesDe,
            swap: false,
        },
        // "X causes Y" / "X provoque Y"
        PatternDef {
            regex: Regex::new(r"(?i)(\S+(?:\s\S+)?)\s+(?:cause[sd]?|provoque|triggers?)\s+(\S+(?:\s\S+)?)").unwrap(),
            channel: Channel::Causal,
            swap: false,
        },
        // "AVOID X" / "ÉVITER X" / "X incompatible Y" — allows multi-word after AVOID
        PatternDef {
            regex: Regex::new(r"(?i)(?:ÉVITER|AVOID)\s+(?:combining\s+)?(\S+(?:\s\S+)?)(?:\s+(?:avec|with|due)\s+(\S+(?:\s\S+)?))?").unwrap(),
            channel: Channel::Alarm,
            swap: false,
        },
        // "X replaces Y" / "X remplace Y"
        PatternDef {
            regex: Regex::new(r"(?i)(\S+)\s+(?:remplace|replaces?)\s+(\S+)").unwrap(),
            channel: Channel::DerivesDe,
            swap: false,
        },
        // "X uses Y" / "X utilise Y" — allows multi-word subject
        PatternDef {
            regex: Regex::new(r"(?i)(\S+(?:\s\S+){0,2})\s+(?:utilise|uses?)\s+(\S+(?:\s\S+)?)").unwrap(),
            channel: Channel::Trail,
            swap: false,
        },
        // Emission pattern: "X emission at/=/: Ynm" or "emit at Ynm"
        PatternDef {
            regex: Regex::new(r"(?i)(\S+)\s+(?:émission|emission|emit|emits?)\s*(?:at|[=:])\s*([\d.]+\s*nm)").unwrap(),
            channel: Channel::Causal,
            swap: false,
        },
        // "engineered to emit at Xnm" — reverse: emission as object
        PatternDef {
            regex: Regex::new(r"(?i)(\S+)\s+(?:has been engineered to|engineered to)\s+emit\s+at\s+(?:exactly\s+)?([\d.]+\s*nm)").unwrap(),
            channel: Channel::Causal,
            swap: false,
        },
        // pLDDT pattern: "X pLDDT=94.2" or "with a pLDDT=94.2"
        PatternDef {
            regex: Regex::new(r"(?i)(\S+)\s+(?:.*\s)?pLDDT\s*[=:>]\s*([\d.]+)").unwrap(),
            channel: Channel::Trail,
            swap: false,
        },
        // "X has Y" / "X a un Y" — for properties
        PatternDef {
            regex: Regex::new(r"(?i)(\S+(?:\s\S+)?)\s+(?:has|a un|possède)\s+(?:a |an )?(.+?)(?:\.|,|$)").unwrap(),
            channel: Channel::Trail,
            swap: false,
        },
        // "X depends on Y" / "X dépend de Y"
        PatternDef {
            regex: Regex::new(r"(?i)(\S+(?:\s\S+)?)\s+(?:dépend de|depends? on|requires?)\s+(\S+(?:\s\S+)?)").unwrap(),
            channel: Channel::Trail,
            swap: false,
        },
        // "X produces Y" / "X produit Y"
        PatternDef {
            regex: Regex::new(r"(?i)(\S+(?:\s\S+)?)\s+(?:produit|produces?|generates?|creates?)\s+(\S+(?:\s\S+)?)").unwrap(),
            channel: Channel::Causal,
            swap: false,
        },
        // "X achieves Y" / "X atteint Y" — for measurements/benchmarks
        PatternDef {
            regex: Regex::new(r"(?i)(\S+(?:\s\S+)?)\s+(?:achieves?|atteint|reaches?)\s+(.+?)(?:\.|,|$)").unwrap(),
            channel: Channel::Trail,
            swap: false,
        },
        // "X compiles to Y" / "X compile vers Y"
        PatternDef {
            regex: Regex::new(r"(?i)(\S+(?:\s\S+)?)\s+(?:compile[sd]?\s+(?:to|vers)|runs?\s+(?:on|in))\s+(\S+(?:\s\S+)?)").unwrap(),
            channel: Channel::Trail,
            swap: false,
        },
        // "X with Y" / "X avec Y" — property association (lower priority)
        PatternDef {
            regex: Regex::new(r"(?i)(\S+)\s+(?:with a|with an|avec un|avec une)\s+(.+?)(?:\.|,|$)").unwrap(),
            channel: Channel::Trail,
            swap: false,
        },
    ]
});

/// Extracted triplet: (subject, object, channel).
#[derive(Debug, Clone)]
pub struct ExtractedTriplet {
    pub subject: String,
    pub object: String,
    pub channel: Channel,
}

/// Pattern-based triplet extractor (L1 level).
pub struct PatternExtractor;

impl PatternExtractor {
    /// Extract triplets from a chunk of text using compiled regex patterns.
    /// Returns all matched triplets.
    ///
    /// Performance: patterns are compiled once (Lazy), matching is O(n × p)
    /// where n = text length, p = number of patterns.
    pub fn extract(text: &str) -> Vec<ExtractedTriplet> {
        let mut triplets = Vec::new();

        for pattern in PATTERNS.iter() {
            for cap in pattern.regex.captures_iter(text) {
                let s = cap.get(1).map(|m| m.as_str().trim().to_string());
                let o = cap.get(2).and_then(|m| {
                    let val = m.as_str().trim().to_string();
                    if val.is_empty() {
                        None
                    } else {
                        Some(val)
                    }
                });

                let (subject, object) = match (s, o) {
                    (Some(s), Some(o)) if !s.is_empty() => {
                        if pattern.swap {
                            (o, s)
                        } else {
                            (s, o)
                        }
                    }
                    (Some(s), None) if !s.is_empty() => (s, "WARNING".to_string()),
                    _ => continue,
                };

                triplets.push(ExtractedTriplet {
                    subject,
                    object,
                    channel: pattern.channel,
                });
            }
        }

        // Deduplicate by (subject, object, channel)
        triplets.dedup_by(|a, b| {
            a.subject == b.subject && a.object == b.object && a.channel == b.channel
        });

        triplets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_is_a() {
        let triplets = PatternExtractor::extract("ChromoQ is a fluorescent protein.");
        assert!(!triplets.is_empty());
        assert_eq!(triplets[0].subject, "ChromoQ");
        assert!(triplets[0].object.contains("fluorescent protein"));
        assert_eq!(triplets[0].channel, Channel::Trail);
    }

    #[test]
    fn extract_derives_from() {
        let triplets = PatternExtractor::extract("ChromoQ derives from EGFP");
        assert!(!triplets.is_empty());
        assert_eq!(triplets[0].subject, "ChromoQ");
        assert_eq!(triplets[0].object, "EGFP");
        assert_eq!(triplets[0].channel, Channel::DerivesDe);
    }

    #[test]
    fn extract_causes() {
        let triplets = PatternExtractor::extract("mutation causes instability");
        assert!(!triplets.is_empty());
        assert_eq!(triplets[0].channel, Channel::Causal);
    }

    #[test]
    fn extract_alarm() {
        let triplets = PatternExtractor::extract("AVOID using deprecated API");
        assert!(!triplets.is_empty());
        assert_eq!(triplets[0].channel, Channel::Alarm);
    }

    #[test]
    fn extract_emission() {
        let triplets = PatternExtractor::extract("ChromoQ emission=523nm");
        assert!(!triplets.is_empty());
    }

    #[test]
    fn extract_plddt() {
        let triplets = PatternExtractor::extract("ChromoQ pLDDT=94.2");
        assert!(!triplets.is_empty());
        assert_eq!(triplets[0].subject, "ChromoQ");
    }

    #[test]
    fn no_false_positives_on_noise() {
        let triplets = PatternExtractor::extract("Hello world, this is just random text.");
        assert!(triplets.is_empty());
    }

    #[test]
    fn case_insensitive() {
        let t1 = PatternExtractor::extract("chromoq IS A protein.");
        let t2 = PatternExtractor::extract("CHROMOQ is a PROTEIN.");
        assert!(!t1.is_empty());
        assert!(!t2.is_empty());
    }
}
