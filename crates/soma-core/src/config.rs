use serde::{Deserialize, Serialize};

/// Global SOMA configuration, parsed from soma.toml.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SomaConfig {
    #[serde(default)]
    pub soma: SomaSection,
    #[serde(default)]
    pub bio: BioSection,
    #[serde(default)]
    pub hdc: HdcSection,
    #[serde(default)]
    pub ingest: IngestSection,
    #[serde(default)]
    pub llm: LlmSection,
    #[serde(default)]
    pub mcp: McpSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaSection {
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    #[serde(default = "default_workspace")]
    pub default_workspace: String,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioSection {
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: f32,
    #[serde(default = "default_physarum_interval")]
    pub physarum_interval_hours: f64,
    #[serde(default = "default_consolidation_interval")]
    pub consolidation_interval_hours: f64,
    #[serde(default = "default_pruning_interval")]
    pub pruning_interval_hours: f64,
    #[serde(default = "default_snapshot_interval")]
    pub snapshot_interval_hours: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcSection {
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    #[serde(default = "default_true")]
    pub tfidf: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestSection {
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmSection {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub embedding_model: Option<String>,
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSection {
    #[serde(default = "default_transport")]
    pub transport: String,
    #[serde(default = "default_tcp_port")]
    pub tcp_port: u16,
}

// Default value functions
fn default_data_dir() -> String {
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("LOCALAPPDATA") {
            return format!("{}/soma", appdata);
        }
    }
    "~/.local/share/soma".to_string()
}
fn default_workspace() -> String {
    "default".to_string()
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_prune_threshold() -> f32 {
    0.05
}
fn default_physarum_interval() -> f64 {
    2.0
}
fn default_consolidation_interval() -> f64 {
    6.0
}
fn default_pruning_interval() -> f64 {
    24.0
}
fn default_snapshot_interval() -> f64 {
    6.0
}
fn default_dimension() -> usize {
    10_000
}
fn default_window_size() -> usize {
    5
}
fn default_true() -> bool {
    true
}
fn default_chunk_size() -> usize {
    5
}
fn default_chunk_overlap() -> usize {
    1
}
fn default_provider() -> String {
    "ollama".to_string()
}
fn default_model() -> String {
    "cogito:8b".to_string()
}
fn default_timeout_ms() -> u64 {
    30_000
}
fn default_endpoint() -> String {
    "http://localhost:11434".to_string()
}
fn default_transport() -> String {
    "stdio".to_string()
}
fn default_tcp_port() -> u16 {
    3333
}

impl Default for SomaSection {
    fn default() -> Self {
        SomaSection {
            data_dir: default_data_dir(),
            default_workspace: default_workspace(),
            log_level: default_log_level(),
        }
    }
}

impl Default for BioSection {
    fn default() -> Self {
        BioSection {
            prune_threshold: default_prune_threshold(),
            physarum_interval_hours: default_physarum_interval(),
            consolidation_interval_hours: default_consolidation_interval(),
            pruning_interval_hours: default_pruning_interval(),
            snapshot_interval_hours: default_snapshot_interval(),
        }
    }
}

impl Default for HdcSection {
    fn default() -> Self {
        HdcSection {
            dimension: default_dimension(),
            window_size: default_window_size(),
            tfidf: default_true(),
        }
    }
}

impl Default for IngestSection {
    fn default() -> Self {
        IngestSection {
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
        }
    }
}

impl Default for LlmSection {
    fn default() -> Self {
        LlmSection {
            enabled: false,
            provider: default_provider(),
            model: default_model(),
            embedding_model: None,
            endpoint: default_endpoint(),
            timeout_ms: default_timeout_ms(),
        }
    }
}

impl Default for McpSection {
    fn default() -> Self {
        McpSection {
            transport: default_transport(),
            tcp_port: default_tcp_port(),
        }
    }
}

impl SomaConfig {
    /// Validate configuration values. Returns Err with description on invalid config.
    pub fn validate(&self) -> Result<(), String> {
        if self.hdc.dimension == 0 {
            return Err("hdc.dimension must be > 0".into());
        }
        if self.hdc.window_size == 0 {
            return Err("hdc.window_size must be > 0".into());
        }
        if self.ingest.chunk_size == 0 {
            return Err("ingest.chunk_size must be > 0".into());
        }
        if self.ingest.chunk_overlap >= self.ingest.chunk_size {
            return Err("ingest.chunk_overlap must be < chunk_size".into());
        }
        if !(0.0..=1.0).contains(&self.bio.prune_threshold) {
            return Err("bio.prune_threshold must be in [0.0, 1.0]".into());
        }
        Ok(())
    }

    /// Resolve data_dir: expand ~ to home directory.
    /// On Windows, also handles `~/.local/share/soma` → `%LOCALAPPDATA%\soma`.
    pub fn resolved_data_dir(&self) -> std::path::PathBuf {
        // Environment variable override (for Docker, CI, etc.)
        if let Ok(env_dir) = std::env::var("SOMA_DATA_DIR") {
            return std::path::PathBuf::from(env_dir);
        }

        let dir = &self.soma.data_dir;

        // On Windows, redirect Linux-style XDG paths to LOCALAPPDATA
        #[cfg(target_os = "windows")]
        if dir == "~/.local/share/soma" {
            if let Ok(appdata) = std::env::var("LOCALAPPDATA") {
                return std::path::PathBuf::from(appdata).join("soma");
            }
        }

        if dir.starts_with('~') {
            if let Some(home) = dirs_home() {
                return home.join(dir.trim_start_matches("~/"));
            }
        }
        std::path::PathBuf::from(dir)
    }
}

fn dirs_home() -> Option<std::path::PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .ok()
            .map(std::path::PathBuf::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME").ok().map(std::path::PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = SomaConfig::default();
        assert_eq!(cfg.hdc.dimension, 10_000);
        assert_eq!(cfg.bio.prune_threshold, 0.05);
        assert_eq!(cfg.mcp.tcp_port, 3333);
        assert!(!cfg.llm.enabled);
    }

    #[test]
    fn validate_default_ok() {
        let cfg = SomaConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_zero_dimension() {
        let mut cfg = SomaConfig::default();
        cfg.hdc.dimension = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_overlap_ge_chunk() {
        let mut cfg = SomaConfig::default();
        cfg.ingest.chunk_overlap = 10;
        cfg.ingest.chunk_size = 5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_prune_out_of_range() {
        let mut cfg = SomaConfig::default();
        cfg.bio.prune_threshold = 1.5;
        assert!(cfg.validate().is_err());
    }
}
