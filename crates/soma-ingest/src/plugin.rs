//! Plugin trait for custom ingestion formats.
//!
//! Implement `IngestPlugin` to add support for new file formats
//! or data sources without modifying the core pipeline.

use soma_core::SomaError;
use soma_graph::StigreGraph;

/// Result of a plugin ingestion.
#[derive(Debug, Default)]
pub struct PluginResult {
    pub nodes_created: usize,
    pub edges_created: usize,
    pub source: String,
}

/// Trait for custom ingestion plugins.
///
/// # Example
///
/// ```ignore
/// struct CsvPlugin;
///
/// impl IngestPlugin for CsvPlugin {
///     fn name(&self) -> &str { "csv" }
///     fn extensions(&self) -> &[&str] { &["csv", "tsv"] }
///     fn ingest(&self, content: &str, graph: &mut StigreGraph, source: &str)
///         -> Result<PluginResult, SomaError> {
///         // parse CSV, insert nodes/edges
///         Ok(PluginResult::default())
///     }
/// }
/// ```
pub trait IngestPlugin: Send + Sync {
    /// Plugin name (e.g., "csv", "bibtex", "yaml").
    fn name(&self) -> &str;

    /// File extensions this plugin handles (e.g., ["csv", "tsv"]).
    fn extensions(&self) -> &[&str];

    /// Ingest content into the graph.
    fn ingest(
        &self,
        content: &str,
        graph: &mut StigreGraph,
        source: &str,
    ) -> Result<PluginResult, SomaError>;
}

/// Registry of ingestion plugins.
pub struct PluginRegistry {
    plugins: Vec<Box<dyn IngestPlugin>>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        PluginRegistry {
            plugins: Vec::new(),
        }
    }

    /// Register a new plugin.
    pub fn register(&mut self, plugin: Box<dyn IngestPlugin>) {
        self.plugins.push(plugin);
    }

    /// Find a plugin by file extension.
    pub fn find_by_extension(&self, ext: &str) -> Option<&dyn IngestPlugin> {
        let ext_lower = ext.to_lowercase();
        self.plugins
            .iter()
            .find(|p| p.extensions().iter().any(|e| e.to_lowercase() == ext_lower))
            .map(|p| p.as_ref())
    }

    /// Find a plugin by name.
    pub fn find_by_name(&self, name: &str) -> Option<&dyn IngestPlugin> {
        self.plugins.iter().find(|p| p.name() == name).map(|p| p.as_ref())
    }

    /// List all registered plugins.
    pub fn list(&self) -> Vec<&str> {
        self.plugins.iter().map(|p| p.name()).collect()
    }

    /// Try to ingest a file using the appropriate plugin.
    pub fn ingest_file(
        &self,
        path: &std::path::Path,
        graph: &mut StigreGraph,
    ) -> Result<Option<PluginResult>, SomaError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        if let Some(plugin) = self.find_by_extension(ext) {
            let content = std::fs::read_to_string(path)
                .map_err(|e| SomaError::Ingest(format!("read error: {}", e)))?;
            let source = format!("plugin:{}", plugin.name());
            Ok(Some(plugin.ingest(&content, graph, &source)?))
        } else {
            Ok(None)
        }
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyPlugin;

    impl IngestPlugin for DummyPlugin {
        fn name(&self) -> &str { "dummy" }
        fn extensions(&self) -> &[&str] { &["dum", "dummy"] }
        fn ingest(
            &self,
            _content: &str,
            _graph: &mut StigreGraph,
            _source: &str,
        ) -> Result<PluginResult, SomaError> {
            Ok(PluginResult {
                nodes_created: 1,
                edges_created: 0,
                source: "dummy".to_string(),
            })
        }
    }

    #[test]
    fn registry_find_by_extension() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(DummyPlugin));

        assert!(reg.find_by_extension("dum").is_some());
        assert!(reg.find_by_extension("dummy").is_some());
        assert!(reg.find_by_extension("txt").is_none());
    }

    #[test]
    fn registry_find_by_name() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(DummyPlugin));

        assert!(reg.find_by_name("dummy").is_some());
        assert!(reg.find_by_name("csv").is_none());
    }

    #[test]
    fn registry_list_plugins() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(DummyPlugin));

        let list = reg.list();
        assert_eq!(list, vec!["dummy"]);
    }
}
