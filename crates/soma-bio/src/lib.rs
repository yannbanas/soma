//! # soma-bio — Biological Scheduler
//!
//! 4 independent tokio tasks mimicking biological memory processes:
//! 1. Evaporation watchdog (1h) — marks dead edges
//! 2. Physarum reshape (2h) — reinforces frequent paths, weakens unused
//! 3. Sleep consolidation (6h) — clusters episodic → concept nodes
//! 4. Daily pruning (24h) — removes dead edges, archives orphans

mod scheduler;

pub use scheduler::{BioConfig, BioScheduler};
