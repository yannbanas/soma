// Reserved for mmap binary storage:
//
// ~/.local/share/soma/<workspace>/hdc/
// ├── vectors.bin       # f32 vectors — mmap, O(1) access
// ├── index.bin         # label_hash(8) + offset(8) + len(4) per entry
// └── vocab.txt         # token list (debug + audit)
//
// Format vectors.bin:
//   [magic: b"SOMA" (4)]
//   [version: u16 (2)]
//   [dim: u32 (4)]
//   [count: u64 (8)]
//   [data: f32 × dim × count]
//
// Will be implemented in Phase 2 for production-scale workloads.
