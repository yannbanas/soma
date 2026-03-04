# Benchmark Datasets

SOMA benchmarks use standard QA datasets. They are **not** included in this repository due to size. Download them manually:

## MuSiQue-Ans

Multi-hop QA dataset (2-4 hops). ~25K questions.

```bash
# From HuggingFace
wget https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl

# Or from the original repo
git clone https://github.com/StonyBrookNLP/musique
```

Place the file at: `data/musique_ans_v1.0_dev.jsonl`

## HotpotQA (Distractor Setting)

Multi-hop QA dataset (2 hops). ~7.4K dev questions.

```bash
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

Place the file at: `data/hotpot_dev_distractor_v1.json`

## Running Benchmarks

```bash
# MuSiQue (first 100 questions)
cargo run -p soma-bench --example musique -- --path data/musique_ans_v1.0_dev.jsonl --limit 100

# HotpotQA (first 100 questions)
cargo run -p soma-bench --example hotpotqa -- --path data/hotpot_dev_distractor_v1.json --limit 100

# Temporal benchmark (no download needed — synthetic)
cargo test -p soma-bench temporal
```
