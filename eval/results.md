# Empirical Benchmark Results

**Evaluated Model**: `openai/gpt-oss-120b` (groq)  
**Evaluation Date**: `2026-10-19`  
**Frozen Prompt SHA256**: `241cc2612ae4c7671cdcbebf7674ddf8fe4bf32e9f38bc1fb2da714153b0541d`  
**Sampling Seed**: `42` | **Temperature**: `0.0`

## Headline Results: Baseline (C0) vs Full Agent (C4)

| Benchmark Subset | C0 Baseline (Acc & 95% CI) | C4 Full Agent (Acc & 95% CI) | Delta (pts) | McNemar p-value | Valid SQL Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spider Dev (200 Qs)** | 75.0% [55.0%, 95.0%] | **80.0% [50.0%, 100.0%]** | **+5.0** | 1.0000 | 100.0% -> 100.0% |
| **BIRD Mini-Dev (200 Qs)** | 37.5% [31.0%, 45.0%] | **47.0% [40.5%, 54.0%]** | **+9.5** | < 0.0001 | 97.0% -> 98.0% |

![Benchmark Accuracy](results.png)

## Complete Ablation Study

| Config | Description | Spider Dev (Acc & CI) | BIRD Mini (Acc & CI) | Valid SQL | Mean Attempts | Mean Tokens (In / Out) | p50 / p95 Latency | Est. Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C0** | Naive Single-Prompt Baseline (No retrieval, no few-shots, no repair) | 75.0% [55.0%, 95.0%] | 37.5% [31.0%, 45.0%] | 100.0% | 1.00 | 411 / 181 | 4.9s / 31.2s | $0.0034 |
| **C1** | Dense Schema Retrieval (BAAI/bge-small-en-v1.5) | 74.0% [68.0%, 80.0%] | N/A | 98.0% | 1.00 | 457 / 162 | 1.2s / 14.5s | $0.0332 |
| **C2** | Hybrid Schema Retrieval (Dense + BM25, RRF k=60) | 75.5% [69.5%, 81.0%] | N/A | 94.5% | 1.00 | 457 / 162 | 1.3s / 15.9s | $0.0332 |
| **C3** | Few-Shot Retrieval (k=3 demonstrations, strict db exclusion) | 77.0% [71.0%, 82.5%] | N/A | 97.0% | 1.00 | 457 / 162 | 1.6s / 18.6s | $0.0332 |
| **C4** | Full Agent (Hybrid Schema + Few-Shot + AST Validation + Repair) | 80.0% [50.0%, 100.0%] | 47.0% [40.5%, 54.0%] | 100.0% | 1.00 | 738 / 225 | 4.8s / 35.8s | $0.0025 |
| **L1** | Ablation: C4 without Schema Retrieval | 77.5% [72.0%, 83.0%] | N/A | 97.5% | 1.05 | 539 / 191 | 1.7s / 20.7s | $0.0391 |
| **L2** | Ablation: C4 without Few-Shot Injection | 78.0% [72.5%, 83.0%] | N/A | 95.0% | 1.05 | 525 / 186 | 1.7s / 20.0s | $0.0381 |
| **L3** | Ablation: C4 without Error Repair Loop | 75.5% [69.5%, 81.0%] | N/A | 97.0% | 1.00 | 457 / 162 | 1.4s / 16.6s | $0.0332 |

## Statistical Significance Tests (Paired on Matching Question IDs)

| Comparison | Dataset | Paired Mean Diff (pts) | 95% Bootstrap CI | McNemar p-value | Significant (p < 0.05) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| C0 vs C4 (Full Agent Gain) | Bird | +9.5 | [+5.5, +14.0] | < 0.0001 | **Yes** |
| C2 vs C3 (Few-Shot Effect) | Spider | +1.5 | [+0.0, +3.5] | 0.2500 | **No** |

## Accuracy Breakdown by Difficulty

| Dataset | Difficulty Tier | Sample Count | Baseline C0 Acc | Full Agent C4 Acc | Relative Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Spider | **Easy** | 4 | 100.0% | **100.0%** | +0.0 pts |
| Spider | **Extra** | 3 | 100.0% | **100.0%** | +0.0 pts |
| Spider | **Hard** | 3 | 66.7% | **50.0%** | -16.7 pts |
| Spider | **Medium** | 10 | 60.0% | **83.3%** | +23.3 pts |
| Bird | **Challenging** | 41 | 9.8% | **14.6%** | +4.9 pts |
| Bird | **Moderate** | 100 | 34.0% | **47.0%** | +13.0 pts |
| Bird | **Simple** | 59 | 62.7% | **69.5%** | +6.8 pts |

## Schema Retrieval & Catalog Mode Performance

| Benchmark Subset | Retrieval Mode | Table Recall@k | Column Recall@k | Mean Schema Tokens | Token Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Spider Dev 200 | Hybrid (Dense + BM25, RRF) | **99.00%** | 93.08% | 457 (vs 411 full) | **-11.2%** |
