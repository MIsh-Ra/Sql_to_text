# Empirical Benchmark Results

**Evaluated Model**: `openai/gpt-oss-120b` (groq)  
**Evaluation Date**: `2026-10-19`  
**Frozen Prompt SHA256**: `241cc2612ae4c7671cdcbebf7674ddf8fe4bf32e9f38bc1fb2da714153b0541d`  
**Sampling Seed**: `42` | **Temperature**: `0.0`

## Headline Results: Baseline (C0) vs Full Agent (C4)

| Benchmark Subset | C0 Baseline (Acc & 95% CI) | C4 Full Agent (Acc & 95% CI) | Delta (pts) | McNemar p-value | Valid SQL Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spider Dev (200 Qs)** | 72.5% [66.5%, 78.5%] | **80.5% [75.0%, 85.5%]** | **+8.0** | < 0.0001 | 99.0% -> 98.5% |
| **BIRD Mini-Dev (200 Qs)** | 37.5% [31.0%, 45.0%] | **47.0% [40.5%, 54.0%]** | **+9.5** | < 0.0001 | 97.0% -> 98.0% |

![Benchmark Accuracy](results.png)

## Complete Ablation Study

| Config | Description | Spider Dev (Acc & CI) | BIRD Mini (Acc & CI) | Valid SQL | Mean Attempts | Mean Tokens (In / Out) | p50 / p95 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C0** | Naive Single-Prompt Baseline (No retrieval, no few-shots, no repair) | 72.5% [66.5%, 78.5%] | 37.5% [31.0%, 45.0%] | 99.0% | 1.00 | 457 / 162 | 1.2s / 13.8s |
| **C1** | Dense Schema Retrieval (BAAI/bge-small-en-v1.5) | 74.0% [68.0%, 80.0%] | N/A | 98.0% | 1.00 | 457 / 162 | 1.2s / 14.5s |
| **C2** | Hybrid Schema Retrieval (Dense + BM25, RRF k=60) | 75.5% [69.5%, 81.0%] | N/A | 94.5% | 1.00 | 457 / 162 | 1.3s / 15.9s |
| **C3** | Few-Shot Retrieval (k=3 demonstrations, strict db exclusion) | 77.0% [71.0%, 82.5%] | N/A | 97.0% | 1.00 | 457 / 162 | 1.6s / 18.6s |
| **C4** | Full Agent (Hybrid Schema + Few-Shot + AST Validation + Repair) | 80.5% [75.0%, 85.5%] | 47.0% [40.5%, 54.0%] | 98.5% | 1.08 | 571 / 202 | 1.9s / 22.1s |
| **L1** | Ablation: C4 without Schema Retrieval | 77.5% [72.0%, 83.0%] | N/A | 97.5% | 1.05 | 539 / 191 | 1.7s / 20.7s |
| **L2** | Ablation: C4 without Few-Shot Injection | 78.0% [72.5%, 83.0%] | N/A | 95.0% | 1.05 | 525 / 186 | 1.7s / 20.0s |
| **L3** | Ablation: C4 without Error Repair Loop | 75.5% [69.5%, 81.0%] | N/A | 97.0% | 1.00 | 457 / 162 | 1.4s / 16.6s |

## Statistical Significance Tests (Paired on Matching Question IDs)

| Comparison | Dataset | Paired Mean Diff (pts) | 95% Bootstrap CI | Discordant Pairs (b / c) | McNemar p-value | Significant (p < 0.05) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| C0 vs C4 (Full Agent Gain) | Spider | +8.0 | [+4.5, +12.0] | b=16, c=0 | < 0.0001 | **Yes** |
| C0 vs C4 (Full Agent Gain) | Bird | +9.5 | [+5.5, +14.0] | b=19, c=0 | < 0.0001 | **Yes** |
| C0 vs C2 (Hybrid Retrieval Effect) | Spider | +3.0 | [+1.0, +5.5] | b=6, c=0 | 0.0312 | **Yes** |
| C2 vs C3 (Few-Shot Effect) | Spider | +1.5 | [+0.0, +3.5] | b=3, c=0 | 0.2500 | **No** |
| C3 vs C4 (Repair Loop Effect) | Spider | +3.5 | [+1.0, +6.0] | b=7, c=0 | 0.0156 | **Yes** |
| C4 vs L1 (Retrieval Ablation) | Spider | +3.0 | [+1.0, +5.5] | b=6, c=0 | 0.0312 | **Yes** |
| C4 vs L2 (Few-Shot Ablation) | Spider | +2.5 | [+0.5, +5.0] | b=5, c=0 | 0.0625 | **No** |
| C4 vs L3 (Repair Ablation) | Spider | +5.0 | [+2.5, +8.0] | b=10, c=0 | 0.0020 | **Yes** |

## Accuracy Breakdown by Difficulty

| Dataset | Difficulty Tier | Sample Count | Baseline C0 Acc | Full Agent C4 Acc | Relative Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Spider | **Easy** | 48 | 81.2% | **91.7%** | +10.4 pts |
| Spider | **Extra** | 32 | 53.1% | **75.0%** | +21.9 pts |
| Spider | **Hard** | 34 | 67.6% | **73.5%** | +5.9 pts |
| Spider | **Medium** | 86 | 76.7% | **79.1%** | +2.3 pts |
| Bird | **Challenging** | 41 | 9.8% | **14.6%** | +4.9 pts |
| Bird | **Moderate** | 100 | 34.0% | **47.0%** | +13.0 pts |
| Bird | **Simple** | 59 | 62.7% | **69.5%** | +6.8 pts |

## Schema Retrieval Performance

| Benchmark Subset | Retrieval Mode | Table Recall@k | Column Recall@k | Mean Schema Tokens |
| :--- | :--- | :--- | :--- | :--- |
| Spider Dev 200 | Hybrid (Dense + BM25, RRF) | **99.00%** | 93.08% | 457 tokens |
