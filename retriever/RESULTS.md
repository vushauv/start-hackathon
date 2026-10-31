# Retriever + reranker results (dev, all questions)

GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), fp16. 526 dev questions over 470 histories; recall/MRR/F1 on the 496 answerable ones (abstain excluded). 115025 B chunks + 292267 F chunks. 20 candidates per question. Per-type k = {'single-session-user': 3, 'single-session-assistant': 3, 'knowledge-update': 4, 'temporal-reasoning': 4, 'multi-session': 6}; F1@k uses the k of the *guessed* type, F1@k(gold type) the k of the true type.

**Chosen default: `base|best2|fused` = RRF + bge-reranker-base (top-2 chunks, rerank fused w/ RRF).**

## Overall

| Config | R@3 | R@5 | MRR | R@k | F1@k | F1@k (gold type) | R@20 (ceiling) |
|---|---|---|---|---|---|---|---|
| RRF only (B+F+BM25) | 0.911 | 0.960 | 0.948 | 0.949 | 0.558 | 0.562 | 0.998 |
| RRF + bge-reranker-base (best chunk, rerank score only) | 0.912 | 0.959 | 0.958 | 0.948 | 0.556 | 0.561 | 0.998 |
| RRF + bge-reranker-base (best chunk, rerank fused w/ RRF) | 0.928 | 0.971 | 0.960 | 0.961 | 0.565 | 0.569 | 0.998 |
| RRF + bge-reranker-base (top-2 chunks, rerank score only) | 0.928 | 0.964 | 0.964 | 0.950 | 0.557 | 0.562 | 0.998 |
| **RRF + bge-reranker-base (top-2 chunks, rerank fused w/ RRF)** | 0.934 | 0.972 | 0.965 | 0.964 | 0.567 | 0.569 | 0.998 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank score only) | 0.919 | 0.961 | 0.963 | 0.949 | 0.556 | 0.561 | 0.998 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank fused w/ RRF) | 0.930 | 0.970 | 0.966 | 0.961 | 0.566 | 0.567 | 0.998 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank score only) | 0.936 | 0.967 | 0.979 | 0.958 | 0.562 | 0.568 | 0.998 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank fused w/ RRF) | 0.933 | 0.972 | 0.969 | 0.963 | 0.567 | 0.570 | 0.998 |

## R@3 per question type (gold type)

| Config | single-session-user (n=64) | single-session-assistant (n=112) | knowledge-update (n=72) | temporal-reasoning (n=127) | multi-session (n=121) |
|---|---|---|---|---|---|
| RRF only (B+F+BM25) | 0.984 | 0.982 | 0.993 | 0.833 | 0.839 |
| RRF + bge-reranker-base (best chunk, rerank score only) | 1.000 | 1.000 | 0.993 | 0.811 | 0.842 |
| RRF + bge-reranker-base (best chunk, rerank fused w/ RRF) | 0.984 | 1.000 | 1.000 | 0.867 | 0.855 |
| RRF + bge-reranker-base (top-2 chunks, rerank score only) | 1.000 | 1.000 | 0.986 | 0.846 | 0.875 |
| RRF + bge-reranker-base (top-2 chunks, rerank fused w/ RRF) | 1.000 | 1.000 | 0.993 | 0.869 | 0.870 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank score only) | 0.984 | 1.000 | 0.993 | 0.852 | 0.836 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank fused w/ RRF) | 1.000 | 1.000 | 0.993 | 0.864 | 0.861 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank score only) | 1.000 | 1.000 | 0.993 | 0.887 | 0.860 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank fused w/ RRF) | 1.000 | 1.000 | 1.000 | 0.874 | 0.858 |

## R@5 per question type (gold type)

| Config | single-session-user (n=64) | single-session-assistant (n=112) | knowledge-update (n=72) | temporal-reasoning (n=127) | multi-session (n=121) |
|---|---|---|---|---|---|
| RRF only (B+F+BM25) | 0.984 | 1.000 | 0.993 | 0.913 | 0.938 |
| RRF + bge-reranker-base (best chunk, rerank score only) | 1.000 | 1.000 | 1.000 | 0.920 | 0.916 |
| RRF + bge-reranker-base (best chunk, rerank fused w/ RRF) | 1.000 | 1.000 | 1.000 | 0.921 | 0.964 |
| RRF + bge-reranker-base (top-2 chunks, rerank score only) | 1.000 | 1.000 | 1.000 | 0.923 | 0.935 |
| RRF + bge-reranker-base (top-2 chunks, rerank fused w/ RRF) | 1.000 | 1.000 | 1.000 | 0.922 | 0.968 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank score only) | 1.000 | 1.000 | 1.000 | 0.916 | 0.929 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank fused w/ RRF) | 1.000 | 1.000 | 1.000 | 0.916 | 0.963 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank score only) | 1.000 | 1.000 | 1.000 | 0.938 | 0.928 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank fused w/ RRF) | 1.000 | 1.000 | 1.000 | 0.930 | 0.957 |

## MRR per question type (gold type)

| Config | single-session-user (n=64) | single-session-assistant (n=112) | knowledge-update (n=72) | temporal-reasoning (n=127) | multi-session (n=121) |
|---|---|---|---|---|---|
| RRF only (B+F+BM25) | 0.960 | 0.976 | 0.986 | 0.888 | 0.955 |
| RRF + bge-reranker-base (best chunk, rerank score only) | 0.958 | 1.000 | 0.993 | 0.897 | 0.960 |
| RRF + bge-reranker-base (best chunk, rerank fused w/ RRF) | 0.973 | 0.996 | 0.993 | 0.897 | 0.966 |
| RRF + bge-reranker-base (top-2 chunks, rerank score only) | 0.992 | 1.000 | 0.986 | 0.910 | 0.961 |
| RRF + bge-reranker-base (top-2 chunks, rerank fused w/ RRF) | 0.982 | 0.996 | 0.993 | 0.912 | 0.968 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank score only) | 0.980 | 1.000 | 0.993 | 0.920 | 0.948 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank fused w/ RRF) | 0.964 | 0.996 | 0.993 | 0.910 | 0.982 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank score only) | 1.000 | 1.000 | 1.000 | 0.949 | 0.968 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank fused w/ RRF) | 0.982 | 1.000 | 0.993 | 0.923 | 0.966 |

## R@k per question type (gold type)

| Config | single-session-user (n=64) | single-session-assistant (n=112) | knowledge-update (n=72) | temporal-reasoning (n=127) | multi-session (n=121) |
|---|---|---|---|---|---|
| RRF only (B+F+BM25) | 0.984 | 0.982 | 0.993 | 0.888 | 0.938 |
| RRF + bge-reranker-base (best chunk, rerank score only) | 1.000 | 1.000 | 0.993 | 0.862 | 0.935 |
| RRF + bge-reranker-base (best chunk, rerank fused w/ RRF) | 0.984 | 1.000 | 1.000 | 0.897 | 0.955 |
| RRF + bge-reranker-base (top-2 chunks, rerank score only) | 1.000 | 1.000 | 0.986 | 0.872 | 0.939 |
| RRF + bge-reranker-base (top-2 chunks, rerank fused w/ RRF) | 1.000 | 1.000 | 0.993 | 0.897 | 0.964 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank score only) | 0.984 | 1.000 | 0.993 | 0.881 | 0.926 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank fused w/ RRF) | 1.000 | 1.000 | 1.000 | 0.893 | 0.954 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank score only) | 1.000 | 1.000 | 0.993 | 0.908 | 0.929 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank fused w/ RRF) | 1.000 | 1.000 | 1.000 | 0.896 | 0.958 |

## F1@k per question type (gold type)

| Config | single-session-user (n=64) | single-session-assistant (n=112) | knowledge-update (n=72) | temporal-reasoning (n=127) | multi-session (n=121) |
|---|---|---|---|---|---|
| RRF only (B+F+BM25) | 0.436 | 0.478 | 0.650 | 0.595 | 0.603 |
| RRF + bge-reranker-base (best chunk, rerank score only) | 0.444 | 0.487 | 0.649 | 0.583 | 0.598 |
| RRF + bge-reranker-base (best chunk, rerank fused w/ RRF) | 0.436 | 0.487 | 0.655 | 0.603 | 0.615 |
| RRF + bge-reranker-base (top-2 chunks, rerank score only) | 0.444 | 0.487 | 0.644 | 0.585 | 0.601 |
| RRF + bge-reranker-base (top-2 chunks, rerank fused w/ RRF) | 0.444 | 0.487 | 0.650 | 0.603 | 0.620 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank score only) | 0.436 | 0.487 | 0.649 | 0.593 | 0.591 |
| RRF + bge-reranker-v2-m3 (best chunk, rerank fused w/ RRF) | 0.444 | 0.487 | 0.655 | 0.600 | 0.614 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank score only) | 0.444 | 0.487 | 0.649 | 0.608 | 0.595 |
| RRF + bge-reranker-v2-m3 (top-2 chunks, rerank fused w/ RRF) | 0.444 | 0.487 | 0.655 | 0.601 | 0.617 |

## Runtime and VRAM

| Step | Time | Throughput | Peak VRAM (torch alloc / reserved) |
|---|---|---|---|
| Index dev (bge-small fp16, 407292 chunks, cold) | 785.5 s | 519 chunks/s | 1289 / 2020 MiB |
| First stage, 526 queries (dense B+F, BM25, RRF) | 25.9 s | | |
| Rerank bge-reranker-base, best1, 10520 pairs | 60.9 s | 172.8 pairs/s | 810 / 922 MiB |
| Rerank bge-reranker-base, best2, 10520 pairs | 118.4 s | 88.8 pairs/s | 810 / 922 MiB |
| Rerank bge-reranker-v2-m3, best1, 10520 pairs | 204.2 s | 51.5 pairs/s | 1263 / 1318 MiB |
| Rerank bge-reranker-v2-m3, best2, 10520 pairs | 359.9 s | 29.2 pairs/s | 1263 / 1318 MiB |
| Whole eval_retriever.py run | 1569.3 s | | |

Neither reranker hit OOM. Batches are sized by a token budget: 16384 tokens for base and 8192 for v2-m3, both sorted by length.

## Type guesser (rule-based, keywords)

Accuracy on dev vs gold question_type: **0.641** (526 questions). The spec's keyword order, checked literally (multi, then temporal, then KU, then SSA), scored 0.433. The shipped order (SSA, then temporal, then KU, then multi, else SSU) with wider keyword lists scores the figure above.

| Gold \ guessed | single-session-user | single-session-assistant | knowledge-update | temporal-reasoning | multi-session | recall |
|---|---|---|---|---|---|---|
| single-session-user | 48 | 0 | 3 | 5 | 14 | 0.686 |
| single-session-assistant | 49 | 54 | 0 | 4 | 5 | 0.482 |
| knowledge-update | 23 | 0 | 23 | 8 | 24 | 0.295 |
| temporal-reasoning | 11 | 0 | 2 | 112 | 8 | 0.842 |
| multi-session | 16 | 0 | 6 | 11 | 100 | 0.752 |

The guessed type is only used to pick k, and SSU/SSA share k=3 while KU/temporal share k=4. So the guess gives the right k for 396/526 = 0.753 of questions. The cost is small: F1@k is 0.567 with guessed types and 0.569 with gold types for the chosen config.

## Why bge-reranker-base (top-2 chunks, fused with RRF) is the default

- **The rerankers help, but only when fused with RRF.** Fused base/best2 against RRF only: R@3 0.911 → 0.934, R@5 0.960 → 0.972, MRR 0.948 → 0.965, R@k 0.949 → 0.964, F1@k 0.558 → 0.567. Multi-session R@k went from 0.938 to 0.964.
- **Rerank score alone is weaker than fused** for both models at R@5 and R@k. For example, v2-m3/best2 scored R@k 0.958 alone and 0.963 fused. It does have the best MRR, at 0.979, and the best temporal R@3/R@5, at 0.887/0.938.
- **v2-m3 and base tie on the fused config.** v2-m3/best2/fused scored R@3 0.933, R@5 0.972, R@k 0.963 and F1@k 0.567. Base/best2/fused scored R@3 0.934, R@5 0.972, R@k 0.964 and F1@k 0.567. One question is 0.002 overall, so the two are identical within noise.
- **v2-m3 costs 3x the time:** 360 s against 118 s for 10.5k pairs, and 1.3 GB against 0.9 GB of VRAM. Base wins on cost at the same quality.
- **Top-2 chunks beat a single best chunk** for both models: base fused R@3 0.934 against 0.928.
- To switch, run `--reranker v2-m3`. On test that is about 2,920 pairs, so roughly 100 s instead of about 20 s.

## Test split (build_evidence.py --split test)

146 questions over 15 histories with 200 sessions each, which is 48,315 chunks. A cold index took 82 s. Retrieval plus rerank took 18.7 s, and the whole job took 110 s. Peak VRAM allocated was 1.19 GB. A warm run from cache takes 29 s. Evidence is at most 13,773 characters per question, with a mean of 7,040, because test sessions are short. Guessed types were 42 SSU, 33 multi-session, 26 temporal, 25 SSA and 20 KU.
