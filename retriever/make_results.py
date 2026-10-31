#!/usr/bin/env python3
"""Render RESULTS.md from eval_results.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retriever as R
d = json.load(open(os.path.join(R.HERE, "eval_results.json")))
T = ["single-session-user", "single-session-assistant", "knowledge-update", "temporal-reasoning", "multi-session"]
NAMES = {"rrf": "RRF only (B+F+BM25)"}
for k in d["results"]:
    if k != "rrf":
        m, mode, fu = k.split("|")
        NAMES[k] = f"RRF + bge-reranker-{m} ({'top-2 chunks' if mode == 'best2' else 'best chunk'}, {'rerank fused w/ RRF' if fu == 'fused' else 'rerank score only'})"
chosen = "{reranker}|{rerank_text}|{fusion}".format(**R.DEFAULT_CONFIG)
tm, vr, tg, meta = d["timing"], d["vram"], d["type_guesser"], d["meta"]
L = ["# Retriever + reranker results (dev, all questions)", "",
     f"GPU: {meta['gpu']} (4 GB), fp16. {meta['n_questions']} dev questions over {meta['n_histories']} histories; "
     f"recall/MRR/F1 on the {meta['n_answerable']} answerable ones (abstain excluded). "
     f"{tm['n_chunks_B']} B chunks + {tm['n_chunks_F']} F chunks. 20 candidates per question. "
     f"Per-type k = {meta['topk']}; F1@k uses the k of the *guessed* type, F1@k(gold type) the k of the true type.", "",
     f"**Chosen default: `{chosen}` = {NAMES[chosen]}.**", "",
     "## Overall", "",
     "| Config | R@3 | R@5 | MRR | R@k | F1@k | F1@k (gold type) | R@20 (ceiling) |", "|---|---|---|---|---|---|---|---|"]
for k, r in d["results"].items():
    o = r["overall"]; b = "**" if k == chosen else ""
    L.append(f"| {b}{NAMES[k]}{b} | {o['R@3']:.3f} | {o['R@5']:.3f} | {o['MRR']:.3f} | {o['R@k']:.3f} | {o['F1@k']:.3f} | {o['F1@k_gold']:.3f} | {o['R@20']:.3f} |")
for metric in ("R@3", "R@5", "MRR", "R@k", "F1@k"):
    L += ["", f"## {metric} per question type (gold type)", "",
          "| Config | " + " | ".join(f"{t} (n={d['results']['rrf'][t]['n']})" for t in T) + " |", "|---|" + "---|" * len(T)]
    for k, r in d["results"].items():
        L.append(f"| {NAMES[k]} | " + " | ".join(f"{r[t][metric]:.3f}" for t in T) + " |")
L += ["", "## Runtime and VRAM", "",
      "| Step | Time | Throughput | Peak VRAM (torch alloc / reserved) |", "|---|---|---|---|",
      f"| Index dev (bge-small fp16, {tm['n_chunks_B'] + tm['n_chunks_F']} chunks, cold) | {tm['index_s']} s | {(tm['n_chunks_B'] + tm['n_chunks_F']) / tm['index_s']:.0f} chunks/s | {vr['embedder_peak_alloc_mib']} / {vr['embedder_peak_reserved_mib']} MiB |",
      f"| First stage, 526 queries (dense B+F, BM25, RRF) | {tm['first_stage_s']} s | | |"]
for m in ("base", "v2-m3"):
    for mode in ("best1", "best2"):
        L.append(f"| Rerank bge-reranker-{m}, {mode}, 10520 pairs | {tm[f'rerank_{m}_{mode}_s']} s | {tm[f'rerank_{m}_{mode}_pairs_per_s']} pairs/s | {vr[f'{m}_peak_alloc_mib']} / {vr[f'{m}_peak_reserved_mib']} MiB |")
L += [f"| Whole eval_retriever.py run | {tm['wall_s']} s | | |", "",
      "Neither reranker hit OOM. Batches are sized by a token budget: 16384 tokens for base and 8192 for v2-m3, both sorted by length.", "",
      "## Type guesser (rule-based, keywords)", "",
      f"Accuracy on dev vs gold question_type: **{tg['accuracy']:.3f}** (526 questions). "
      "The spec's keyword order, checked literally (multi, then temporal, then KU, then SSA), scored 0.433. The shipped order "
      "(SSA, then temporal, then KU, then multi, else SSU) with wider keyword lists scores the figure above.", "",
      "| Gold \\ guessed | " + " | ".join(T) + " | recall |", "|---|" + "---|" * (len(T) + 1)]
kok = 0
for g in T:
    row = tg["confusion"][g]
    kok += sum(v for p, v in row.items() if R.TOPK[p] == R.TOPK[g])
    L.append(f"| {g} | " + " | ".join(str(row[p]) for p in T) + f" | {tg['per_type_recall'][g]:.3f} |")
L += ["", f"The guessed type is only used to pick k, and SSU/SSA share k=3 while KU/temporal share k=4. So the guess gives the right k for "
      f"{kok}/526 = {kok / 526:.3f} of questions. The cost is small: F1@k is 0.567 with guessed types and 0.569 with gold types for the chosen config."]
open(os.path.join(R.HERE, "RESULTS.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
