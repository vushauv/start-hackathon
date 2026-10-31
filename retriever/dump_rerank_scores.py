#!/usr/bin/env python3
"""Dump reranker relevance scores for every candidate of every question, with gold labels,
for tuning a reranker score threshold.

usage: python dump_rerank_scores.py --data DIR [--out FILE] [--rerank-text best2|best1] [--device cuda|cpu]
DIR must contain questions.jsonl, histories.jsonl, gold.jsonl.
Output: one JSON line per question:
  {question_id, question, question_type, guessed_type, abstain, gold_sessions,
   candidates: [{sid, rrf_rank, rerank_score, final_rank, is_gold}]}
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retriever as R
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--rerank-text", default=R.DEFAULT_CONFIG["rerank_text"])
ap.add_argument("--reranker", default=R.DEFAULT_CONFIG["reranker"])
ap.add_argument("--fusion", default=R.DEFAULT_CONFIG["fusion"])
ap.add_argument("--device", default="cuda")
ap.add_argument("--n-cand", type=int, default=R.N_CAND)
a = ap.parse_args()

d = os.path.abspath(os.path.expanduser(a.data))
out = a.out or os.path.join(d, "rerank_scores.jsonl")
qs = R.load_jsonl(os.path.join(d, "questions.jsonl"))
gold = {g["question_id"]: g for g in R.load_jsonl(os.path.join(d, "gold.jsonl"))} \
    if os.path.exists(os.path.join(d, "gold.jsonl")) else {}
print(f"{len(qs)} questions from {d}  (gold: {len(gold)})", flush=True)

hs = R.load_histories(os.path.join(d, "histories.jsonl"), {q["history_id"] for q in qs})
emb = R.Embedder(device=a.device)
cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", os.path.basename(d))
idx = R.build_indexes(hs, emb, cache, log=lambda s: print(s, flush=True))
rr = R.Reranker(a.reranker, device=a.device)

done = set()
if os.path.exists(out):
    done = {json.loads(l)["question_id"] for l in open(out) if l.strip()}
    print(f"resuming: {len(done)} already in {out}", flush=True)

t0 = time.perf_counter(); hits = 0; n = 0
with open(out, "a") as fo:
    bar = tqdm([q for q in qs if q["question_id"] not in done], unit="q", dynamic_ncols=True)
    for q in bar:
        ix = idx[q["history_id"]]
        qv = emb.encode_queries([q["question"]])[0]
        cands = R.first_stage(ix, qv, q["question"], n_cand=a.n_cand)
        texts = [R.rerank_text(ix, c, a.rerank_text) for c in cands]
        sc = rr.score([q["question"]] * len(cands), texts) if cands else []
        fin = R.final_order(cands, sc, a.fusion)
        frank = {c["sid"]: i for i, c in enumerate(fin)}
        g = gold.get(q["question_id"], {})
        gs = set(g.get("evidence", []))
        rows = [{"sid": c["sid"], "rrf_rank": i, "rerank_score": float(sc[i]),
                 "final_rank": frank[c["sid"]], "is_gold": c["sid"] in gs}
                for i, c in enumerate(cands)]
        fo.write(json.dumps({"question_id": q["question_id"], "question": q["question"],
                             "question_type": g.get("question_type"), "guessed_type": R.guess_type(q["question"]),
                             "abstain": g.get("abstain"), "gold_sessions": sorted(gs),
                             "candidates": rows}) + "\n")
        fo.flush()
        n += 1
        if gs and any(r["is_gold"] for r in rows if r["final_rank"] < 5):
            hits += 1
        bar.set_postfix(gold_in_top5=f"{hits}/{n}")
print(f"done in {time.perf_counter() - t0:.0f}s -> {out}", flush=True)
