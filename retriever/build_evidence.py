#!/usr/bin/env python3
"""Produce per-question evidence for a whole split.

  python build_evidence.py --split dev|test [--out evidence_{split}.jsonl]
      [--reranker v2-m3|base|none] [--rerank-text best1|best2] [--fusion fused|rerank]

Output lines: {question_id, history_id, question, question_date, guessed_type,
               evidence: [{id, date, text}], hint, candidates}
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retriever as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--data", help="split dir (default ~/reactor-test/labeler/data/<split>)")
    ap.add_argument("--out")
    ap.add_argument("--reranker", default=R.DEFAULT_CONFIG["reranker"])
    ap.add_argument("--rerank-text", default=R.DEFAULT_CONFIG["rerank_text"], choices=["best1", "best2"])
    ap.add_argument("--fusion", default=R.DEFAULT_CONFIG["fusion"], choices=["fused", "rerank"])
    a = ap.parse_args()
    import torch
    data = a.data or os.path.join(R.DATA_ROOT, a.split)
    out = a.out or os.path.join(R.HERE, f"evidence_{a.split}.jsonl")
    T0 = time.perf_counter()
    rr = None if a.reranker in ("none", "") else a.reranker
    ret = R.Retriever(data, os.path.join(R.HERE, "cache", a.split), reranker=rr,
                      rerank_text_mode=a.rerank_text, fusion=a.fusion,
                      log=lambda s: print(s, flush=True))
    print(f"indexed {len(ret.idx)} histories in {ret.index_s:.1f}s", flush=True)
    t = time.perf_counter()
    rows = ret.run()
    run_s = time.perf_counter() - t
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = [sum(len(e["text"]) for e in r["evidence"]) for r in rows]
    print(f"wrote {len(rows)} questions -> {out}; retrieve+rerank {run_s:.1f}s; total {time.perf_counter() - T0:.1f}s; "
          f"evidence chars max={max(chars)} mean={sum(chars) / len(chars):.0f}; "
          f"peak VRAM alloc={torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
