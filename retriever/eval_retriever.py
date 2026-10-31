#!/usr/bin/env python3
"""Evaluate first stage + rerankers on ALL dev questions. Writes eval_results.json."""
import argparse, json, os, subprocess, sys, time
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retriever as R

TYPES = ["single-session-user", "single-session-assistant", "knowledge-update", "temporal-reasoning", "multi-session"]


def f1(chosen, gold):
    hit = len(set(chosen) & set(gold))
    if not hit:
        return 0.0
    p, r = hit / len(chosen), hit / len(gold)
    return 2 * p * r / (p + r)


def qmetrics(rank, gold, k_guess, k_gold):
    g = set(gold)
    return {"R@3": len(g & set(rank[:3])) / len(g), "R@5": len(g & set(rank[:5])) / len(g),
            "R@20": len(g & set(rank[:20])) / len(g),
            "MRR": next((1 / (i + 1) for i, s in enumerate(rank) if s in g), 0.0),
            "R@k": len(g & set(rank[:k_guess])) / len(g),
            "F1@k": f1(rank[:k_guess], gold), "F1@k_gold": f1(rank[:k_gold], gold)}


def agg(rows):
    out = {}
    groups = [("overall", rows)] + [(t, [r for r in rows if r["type"] == t]) for t in TYPES]
    for name, rs in groups:
        if rs:
            out[name] = {k: round(float(np.mean([r["m"][k] for r in rs])), 4) for k in rs[0]["m"]} | {"n": len(rs)}
    return out


def gpu_used_mib():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(R.DATA_ROOT, "dev"))
    ap.add_argument("--cache", default=os.path.join(R.HERE, "cache", "dev"))
    ap.add_argument("--rerankers", default="base,v2-m3")
    ap.add_argument("--modes", default="best1,best2")
    ap.add_argument("--out", default=os.path.join(R.HERE, "eval_results.json"))
    a = ap.parse_args()
    import torch
    assert torch.cuda.is_available(), "CUDA required"
    T0 = time.perf_counter()
    timing, vram = {}, {}

    qs = R.load_jsonl(os.path.join(a.data, "questions.jsonl"))
    gold = {g["question_id"]: g for g in R.load_jsonl(os.path.join(a.data, "gold.jsonl"))}
    hs = R.load_histories(os.path.join(a.data, "histories.jsonl"), {q["history_id"] for q in qs})
    timing["load_s"] = round(time.perf_counter() - T0, 1)
    print(f"questions={len(qs)} histories={len(hs)} sessions={sum(len(h['sessions']) for h in hs.values())}", flush=True)

    # --- type guesser
    conf = Counter()
    for q in qs:
        conf[(gold[q["question_id"]]["question_type"], R.guess_type(q["question"]))] += 1
    tg = {"accuracy": round(sum(v for (g, p), v in conf.items() if g == p) / len(qs), 4),
          "per_type_recall": {t: round(conf[(t, t)] / max(1, sum(v for (g, _), v in conf.items() if g == t)), 4) for t in TYPES},
          "confusion": {g: {p: conf[(g, p)] for p in TYPES} for g in TYPES}}
    print("type guesser acc", tg["accuracy"], tg["per_type_recall"], flush=True)

    # --- index
    torch.cuda.reset_peak_memory_stats()
    emb = R.Embedder()
    t = time.perf_counter()
    idx = R.build_indexes(hs, emb, a.cache, log=lambda s: print(s, flush=True))
    torch.cuda.synchronize()
    timing["index_s"] = round(time.perf_counter() - t, 1)
    timing["n_chunks_B"] = int(sum(len(i.B_emb) for i in idx.values()))
    timing["n_chunks_F"] = int(sum(len(i.F_emb) for i in idx.values()))
    vram["embedder_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    vram["embedder_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)

    # --- first stage
    t = time.perf_counter()
    Q = emb.encode_queries([q["question"] for q in qs])
    cands = [R.first_stage(idx[q["history_id"]], Q[i], q["question"]) for i, q in enumerate(qs)]
    timing["first_stage_s"] = round(time.perf_counter() - t, 1)
    print(f"first stage done {timing['first_stage_s']}s", flush=True)
    del emb; torch.cuda.empty_cache()

    # --- rerank scores
    rr_scores = {}
    for name in [x for x in a.rerankers.split(",") if x]:
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        rr = R.Reranker(name)
        load_s = time.perf_counter() - t
        for mode in a.modes.split(","):
            qq, pp, spans = [], [], []
            for q, cs in zip(qs, cands):
                ix = idx[q["history_id"]]
                spans.append((len(qq), len(qq) + len(cs)))
                for c in cs:
                    qq.append(q["question"]); pp.append(R.rerank_text(ix, c, mode))
            torch.cuda.synchronize(); t = time.perf_counter()
            try:
                sc = rr.score(qq, pp)
            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM {name} {mode}: {e}", flush=True); timing[f"{name}/{mode}"] = "OOM"; continue
            torch.cuda.synchronize(); dt = time.perf_counter() - t
            rr_scores[(name, mode)] = [sc[x:y] for x, y in spans]
            timing[f"rerank_{name}_{mode}_s"] = round(dt, 1)
            timing[f"rerank_{name}_{mode}_pairs_per_s"] = round(len(qq) / dt, 1)
            print(f"rerank {name} {mode}: {len(qq)} pairs in {dt:.1f}s (tok budget {rr.tok_budget}), smi={gpu_used_mib()} MiB", flush=True)
        vram[f"{name}_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        vram[f"{name}_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
        timing[f"rerank_{name}_load_s"] = round(load_s, 1)
        del rr; torch.cuda.empty_cache()

    # --- configs
    configs = {"rrf": None}
    for (name, mode) in rr_scores:
        for fusion in ("rerank", "fused"):
            configs[f"{name}|{mode}|{fusion}"] = (name, mode, fusion)
    results = {}
    for cname, cfg in configs.items():
        rows = []
        for i, q in enumerate(qs):
            g = gold[q["question_id"]]
            if g["abstain"]:
                continue
            sc = None if cfg is None else rr_scores[(cfg[0], cfg[1])][i]
            order = [c["sid"] for c in R.final_order(cands[i], sc, cfg[2] if cfg else None)]
            rows.append({"type": g["question_type"],
                         "m": qmetrics(order, g["evidence"], R.topk_for(R.guess_type(q["question"])), R.topk_for(g["question_type"]))})
        results[cname] = agg(rows)
        o = results[cname]["overall"]
        print(f"{cname:28s} R@3={o['R@3']:.3f} R@5={o['R@5']:.3f} MRR={o['MRR']:.3f} F1@k={o['F1@k']:.3f} R@20={o['R@20']:.3f}", flush=True)
    timing["wall_s"] = round(time.perf_counter() - T0, 1)
    meta = {"n_questions": len(qs), "n_answerable": sum(not g["abstain"] for g in gold.values()),
            "n_histories": len(hs), "gpu": torch.cuda.get_device_name(0), "topk": R.TOPK, "n_cand": R.N_CAND}
    json.dump({"meta": meta, "timing": timing, "vram": vram, "type_guesser": tg, "results": results},
              open(a.out, "w"), indent=2)
    print("wall", timing["wall_s"], "->", a.out, flush=True)


if __name__ == "__main__":
    main()
