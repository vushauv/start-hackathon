"""HTTP bridge for the web chat (``web/``): the ``chat`` REPL's steps as JSON instead of rich tables.

    uv run --extra retrieval pipeline serve --split dev [--port 8765] [--device cuda|cpu]

Endpoints (stdlib only, no web framework):
    GET  /api/meta              split, device, models, histories, defaults
    GET  /api/history/<hid>     index a history; its stats, dataset questions and default question date
    POST /api/ask               {history_id, question, question_id?, question_date?, settings{}} ->
                                NDJSON stream: start, step (embedding, first_stage, rerank, selection,
                                evidence, compose), answer, gold, done | error

Per-request ``settings`` mirror the ``chat`` flags: reranker, rerank_text, fusion, rerank_threshold, provider,
model, max_chars, attempts, timeout, dry_run, show_message. Models and Reactor are used one question at a time.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from .answering import GROQ_DEFAULT_MAX_CHARS, GROQ_DEFAULT_MODEL, OPENAI_DEFAULT_MODEL
from .data import Split, load_histories, split_dir
from .models import Evidence, Retrieved
from .prompting import compose
from .retrieval import RETRIEVER_DIR

INDEX_CACHE = 4  # histories kept indexed in memory
CHANNELS = (("dense_B", "Dense · user turns"), ("dense_F", "Dense · pair windows"), ("bm25", "BM25 · pair windows"))
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _log(s: str) -> None:
    print(s, file=sys.stderr, flush=True)


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


class Engine:
    def __init__(self, a):
        self.a = a
        self.dir = split_dir(a.split, a.data)
        self.split = Split(self.dir)
        self.qs_by_id = {q["question_id"]: q for q in self.split.questions}
        self.by_hid: dict[str, list[dict]] = {}
        for q in self.split.questions:
            self.by_hid.setdefault(q["history_id"], []).append(q)
        self.cache = Path(a.cache) if a.cache else RETRIEVER_DIR / "cache" / self.dir.name
        if str(RETRIEVER_DIR) not in sys.path:
            sys.path.insert(0, str(RETRIEVER_DIR))
        device = a.device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        import retriever as R
        self.R = R
        t0 = time.perf_counter()
        self.emb = R.Embedder(device=device)
        if device == "cpu":  # fp16 is slow / partly unsupported on CPU
            self.emb.m.float()
        self.rerankers: dict[str, object] = {}
        self._reranker(a.reranker)
        _log(f"models ready on {device} in {_ms(t0)} ms")
        self.indexes: OrderedDict[str, object] = OrderedDict()
        self.answerers: dict[tuple, object] = {}
        self.reactor_hid_for: dict[tuple, str] = {}
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- resources
    def _reranker(self, name: str):
        if name == "none":
            return None
        if name not in self.rerankers:
            rr = self.R.Reranker(name, device=self.device)
            if self.device == "cpu":
                rr.m.float()
            self.rerankers[name] = rr
        return self.rerankers[name]

    def _index(self, hid: str):
        if hid in self.indexes:
            self.indexes.move_to_end(hid)
            return self.indexes[hid]
        h = load_histories(self.dir / "histories.jsonl", {hid})[hid]
        idx = self.R.build_indexes({hid: h}, self.emb, str(self.cache), log=lambda s: None)[hid]
        self.indexes[hid] = idx
        while len(self.indexes) > INDEX_CACHE:
            self.indexes.popitem(last=False)
        return idx

    def _latest_date(self, idx) -> str:
        dates = [d for d in idx.dates if d]
        return max(dates, key=lambda d: self.R.parse_date(d) or datetime.min.date()) if dates else ""

    # ---------------------------------------------------------------- read endpoints
    def meta(self) -> dict:
        a = self.a
        return {
            "split": a.split, "device": self.device, "has_gold": bool(self.split.gold),
            "embedder": self.R.EMB_MODEL, "rerankers": ["base", "v2-m3", "none"],
            "histories": [{"id": hid, "questions": len(qs), "first_question": qs[0]["question"]}
                          for hid, qs in self.by_hid.items()],
            "defaults": {
                "reranker": a.reranker, "rerank_text": "best2", "fusion": "fused", "rerank_threshold": None,
                "provider": a.provider, "model": a.model or "", "max_chars": a.max_chars,
                "attempts": None, "timeout": None, "dry_run": a.dry_run, "show_message": False,
            },
            "provider_defaults": {
                "reactor": {"model": "", "max_chars": 16000},
                "groq": {"model": GROQ_DEFAULT_MODEL, "max_chars": GROQ_DEFAULT_MAX_CHARS},
                "openai": {"model": OPENAI_DEFAULT_MODEL, "max_chars": GROQ_DEFAULT_MAX_CHARS},
            },
        }

    def history(self, hid: str) -> dict:
        if hid not in self.by_hid:
            raise KeyError(f"unknown history {hid}")
        t0 = time.perf_counter()
        with self.lock:
            idx = self._index(hid)
        qs = self.by_hid[hid]
        return {
            "id": hid, "sessions": len(idx.sids), "b_chunks": len(idx.B_emb), "f_chunks": len(idx.F_text),
            "question_date": self._latest_date(idx), "index_ms": _ms(t0),
            "questions": [{"id": q["question_id"], "question": q["question"], "date": q.get("question_date", ""),
                           "type": self.split.gold.get(q["question_id"], {}).get("question_type", "")}
                          for q in qs],
        }

    # ---------------------------------------------------------------- ask
    def ask(self, req: dict) -> Iterator[dict]:
        if not self.lock.acquire(blocking=False):
            yield {"type": "queued"}
            self.lock.acquire()
        try:
            yield from self._ask(req)
        except Exception as e:  # keep the server alive; the client shows the error
            _log(f"ask failed: {type(e).__name__}: {e}")
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        finally:
            self.lock.release()

    def _settings(self, s: dict) -> dict:
        a = self.a
        out = {
            "reranker": s.get("reranker") or a.reranker,
            "rerank_text": s.get("rerank_text") or "best2",
            "fusion": s.get("fusion") or "fused",
            "rerank_threshold": s.get("rerank_threshold"),
            "provider": s.get("provider") or a.provider,
            "model": s.get("model") or None,
            "max_chars": s.get("max_chars") or None,
            "attempts": s.get("attempts") or None,
            "timeout": s.get("timeout") or None,
            "dry_run": bool(s.get("dry_run", a.dry_run)),
            "show_message": bool(s.get("show_message", False)),
        }
        api = out["provider"] in ("groq", "openai")
        if not out["model"] and api:
            out["model"] = OPENAI_DEFAULT_MODEL if out["provider"] == "openai" else GROQ_DEFAULT_MODEL
        if not out["max_chars"]:
            out["max_chars"] = GROQ_DEFAULT_MAX_CHARS if api else 16000
        return out

    def _ask(self, req: dict) -> Iterator[dict]:
        R = self.R
        s = self._settings(req.get("settings") or {})
        qmeta = self.qs_by_id.get(req.get("question_id") or "")
        hid = qmeta["history_id"] if qmeta else req.get("history_id")
        if hid not in self.by_hid:
            raise KeyError(f"unknown history {hid}")
        question = (req.get("question") or (qmeta or {}).get("question") or "").strip()
        if not question:
            raise ValueError("empty question")
        T0 = time.perf_counter()
        idx = self._index(hid)
        qdate = (req.get("question_date") or "").strip()
        if _ISO.match(qdate):
            qdate = datetime.strptime(qdate, "%Y-%m-%d").strftime("%Y/%m/%d (%a) 23:59")
        qdate = qdate or (qmeta or {}).get("question_date") or self._latest_date(idx)
        gold = self.split.gold.get(qmeta["question_id"]) if qmeta else None
        gold_ev = set(gold.get("evidence", [])) if gold else set()
        sess = lambda si: {"sid": idx.sids[si], "date": idx.dates[si], "gold": idx.sids[si] in gold_ev}
        yield {"type": "start", "history_id": hid, "question": question, "question_date": qdate,
               "question_id": qmeta["question_id"] if qmeta else None, "settings": s}

        # 1. embedding
        t0 = time.perf_counter()
        qv = self.emb.encode_queries([question])[0]
        yield {"type": "step", "id": "embedding", "ms": _ms(t0), "data": {
            "model": R.EMB_MODEL, "prefix": R.BGE_PREFIX.strip(), "dim": int(qv.shape[0]),
            "norm": float(np.linalg.norm(qv)), "head": [float(x) for x in qv[:8]]}}

        # 2. first stage
        t0 = time.perf_counter()
        tr: dict = {}
        cands = R.first_stage(idx, qv, question, trace=tr)
        ranks, channels = {}, []
        for name, label in CHANNELS:
            sc = tr[name]
            order = [int(i) for i in np.argsort(-sc, kind="stable") if np.isfinite(sc[i])]
            ranks[name] = {si: r for r, si in enumerate(order)}
            channels.append({"name": name, "label": label,
                             "top": [{**sess(si), "score": float(sc[si])} for si in order[:10]]})
        yield {"type": "step", "id": "first_stage", "ms": _ms(t0), "data": {
            "rrf_k": R.RRF_K, "channels": channels,
            "candidates": [{**sess(c["si"]), "rrf": c["rrf"],
                            "ranks": {n: (ranks[n][c["si"]] + 1 if c["si"] in ranks[n] else None) for n, _ in CHANNELS}}
                           for c in cands]}}

        # 3. rerank
        t0 = time.perf_counter()
        rr = self._reranker(s["reranker"])
        if rr:
            texts = [R.rerank_text(idx, c, s["rerank_text"]) for c in cands]
            scores = rr.score([question] * len(cands), texts)
            ordered = R.final_order(cands, scores, s["fusion"])
            rr_rank = {cands[i]["sid"]: r for r, i in enumerate(np.argsort(-scores, kind="stable"))}
            rr_score = {c["sid"]: float(x) for c, x in zip(cands, scores)}
        else:
            ordered, rr_rank, rr_score = list(cands), None, None
        gt = R.guess_type(question)
        thr = s["rerank_threshold"]
        chosen = R.select_evidence(ordered, rr_score, gt, None if thr in (None, "") else float(thr))
        chosen_ids = {c["sid"] for c in chosen}
        rrf_pos = {c["sid"]: i for i, c in enumerate(cands)}
        yield {"type": "step", "id": "rerank", "ms": _ms(t0), "data": {
            "reranker": rr.name if rr else "none", "fusion": s["fusion"], "rerank_text": s["rerank_text"],
            "order": [{**sess(c["si"]), "score": rr_score[c["sid"]] if rr_score else None,
                       "rerank_rank": rr_rank[c["sid"]] + 1 if rr_rank else None,
                       "rrf_rank": rrf_pos[c["sid"]] + 1, "moved": rrf_pos[c["sid"]] - r,
                       "selected": c["sid"] in chosen_ids} for r, c in enumerate(ordered)]}}

        # 4. type / k / hint
        ev = R.pack_evidence(idx, chosen)
        hint = R.make_hint(gt, qdate, ev)
        yield {"type": "step", "id": "selection", "ms": 0, "data": {
            "guessed_type": gt, "gold_type": gold.get("question_type") if gold else None,
            "k": len(chosen), "rule": f"threshold ≥ {thr}" if thr not in (None, "") else f"top-{R.topk_for(gt)} for type",
            "question_date": qdate, "hint": hint}}

        # 5. evidence
        yield {"type": "step", "id": "evidence", "ms": 0, "data": {
            "chars": sum(len(e["text"]) for e in ev), "cap": R.EVIDENCE_CHARS,
            "passages": [{"label": f"E{i + 1}", "id": e["id"], "date": e["date"], "chars": len(e["text"]),
                          "text": e["text"], "gold": e["id"] in gold_ev} for i, e in enumerate(ev)]}}

        # 6. compose
        r = Retrieved(question_id=qmeta["question_id"] if qmeta else f"chat-{int(time.time())}",
                      history_id=hid, question=question, question_date=qdate, guessed_type=gt,
                      evidence=[Evidence(**e) for e in ev], hint=hint, candidates=[c["sid"] for c in ordered])
        comp = compose(r, s["max_chars"])
        yield {"type": "step", "id": "compose", "ms": 0, "data": {
            "chars": comp.chars, "limit": s["max_chars"], "passages": len(comp.evidence),
            "notes_chars": len(comp.notes), "retrieval_ms": _ms(T0),
            "message": comp.message if s["show_message"] else None}}

        # 7. answer
        who = f"{s['provider']} · {s['model']}" if s["provider"] != "reactor" else "Reactor"
        if s["dry_run"]:
            yield {"type": "answer", "status": "dry-run", "who": who, "answer": "", "citations": []}
        else:
            yield {"type": "waiting", "who": who,
                   "expect": "a few seconds" if s["provider"] != "reactor" else "typically 40–120 s"}
            yield self._answer(r, comp, s, who)
        if gold:
            yield {"type": "gold", "abstain": bool(gold.get("abstain")), "answer": str(gold.get("answer", "")),
                   "evidence": gold.get("evidence", []), "question_type": gold.get("question_type")}
        yield {"type": "done", "total_ms": _ms(T0)}

    def _answer(self, r: Retrieved, comp, s: dict, who: str) -> dict:
        from reactor_service import NoAnswer
        from .answering import make_answerer
        key = (s["provider"], s["model"], s["attempts"], s["timeout"])
        if key not in self.answerers:
            self.answerers[key] = make_answerer(s["provider"], model=s["model"], attempts=s["attempts"],
                                                timeout=s["timeout"], log=_log)
        answerer = self.answerers[key]
        t0 = time.perf_counter()
        try:
            if self.reactor_hid_for.get(key) != r.history_id:
                answerer.start_history(r.history_id)
                self.reactor_hid_for[key] = r.history_id
            a, attempts = answerer.ask(r, comp)
        except NoAnswer as e:
            return {"type": "answer", "status": "no-answer", "who": who, "answer": "", "citations": [],
                    "error": f"{type(e).__name__}: {e}", "latency_s": (time.perf_counter() - t0)}
        dates = {e.id: e.date for e in comp.evidence}
        return {"type": "answer", "status": "abstained" if a.abstained else "answered", "who": who,
                "answer": a.answer or "", "reply": a.reply or "", "latency_s": a.latency_s, "attempts": attempts,
                "reasoning": getattr(answerer, "last_reasoning", "") or "",
                "citations": [{"label": c.label, "id": c.id, "date": dates.get(c.id, "")} for c in a.citations]}


def _handler(engine: Engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter than the default per-request stderr line
            pass

        def _head(self, code: int, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _json(self, code: int, body: dict) -> None:
            self._head(code)
            self.wfile.write(json.dumps(body, ensure_ascii=False).encode())

        def do_OPTIONS(self):
            self._head(204)

        def do_GET(self):
            try:
                if self.path == "/api/meta":
                    return self._json(200, engine.meta())
                if self.path.startswith("/api/history/"):
                    return self._json(200, engine.history(self.path.rsplit("/", 1)[1]))
                self._json(404, {"error": "not found"})
            except KeyError as e:
                self._json(404, {"error": str(e).strip("'")})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            if self.path != "/api/ask":
                return self._json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError as e:
                return self._json(400, {"error": f"bad JSON: {e}"})
            self._head(200, "application/x-ndjson")
            try:
                with closing(engine.ask(req)) as events:  # closing releases the engine lock on disconnect
                    for ev in events:
                        self.wfile.write((json.dumps(ev, ensure_ascii=False) + "\n").encode())
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser went away

    return H


def cmd_serve(a) -> int:
    engine = Engine(a)
    srv = ThreadingHTTPServer((a.host, a.port), _handler(engine))
    _log(f"pipeline API on http://{a.host}:{a.port}  (split {a.split}, {len(engine.by_hid)} histories)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
