#!/usr/bin/env python3
"""Production retriever for the RxAI memory challenge.

Pipeline per question:
  1. Index (once per history, cached as npz): B = user-turn chunks, F = user+assistant
     pair windows (256 tok, stride 192), BM25 over F chunks. bge-small-en-v1.5 fp16.
  2. First stage: RRF(k=60) over session rankings from B, F and BM25
     (session score = max over its chunks). Keep top 20 candidates + their best chunks.
  3. Rerank (optional): cross-encoder over (question, best chunk(s)) for the 20 candidates.
  4. Per-type top-k using a rule-based type guesser.
  5. Evidence packing (<= 14000 chars) + date hints for temporal / knowledge-update.
"""
import json, os, re, time
from datetime import date as _date

import numpy as np

EMB_MODEL = "BAAI/bge-small-en-v1.5"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60
N_CAND = 20
WIN, STRIDE = 256, 192
CACHE_VER = "v1"
EVIDENCE_CHARS = 14000
TOPK = {"single-session-user": 3, "single-session-assistant": 3, "knowledge-update": 4,
        "temporal-reasoning": 4, "multi-session": 6}
RERANKERS = {"base": "BAAI/bge-reranker-base", "v2-m3": "BAAI/bge-reranker-v2-m3"}
# Defaults chosen from eval_retriever.py on all 526 dev questions (see RESULTS.md).
DEFAULT_CONFIG = {"reranker": "base", "rerank_text": "best2", "fusion": "fused"}
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("RETRIEVER_DATA_ROOT", os.path.expanduser("~/reactor-test/labeler/data"))


# ------------------------------------------------------------------ data
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_histories(path, wanted=None):
    hs = {}
    with open(path) as f:
        for line in f:
            if wanted is not None:
                m = re.search(r'"history_id":\s*"([^"]+)"', line[:200])
                if m and m.group(1) not in wanted:
                    continue
            h = json.loads(line)
            if wanted is None or h["history_id"] in wanted:
                hs[h["history_id"]] = h
    return hs


# ------------------------------------------------------------------ type guesser
_RULES = [
    ("single-session-assistant", re.compile(
        r"\byou (said|told|mentioned|suggested|recommended|provided|listed|gave|shared|described|explained)\b"
        r"|\brecommend|\bsuggest|previous (conversation|chat|discussion)|remind me|our (last |earlier )?(conversation|chat)", re.I)),
    ("temporal-reasoning", re.compile(
        r"\b(days?|weeks?|months?|years?) (ago|before|after|passed|had passed|between|earlier|later)\b"
        r"|\bago\b|\bbefore\b|\bafter\b|\bfirst\b|\bwhen did\b|\bhow long (had|ago|did it take)\b|\border\b|\bearliest\b|\bbetween\b", re.I)),
    ("knowledge-update", re.compile(
        r"\b(current(ly)?|now|still|latest|most recent(ly)?|so far|these days|usually|anymore)\b", re.I)),
    ("multi-session", re.compile(r"\bhow many\b|\bhow much\b|\btotal\b|\ball\b|\bin total\b|\bcombined\b|\baltogether\b", re.I)),
]


def guess_type(question):
    for t, rx in _RULES:
        if rx.search(question):
            return t
    return "single-session-user"


def topk_for(qtype):
    return TOPK.get(qtype, 4)


# ------------------------------------------------------------------ chunking
def session_pairs(turns):
    """List of pairs; each pair = list of (role, content). Assistant attaches to preceding user."""
    out, cur = [], []
    for t in turns:
        c = (t.get("content") or "").strip()
        if not c:
            continue
        if t["role"] == "user" and cur:
            out.append(cur); cur = []
        cur.append((t["role"], c))
    if cur:
        out.append(cur)
    return out


def pair_text(p):
    return "\n".join(f"{r}: {c}" for r, c in p)


def chunk_history(h, tokenizer):
    B_text, B_sess, F_text, F_sess, F_pair = [], [], [], [], []
    src = []
    for si, s in enumerate(h["sessions"]):
        for t in s["turns"]:
            c = (t.get("content") or "").strip()
            if c and t["role"] == "user":
                B_text.append(c); B_sess.append(si)
        for pi, p in enumerate(session_pairs(s["turns"])):
            src.append((pair_text(p), si, pi))
    if src:
        enc = tokenizer([x[0] for x in src], add_special_tokens=False,
                        return_offsets_mapping=True, truncation=False)
        for (text, si, pi), offs in zip(src, enc["offset_mapping"]):
            n = len(offs)
            if n <= WIN:
                F_text.append(text); F_sess.append(si); F_pair.append(pi); continue
            start = 0
            while True:
                end = min(start + WIN, n)
                F_text.append(text[offs[start][0]:offs[end - 1][1]]); F_sess.append(si); F_pair.append(pi)
                if end == n:
                    break
                start += STRIDE
    return {"B_text": B_text, "B_sess": np.array(B_sess, dtype=np.int32),
            "F_text": F_text, "F_sess": np.array(F_sess, dtype=np.int32),
            "F_pair": np.array(F_pair, dtype=np.int32)}


# ------------------------------------------------------------------ BM25 (numpy, Okapi)
_TOK = re.compile(r"[a-z0-9]+")


def tok(s):
    return _TOK.findall(s.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.vocab, rows, cols, vals = {}, [], [], []
        lens = np.zeros(len(docs), dtype=np.float32)
        for i, d in enumerate(docs):
            toks = tok(d); lens[i] = len(toks)
            tf = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            for w, c in tf.items():
                j = self.vocab.setdefault(w, len(self.vocab))
                rows.append(i); cols.append(j); vals.append(c)
        from scipy.sparse import csc_matrix
        self.tf = csc_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(docs), max(1, len(self.vocab))))
        df = np.diff(self.tf.indptr).astype(np.float32)
        N = len(docs)
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
        self.norm = (k1 * (1 - b + b * lens / max(1.0, lens.mean()))).astype(np.float32)
        self.n = N

    def scores(self, query):
        out = np.zeros(self.n, dtype=np.float32)
        for w in set(tok(query)):
            j = self.vocab.get(w)
            if j is None:
                continue
            col = self.tf.getcol(j)
            idx, tf = col.indices, col.data
            out[idx] += self.idf[j] * tf * (self.k1 + 1) / (tf + self.norm[idx])
        return out


# ------------------------------------------------------------------ models
class Embedder:
    def __init__(self, model=EMB_MODEL, device="cuda", batch_size=256):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device=device)
        self.m.half(); self.m.max_seq_length = 512
        self.bs = batch_size
        self.tokenizer = self.m.tokenizer

    def encode(self, texts):
        if not texts:
            return np.zeros((0, self.m.get_sentence_embedding_dimension()), np.float16)
        return self.m.encode(texts, batch_size=self.bs, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype(np.float16)

    def encode_queries(self, qs):
        return self.encode([BGE_PREFIX + q for q in qs]).astype(np.float32)


class Reranker:
    def __init__(self, name, device="cuda", batch_size=None, max_length=512):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        name = RERANKERS.get(name, name)
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float16).to(device).eval()
        self.device, self.max_length = device, max_length
        # token budget per batch (tokens = batch * padded_len); tuned to fit 4 GB
        self.tok_budget = batch_size or (8192 if "m3" in name else 16384)

    def score(self, queries, passages):
        torch = self.torch
        enc = self.tok(queries, passages, truncation="only_second", max_length=self.max_length)
        ids = enc["input_ids"]
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out = np.zeros(len(ids), np.float32)
        i = 0
        budget = self.tok_budget
        with torch.inference_mode():
            while i < len(order):
                L = len(ids[order[i]])
                bs = max(1, budget // L)
                sel = order[i:i + bs]
                batch = self.tok.pad({k: [enc[k][j] for j in sel] for k in enc.keys()}, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                try:
                    logits = self.m(**batch).logits[:, 0].float().cpu().numpy()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); budget //= 2
                    if budget < 512:
                        raise
                    continue
                out[sel] = logits
                i += len(sel)
        self.tok_budget = budget
        return out


# ------------------------------------------------------------------ index
class HistoryIndex:
    def __init__(self, h, arrays):
        self.hid = h["history_id"]
        self.sessions = h["sessions"]
        self.sids = [s["session_id"] for s in self.sessions]
        self.dates = [s.get("date", "") for s in self.sessions]
        self.B_emb, self.B_sess = arrays["B_emb"].astype(np.float32), arrays["B_sess"]
        self.F_emb, self.F_sess = arrays["F_emb"].astype(np.float32), arrays["F_sess"]
        self.F_pair, self.F_text = arrays["F_pair"], list(arrays["F_text"])
        self._bm25 = None
        self._pairs = {}

    @property
    def bm25(self):
        if self._bm25 is None:
            self._bm25 = BM25(self.F_text)
        return self._bm25

    def pairs(self, si):
        if si not in self._pairs:
            self._pairs[si] = session_pairs(self.sessions[si]["turns"])
        return self._pairs[si]


def build_indexes(hs, embedder, cache_dir, group=40, log=print):
    """Return {hid: HistoryIndex}. Encodes uncached histories in large GPU batches."""
    os.makedirs(cache_dir, exist_ok=True)
    out, todo = {}, []
    path = lambda hid: os.path.join(cache_dir, f"{hid}.{CACHE_VER}.npz")
    for hid, h in hs.items():
        p = path(hid)
        if os.path.exists(p):
            a = dict(np.load(p))
            c = chunk_history(h, embedder.tokenizer)  # text is re-derived; only vectors are cached
            if len(c["F_text"]) == len(a["F_emb"]) and len(c["B_text"]) == len(a["B_emb"]):
                out[hid] = HistoryIndex(h, {**c, **a}); continue
        todo.append(hid)
    log(f"index: {len(out)} cached, {len(todo)} to encode")
    t0 = time.perf_counter()
    for g in range(0, len(todo), group):
        hids = todo[g:g + group]
        chunks = {hid: chunk_history(hs[hid], embedder.tokenizer) for hid in hids}
        texts, spans = [], {}
        for hid, c in chunks.items():
            b0 = len(texts); texts += c["B_text"]; f0 = len(texts); texts += c["F_text"]
            spans[hid] = (b0, f0, len(texts))
        E = embedder.encode(texts)
        for hid, c in chunks.items():
            b0, f0, f1 = spans[hid]
            a = {"B_emb": E[b0:f0], "F_emb": E[f0:f1]}
            np.savez(path(hid), **a)
            out[hid] = HistoryIndex(hs[hid], {**c, **a})
        log(f"  encoded {min(g + group, len(todo))}/{len(todo)} histories, {len(texts)} chunks "
            f"({time.perf_counter() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------ retrieval
def _sess_max(scores, sess, n):
    best = np.full(n, -np.inf, dtype=np.float32)
    np.maximum.at(best, sess, scores)
    return best


def _ranks(scores):
    """rank position (0 = best) for each element; -inf entries get rank len."""
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64); r[order] = np.arange(len(scores))
    r[~np.isfinite(scores)] = len(scores) + 10**6
    return r


def first_stage(idx, qvec, qtext, n_cand=N_CAND):
    """Return list of candidates [{si, sid, rrf, chunks:[F idx ordered best first]}]."""
    n = len(idx.sids)
    b = idx.B_emb @ qvec if len(idx.B_emb) else np.zeros(0, np.float32)
    f = idx.F_emb @ qvec
    bm = idx.bm25.scores(qtext)
    rrf = np.zeros(n, dtype=np.float64)
    for sc, sess in ((b, idx.B_sess), (f, idx.F_sess), (bm, idx.F_sess)):
        if len(sc) == 0:
            continue
        r = _ranks(_sess_max(sc, sess, n))
        rrf += 1.0 / (RRF_K + r + 1)
    cand = np.argsort(-rrf, kind="stable")[:n_cand]
    # chunk-level ordering inside a session: RRF of dense-F rank and BM25 rank
    crrf = 1.0 / (RRF_K + _ranks(f) + 1) + 1.0 / (RRF_K + _ranks(bm) + 1)
    out = []
    for si in cand:
        ch = np.where(idx.F_sess == si)[0]
        ch = ch[np.argsort(-crrf[ch], kind="stable")]
        out.append({"si": int(si), "sid": idx.sids[si], "rrf": float(rrf[si]), "chunks": ch.tolist()})
    return out


def rerank_text(idx, c, mode):
    ch = c["chunks"]
    if not ch:
        return ""
    if mode == "best1":
        return idx.F_text[ch[0]]
    return "\n...\n".join(idx.F_text[i] for i in ch[:2])


def final_order(cands, rr_scores=None, fusion="fused"):
    """cands are in RRF order. Returns reordered list."""
    if rr_scores is None:
        return list(cands)
    rr_rank = _ranks(np.asarray(rr_scores, np.float32))
    if fusion == "rerank":
        key = -rr_rank.astype(np.float64)
    else:
        key = 1.0 / (RRF_K + rr_rank + 1) + 1.0 / (RRF_K + np.arange(len(cands)) + 1)
    o = np.argsort(-key, kind="stable")
    return [cands[i] for i in o]


# ------------------------------------------------------------------ dates / hints
_DATE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")


def parse_date(s):
    m = _DATE.search(s or "")
    return _date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def make_hint(qtype, question_date, evidence):
    if qtype not in ("temporal-reasoning", "knowledge-update") or not evidence:
        return ""
    qd = parse_date(question_date)
    rows = []
    for i, e in enumerate(evidence):
        d = parse_date(e["date"])
        rows.append((d, f"E{i + 1}", e["id"], e["date"]))
    lines = [f"[hint] question date (today) = {question_date}"]
    for d, lab, sid, ds in rows:
        ago = f", {(qd - d).days} days before today" if (d and qd) else ""
        lines.append(f"[hint] {lab} ({sid}) date = {ds}{ago}")
    dated = sorted([r for r in rows if r[0]], key=lambda r: r[0])
    if len(dated) > 1:
        lines.append("[hint] chronological order: " + " < ".join(r[1] for r in dated))
        for a, b in zip(dated, dated[1:]):
            lines.append(f"[hint] {a[1]} -> {b[1]}: {(b[0] - a[0]).days} days")
        if len(dated) > 2:
            lines.append(f"[hint] {dated[0][1]} -> {dated[-1][1]}: {(dated[-1][0] - dated[0][0]).days} days")
    if qtype == "knowledge-update" and dated:
        lines.append(f"[hint] most recent session = {dated[-1][1]} ({dated[-1][3]}); "
                     "if values conflict, the latest one is current")
    return "\n".join(lines)


# ------------------------------------------------------------------ evidence packing
def _trim(s, n):
    s = s.strip()
    return s if len(s) <= n else s[:n].rstrip() + " [...]"


def session_text(idx, c, budget, top_chunks=3):
    """Top-ranked chunks first (as their windows), then remaining pairs in order, trimmed."""
    si = c["si"]
    pairs = idx.pairs(si)
    parts, used_pairs = [], set()
    for ci in c["chunks"][:top_chunks]:
        pi = int(idx.F_pair[ci])
        w = idx.F_text[ci]
        if not w.startswith("user:") and pairs and pairs[pi][0][0] == "user":
            w = f"user: {_trim(pairs[pi][0][1], 600)}\n[...] {w}"
        parts.append(w); used_pairs.add(pi)
    for pi, p in enumerate(pairs):
        if pi in used_pairs:
            continue
        parts.append("\n".join(f"{r}: {_trim(t, 500 if r == 'user' else 300)}" for r, t in p))
    out, n = [], 0
    for p in parts:
        if n + len(p) + 2 > budget:
            rest = budget - n - 2
            if rest > 200:
                out.append(p[:rest].rstrip() + " [...]")
            break
        out.append(p); n += len(p) + 2
    return "\n\n".join(out)


def pack_evidence(idx, chosen, total=EVIDENCE_CHARS):
    """Water-filling budget: equal share, unused share flows to the rest; best sessions first."""
    k = len(chosen)
    if not k:
        return []
    full = [len(session_text(idx, c, 10**9)) for c in chosen]
    alloc, left, rem = [0] * k, total - 60 * k, list(range(k))
    while rem:
        share = left // len(rem)
        small = [i for i in rem if full[i] <= share]
        if not small:
            for i in rem:
                alloc[i] = share
            break
        for i in small:
            alloc[i] = full[i]; left -= full[i]
        rem = [i for i in rem if i not in small]
    return [{"id": c["sid"], "date": idx.dates[c["si"]], "text": session_text(idx, c, alloc[i])}
            for i, c in enumerate(chosen)]


# ------------------------------------------------------------------ high-level API
class Retriever:
    def __init__(self, split_dir, cache_dir, reranker=DEFAULT_CONFIG["reranker"],
                 rerank_text_mode=DEFAULT_CONFIG["rerank_text"], fusion=DEFAULT_CONFIG["fusion"],
                 device="cuda", log=print):
        self.log = log
        self.questions = load_jsonl(os.path.join(split_dir, "questions.jsonl"))
        hs = load_histories(os.path.join(split_dir, "histories.jsonl"),
                            {q["history_id"] for q in self.questions})
        self.emb = Embedder(device=device)
        t = time.perf_counter()
        self.idx = build_indexes(hs, self.emb, cache_dir, log=log)
        self.index_s = time.perf_counter() - t
        self.rr = Reranker(reranker, device=device) if reranker else None
        self.mode, self.fusion = rerank_text_mode, fusion

    def run(self, questions=None):
        qs = questions or self.questions
        Q = self.emb.encode_queries([q["question"] for q in qs])
        cands = [first_stage(self.idx[q["history_id"]], Q[i], q["question"]) for i, q in enumerate(qs)]
        rr = [None] * len(qs)
        if self.rr:
            qq, pp, spans = [], [], []
            for q, cs in zip(qs, cands):
                idx = self.idx[q["history_id"]]
                spans.append((len(qq), len(qq) + len(cs)))
                for c in cs:
                    qq.append(q["question"]); pp.append(rerank_text(idx, c, self.mode))
            sc = self.rr.score(qq, pp)
            rr = [sc[a:b] for a, b in spans]
        out = []
        for q, cs, r in zip(qs, cands, rr):
            idx = self.idx[q["history_id"]]
            ordered = final_order(cs, r, self.fusion)
            gt = guess_type(q["question"])
            ev = pack_evidence(idx, ordered[:topk_for(gt)])
            out.append({"question_id": q["question_id"], "history_id": q["history_id"],
                        "question": q["question"], "question_date": q.get("question_date", ""),
                        "guessed_type": gt, "evidence": ev,
                        "hint": make_hint(gt, q.get("question_date", ""), ev),
                        "candidates": [c["sid"] for c in ordered]})
        return out
