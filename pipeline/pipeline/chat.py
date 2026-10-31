"""Interactive REPL: type a question, watch every pipeline step (embedding -> dense/BM25 -> RRF -> reranker
-> type/k/hint -> evidence -> Reactor reasoning + answer).

    uv run pipeline chat --split dev [--history HID] [--dry-run] [--device cuda|cpu]

Commands:  :list   :h <history_id>   :q <question_id>   :date YYYY-MM-DD   :help   :exit
Anything else is a question about the current history.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape
from rich.text import Text

from .data import Split, load_histories, split_dir
from .models import Evidence, Retrieved
from .prompting import compose
from .retrieval import RETRIEVER_DIR

con = Console(highlight=False)
HELP = (":list  list histories | :h <id> switch history | :q <question_id> load a dataset question | "
        ":date YYYY-MM-DD set question date | :help | :exit (or Ctrl-D)")


def _t(t0: float) -> str:
    return f"{(time.perf_counter() - t0) * 1000:.0f} ms"


class Chat:
    def __init__(self, a):
        self.a = a
        self.dir = split_dir(a.split, a.data)
        self.split = Split(self.dir)
        self.qs_by_id = {q["question_id"]: q for q in self.split.questions}
        self.counts: dict[str, int] = {}
        for q in self.split.questions:
            self.counts[q["history_id"]] = self.counts.get(q["history_id"], 0) + 1
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
        with con.status(f"loading {R.EMB_MODEL} + reranker on {device} ..."):
            self.emb = R.Embedder(device=device)
            self.rr = R.Reranker(a.reranker, device=device) if a.reranker != "none" else None
            if device == "cpu":  # fp16 is slow / partly unsupported on CPU
                self.emb.m.float()
                if self.rr:
                    self.rr.m.float()
        con.print(f"[green]models ready[/] on {device} in {_t(t0)}  "
                  f"(embedder {R.EMB_MODEL}, reranker {self.rr.name if self.rr else 'none'})")
        self.answerer = None
        self.reactor_hid_for: str | None = None
        self.hid: str | None = None
        self.idx = None
        self.qdate = ""
        self.current_q: dict | None = None
        self.switch(a.history or self.split.history_ids[0])

    # ---------------------------------------------------------------- history
    def switch(self, hid: str) -> None:
        if hid not in self.counts:
            con.print(f"[red]unknown history {hid}[/] (:list to see them)")
            return
        t0 = time.perf_counter()
        with con.status(f"indexing {hid} ..."):
            h = load_histories(self.dir / "histories.jsonl", {hid})[hid]
            self.idx = self.R.build_indexes({hid: h}, self.emb, str(self.cache), log=lambda s: None)[hid]
        self.hid = hid
        dates = [d for d in self.idx.dates if d]
        self.qdate = max(dates, key=lambda d: self.R.parse_date(d) or datetime.min.date()) if dates else ""
        self.current_q = None
        idx = self.idx
        con.print(Panel(
            f"history [bold]{hid}[/]: {len(idx.sids)} sessions, {len(idx.B_text) if hasattr(idx, 'B_text') else len(idx.B_emb)} "
            f"user-turn chunks (B), {len(idx.F_text)} pair-window chunks (F)  [{_t(t0)}]\n"
            f"question date = {self.qdate or '(none)'} (latest session; change with :date)\n"
            f"{self.counts[hid]} dataset question(s) on this history: "
            + ", ".join(q["question_id"] for q in self.split.questions if q["history_id"] == hid)[:400],
            title="history", border_style="cyan"))

    def list(self) -> None:
        t = Table(title=f"{self.a.split}: {len(self.counts)} histories")
        t.add_column("history_id"); t.add_column("#questions", justify="right"); t.add_column("first question")
        first = {}
        for q in self.split.questions:
            first.setdefault(q["history_id"], q["question"])
        for hid, n in self.counts.items():
            t.add_row(("* " if hid == self.hid else "") + hid, str(n), escape(first[hid][:70]))
        con.print(t)

    # ---------------------------------------------------------------- question
    def ask(self, question: str, qmeta: dict | None = None) -> None:
        R, idx = self.R, self.idx
        gold = self.split.gold.get(qmeta["question_id"]) if qmeta else None
        gold_ev = set(gold.get("evidence", [])) if gold else set()
        star = lambda sid: f"[yellow]★ {sid}[/]" if sid in gold_ev else sid
        date_of = lambda si: idx.dates[si]
        T0 = time.perf_counter()

        # 1. embedding
        t0 = time.perf_counter()
        qv = self.emb.encode_queries([question])[0]
        con.rule(f"[bold]1. Query embedding[/]  ({_t(t0)})")
        con.print(f"model {R.EMB_MODEL} | prefix '{R.BGE_PREFIX.strip()}' | dim {qv.shape[0]} | "
                  f"norm {np.linalg.norm(qv):.4f}")
        con.print("first 8: [" + ", ".join(f"{x:+.4f}" for x in qv[:8]) + ", ...]")

        # 2. first stage
        t0 = time.perf_counter()
        tr: dict = {}
        cands = R.first_stage(idx, qv, question, trace=tr)
        con.rule(f"[bold]2. First stage: dense-B, dense-F, BM25 -> RRF(k={R.RRF_K})[/]  ({_t(t0)})")
        if gold_ev:
            con.print("[yellow]★ = gold evidence session[/]")
        ranks = {}
        tables = []
        for name, label in (("dense_B", "dense-B (user turns, cos)"), ("dense_F", "dense-F (pair windows, cos)"),
                            ("bm25", "BM25 (pair windows)")):
            sc = tr[name]
            order = [i for i in np.argsort(-sc, kind="stable") if np.isfinite(sc[i])]
            ranks[name] = {int(si): r for r, si in enumerate(order)}
            t = Table(title=label, title_style="bold", pad_edge=False)
            t.add_column("#", justify="right"); t.add_column("session"); t.add_column("date"); t.add_column("score", justify="right")
            for r, si in enumerate(order[:10]):
                t.add_row(str(r + 1), star(idx.sids[si]), date_of(si)[:10], f"{sc[si]:.3f}")
            tables.append(t)
        from rich.columns import Columns
        con.print(Columns(tables))
        t = Table(title=f"RRF top {len(cands)} (candidates for the reranker)", title_style="bold")
        for c in ("#", "session", "date", "rrf", "B#", "F#", "BM25#"):
            t.add_column(c, justify="right" if c != "session" else "left")
        for r, c in enumerate(cands):
            si = c["si"]
            rk = lambda n: str(ranks[n].get(si, -1) + 1) if si in ranks[n] else "-"
            t.add_row(str(r + 1), star(c["sid"]), date_of(si)[:10], f"{c['rrf']:.4f}", rk("dense_B"), rk("dense_F"), rk("bm25"))
        con.print(t)

        # 3. rerank
        t0 = time.perf_counter()
        if self.rr:
            texts = [R.rerank_text(idx, c, self.a.rerank_text) for c in cands]
            scores = self.rr.score([question] * len(cands), texts)
            ordered = R.final_order(cands, scores, self.a.fusion)
        else:
            scores, ordered = None, list(cands)
        con.rule(f"[bold]3. Reranker ({self.rr.name if self.rr else 'none'}, fusion={self.a.fusion})[/]  ({_t(t0)})")
        rrf_pos = {c["sid"]: i for i, c in enumerate(cands)}
        if scores is not None:
            rr_rank = {cands[i]["sid"]: r for r, i in enumerate(np.argsort(-scores, kind="stable"))}
            rr_score = {c["sid"]: float(s) for c, s in zip(cands, scores)}
        t = Table(title="final order")
        for c in ("#", "session", "date", "rerank score", "rerank #", "RRF #", "moved"):
            t.add_column(c, justify="left" if c == "session" else "right")
        gt = R.guess_type(question)
        k = R.topk_for(gt)
        for r, c in enumerate(ordered):
            d = rrf_pos[c["sid"]] - r
            mv = f"[green]↑{d}[/]" if d > 0 else (f"[red]↓{-d}[/]" if d < 0 else "=")
            row = [str(r + 1), star(c["sid"]), date_of(c["si"])[:10],
                   f"{rr_score[c['sid']]:+.3f}" if scores is not None else "-",
                   str(rr_rank[c["sid"]] + 1) if scores is not None else "-", str(rrf_pos[c["sid"]] + 1), mv]
            t.add_row(*row, style="bold" if r < k else "dim")
        con.print(t)

        # 4. type / k / hint
        chosen = ordered[:k]
        ev = R.pack_evidence(idx, chosen)
        hint = R.make_hint(gt, self.qdate, ev)
        con.rule("[bold]4. Type guess, k, hint[/]")
        gtrue = f"  (gold type: {gold['question_type']})" if gold and gold.get("question_type") else ""
        con.print(f"guessed_type = [bold]{gt}[/]{gtrue}   k = {k} (top-{k} of the final order are sent)   "
                  f"question_date = {self.qdate}")
        con.print(Text(hint)) if hint else con.print("[dim](no hint: only for temporal-reasoning / knowledge-update)[/]")

        # 5. evidence
        con.rule(f"[bold]5. Evidence sent ({sum(len(e['text']) for e in ev)} chars, cap {R.EVIDENCE_CHARS})[/]")
        for i, e in enumerate(ev):
            prev = e["text"][:200].replace("\n", " ⏎ ")
            con.print(f"[bold cyan][E{i + 1}][/] {star(e['id'])}  {e['date']}  {len(e['text'])} chars")
            con.print(Text("    " + prev + ("..." if len(e["text"]) > 200 else ""), style="dim"))

        # 6. reactor
        r = Retrieved(question_id=qmeta["question_id"] if qmeta else f"chat-{int(time.time())}",
                      history_id=self.hid, question=question, question_date=self.qdate, guessed_type=gt,
                      evidence=[Evidence(**e) for e in ev], hint=hint, candidates=[c["sid"] for c in ordered])
        comp = compose(r, self.a.max_chars)
        con.rule(f"[bold]6. {self._who()}[/]")
        con.print(f"message {comp.chars} chars (limit {self.a.max_chars}), {len(comp.evidence)} passages, "
                  f"notes {len(comp.notes)} chars   [retrieval total {_t(T0)}]")
        if self.a.show_message:
            con.print(Panel(Text(comp.message, style="dim"), title="message sent"))
        if self.a.dry_run:
            con.print(f"[yellow]--dry-run: nothing sent to {self._who()}[/]")
        else:
            self._reactor(r, comp)
        if gold:
            g = "ABSTAIN (unanswerable)" if gold.get("abstain") else escape(str(gold.get("answer")))
            con.print(f"[magenta]gold answer:[/] {g}   [magenta]gold evidence:[/] {', '.join(gold.get('evidence', []))}")

    def _who(self) -> str:
        return f"Groq {self.a.model}" if self.a.provider == "groq" else "Reactor"

    def _reactor(self, r: Retrieved, comp) -> None:
        from reactor_service import NoAnswer
        from .answering import make_answerer
        if self.answerer is None:
            self.answerer = make_answerer(self.a.provider, model=self.a.model, attempts=self.a.attempts,
                                          timeout=self.a.timeout,
                                            log=lambda s: con.print(f"[dim]{s}[/]"))
        t0 = time.perf_counter()
        try:
            wait = "a few seconds" if self.a.provider == "groq" else "typically 40-120 s"
            with con.status(f"waiting for {self._who()} ({wait}) ..."):
                if self.reactor_hid_for != self.hid:
                    self.answerer.start_history(self.hid)
                    self.reactor_hid_for = self.hid
                a, attempts = self.answerer.ask(r, comp)
        except NoAnswer as e:
            con.print(f"[red]no answer after retries: {type(e).__name__}: {e}[/]  ({_t(t0)})")
            return
        con.print(f"[dim]reply after {a.latency_s:.1f}s, attempt {attempts}[/]")
        reasoning = getattr(self.answerer, "last_reasoning", "")
        if reasoning:
            con.print(Panel(Text(reasoning, style="dim"), title=f"{self._who()} reasoning (hidden chain of thought)"))
        con.print(Panel(Text(a.reply or "(empty reply)", style="dim"), title=f"{self._who()} full reply"))
        label = "ABSTAINED" if a.abstained else "FINAL ANSWER"
        con.print(Panel(Text(a.answer or "", style="bold"), title=label, border_style="green"))
        dates = {e.id: e.date for e in comp.evidence}
        for c in a.citations:
            con.print(Text(f"  cites [{c.label}] -> {c.id}  {dates.get(c.id, '')}"))
        if not a.citations:
            con.print("  [dim](no citations)[/]")

    # ---------------------------------------------------------------- loop
    def loop(self) -> int:
        try:
            import readline  # noqa: F401  (line editing + history)
        except ImportError:
            pass
        con.print(f"[dim]{HELP}[/]")
        while True:
            try:
                line = input(f"\n[{self.hid}] ? ").strip()
            except (EOFError, KeyboardInterrupt):
                con.print()
                return 0
            if not line:
                continue
            try:
                if line in (":exit", ":quit", ":x"):
                    return 0
                if line == ":help":
                    con.print(HELP)
                elif line == ":list":
                    self.list()
                elif line.startswith(":h "):
                    self.switch(line[3:].strip())
                elif line.startswith(":date "):
                    d = datetime.strptime(line[6:].strip(), "%Y-%m-%d")
                    self.qdate = d.strftime("%Y/%m/%d (%a) 23:59")
                    con.print(f"question date = {self.qdate}")
                elif line.startswith(":q "):
                    q = self.qs_by_id.get(line[3:].strip())
                    if not q:
                        con.print("[red]unknown question id[/]")
                        continue
                    if q["history_id"] != self.hid:
                        self.switch(q["history_id"])
                    self.qdate = q.get("question_date") or self.qdate
                    con.print(Panel(f"{escape(q['question'])}\n[dim]{q['question_id']} | date {self.qdate}[/]", title="dataset question"))
                    self.ask(q["question"], q)
                elif line.startswith(":"):
                    con.print(f"[red]unknown command[/]  {HELP}")
                else:
                    self.ask(line)
            except Exception as e:  # keep the REPL alive
                con.print_exception(max_frames=3)
                con.print(f"[red]{type(e).__name__}: {e}[/]")


def cmd_chat(a) -> int:
    return Chat(a).loop()
