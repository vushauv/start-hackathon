"""Chunk every session of a split into overlapping windows of whole sentences, stored in that split's chunks table.

    python db/chunk.py --split dev                          # chunk the sessions not chunked yet
    python db/chunk.py --split dev --reset                  # drop the split's chunks (embeddings too), chunk again
    python db/chunk.py --split dev --show 5                 # print 5 sample chunks with a question; writes nothing
    python db/chunk.py --split dev --expand 's_abc:3' --before 1 --after 1   # a chunk merged with its neighbours

How a session is chunked:
1. Units. Each turn is cut into lines, each line into sentences, and a sentence longer than --unit-cap tokens at token
   boundaries. A unit keeps its turn, its character span in the turn's content, and how strong a boundary its start
   is: 3 a new turn, 2 a new paragraph (after a blank line), 1 a new line, 0 a new sentence.
2. Greedy packing. Units go into a chunk while it stays within --max-tokens; the unit that would overflow starts the
   next chunk. Once the chunk is past --min-fill of the limit, the cut moves back to the strongest boundary there, so a
   chunk ends where a turn or a paragraph does rather than mid-paragraph.
3. Questions. A unit with a '?' opens a question block: the rest of that speaker's turn and the next turn, the answer.
   When a block that fits in one chunk would be cut, the chunk ends before the question, so the next chunk holds
   question and answer together; when the chunk is still short, the overlap in front shrinks instead. A block too long
   for one chunk gets as much of the answer as fits.
4. Overlap. The next chunk starts with up to --overlap tokens of the previous chunk's end, starting at the strongest
   boundary in that stretch; less when a question block right after needs the room.
Each turn's piece of a chunk starts with [USER] or [ASSISTANT], including a turn the chunk enters midway. Token counts
come from the embedding model's tokenizer (bge-m3 by default).
"""
import argparse
import random
import re
import sys
from dataclasses import dataclass

import psycopg
from tokenizers import Tokenizer

from settings import DSN

TAGS = {"user": "[USER]", "assistant": "[ASSISTANT]"}
TURN, PARAGRAPH, LINE, SENTENCE = 3, 2, 1, 0

SENTENCE_END = re.compile(r"[.!?]+[\"')\]*_]*\s+")   # end punctuation, closing quotes/brackets/markdown, whitespace
QUESTION = re.compile(r"\?(?=[\s\"')\]*_]|$)")        # a '?' ending a word: not the one in a URL's query string
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "approx", "no", "fig",
                 "inc", "ltd", "co", "mt", "ft", "est", "dept", "jan", "feb", "aug", "sept", "oct", "nov", "dec"}


@dataclass
class Unit:
    turn: int        # turns.idx
    start: int       # character span in that turn's content, end exclusive
    end: int
    level: int       # boundary strength of the unit's start: TURN, PARAGRAPH, LINE or SENTENCE
    tokens: int = 0
    question: bool = False


@dataclass
class Config:
    max_tokens: int = 500
    overlap: int = 200
    min_fill: float = 0.75
    unit_cap: int = 300


# ---------------------------------------------------------------- units

def real_break(line, start, m):
    """Whether the '.' that m matched in line ends a sentence, rather than 'e.g.', 'Dr.' or the '1.' of a list."""
    if line[m.start()] != ".":
        return True
    before = line[start:m.start()]
    word = re.search(r"(\S*)$", before).group(1).lower().lstrip("(\"'*_")
    if word in ABBREVIATIONS or len(word) == 1 and word.isalpha():
        return False
    if word.isdigit() and before.strip() == word:   # a list marker: '1.' alone so far
        return False
    return not line[m.end():m.end() + 1].islower()   # 'approx. five': no capital, no new sentence


def sentence_spans(content):
    """(start, end, level) of each sentence in one turn's content."""
    spans, prev_end = [], None
    for line in re.finditer(r"[^\n]+", content):
        if not line.group().strip():
            continue
        if prev_end is None:
            level = TURN
        else:
            level = PARAGRAPH if content.count("\n", prev_end, line.start()) >= 2 else LINE
        text, start = line.group(), 0
        cuts = [m for m in SENTENCE_END.finditer(text) if real_break(text, start, m)] if text else []
        for m in cuts + [None]:
            end = len(text) if m is None else m.start() + len(m.group().rstrip())
            piece = text[start:end]
            if piece.strip():
                lead = len(piece) - len(piece.lstrip())
                spans.append((line.start() + start + lead, line.start() + len(piece.rstrip()) + start, level))
                level = SENTENCE
            if m is not None:
                start = m.end()
        prev_end = line.end()
    return spans


def make_units(turns, tokenizer, cfg):
    """The units of a session, turns being {idx: (role, content)}; sentences over cfg.unit_cap tokens are cut."""
    raw = [(t, s, e, lvl) for t, (_, content) in sorted(turns.items()) for s, e, lvl in sentence_spans(content)]
    encodings = tokenizer.encode_batch([turns[t][1][s:e] for t, s, e, _ in raw], add_special_tokens=False)
    units = []
    for (t, s, e, lvl), enc in zip(raw, encodings):
        n = len(enc.ids)
        if n <= cfg.unit_cap:
            units.append(Unit(t, s, e, lvl, n))
            continue
        for k in range(0, n, cfg.unit_cap):    # cut at token starts, cfg.unit_cap tokens a piece
            a = s + enc.offsets[k][0] if k else s
            b = s + enc.offsets[k + cfg.unit_cap][0] if k + cfg.unit_cap < n else e
            units.append(Unit(t, a, b, lvl if k == 0 else SENTENCE, min(cfg.unit_cap, n - k)))
    for u in units:
        u.question = bool(QUESTION.search(turns[u.turn][1], u.start, u.end))
    return units


# ---------------------------------------------------------------- packing

def pack(units, cfg, tag_tokens):
    """[(a, b)]: each chunk as a range of unit indices, b exclusive, in order; consecutive chunks may overlap."""
    n = len(units)
    if not n:
        return []
    prefix, turn_starts = [0], [0]
    for u in units:
        prefix.append(prefix[-1] + u.tokens + 1)                       # +1: the space or newline before it
        turn_starts.append(turn_starts[-1] + (u.level == TURN))

    def cost(a, b):
        """Approximate tokens of units a..b-1 rendered: their tokens plus a speaker tag per turn they touch."""
        return prefix[b] - prefix[a] + tag_tokens * (1 + turn_starts[b] - turn_starts[a + 1])

    turn_end = {}
    for i, u in enumerate(units):
        turn_end[u.turn] = i + 1
    order = sorted(turn_end)
    answer_turn = dict(zip(order, order[1:]))
    # where the question block opened by unit i ends: its turn's end, or the end of the next turn, the answer
    block_end = [None if not u.question else turn_end[answer_turn.get(u.turn, u.turn)] for u in units]
    limit, floor = cfg.max_tokens, cfg.min_fill * cfg.max_tokens

    def fits(q):
        return block_end[q] is not None and cost(q, block_end[q]) <= limit

    def splits_block(c, start):
        """Whether cutting before unit c separates a question in start..c-1 from part of its fitting block."""
        return any(fits(q) and block_end[q] > c for q in range(start, c))

    def grow(start, end):
        while end < n and cost(start, end + 1) <= limit:
            end += 1
        return end

    def adjust(start, new, end):
        """The chunk start..end-1 (end < n) after the question and boundary rules; new = first unit not seen yet."""
        for q in range(new, end):
            if fits(q) and block_end[q] > end:                         # a block that fits is about to be cut
                if q > new and cost(start, q) >= floor:
                    return start, q                                    # end before the question
                s = next((s for s in range(start, new + 1) if cost(s, block_end[q]) <= limit), None)
                if s is not None:                                      # less overlap in front: the block fits
                    return s, grow(s, block_end[q])
                if q > new:
                    return start, q
                break
        best = end
        for c in range(end - 1, new, -1):                              # back off to a stronger boundary
            if cost(start, c) < floor:
                break
            if units[c].level > units[best].level and not splits_block(c, start):
                best = c
        return start, best

    def next_start(start, end):
        """Where the chunk after start..end-1 starts: up to cfg.overlap tokens back, at the strongest boundary."""
        budget = cfg.overlap
        if fits(end):
            budget = min(budget, limit - cost(end, block_end[end]))
        lo = end
        while lo - 1 > start and cost(lo - 1, end) <= budget:
            lo -= 1
        return min(range(lo, end), key=lambda i: (-units[i].level, i)) if lo < end else end

    chunks, start, new = [], 0, 0
    while new < n:
        end = grow(start, new + 1)                                     # one new unit at least, even if over
        if end < n:
            start, end = adjust(start, new, end)
        chunks.append((start, end))
        if end == n:
            break
        start, new = next_start(start, end), end
    return chunks


# ---------------------------------------------------------------- text

def render(turns, start_turn, start_char, end_turn, end_char):
    """The text of a span of a session: each turn's piece on its own line, after its speaker's tag."""
    parts = []
    for t in range(start_turn, end_turn + 1):
        if t not in turns:
            continue
        role, content = turns[t]
        piece = content[start_char if t == start_turn else 0:end_char if t == end_turn else len(content)].strip()
        if piece:
            parts.append(f"{TAGS[role]} {piece}")
    return "\n".join(parts)


def chunk_session(session_id, turns, tokenizer, cfg, tag_tokens):
    """The chunk rows of one session, turns being {idx: (role, content)}."""
    units = make_units(turns, tokenizer, cfg)
    ranges = pack(units, cfg, tag_tokens)
    spans = [(units[a].turn, units[a].start, units[b - 1].turn, units[b - 1].end) for a, b in ranges]
    texts = [render(turns, *span) for span in spans]
    counts = [len(e.ids) for e in tokenizer.encode_batch(texts, add_special_tokens=False)]
    ids = [f"{session_id}:{i}" for i in range(len(ranges))]
    return [{"chunk_id": ids[i], "session_id": session_id, "idx": i,
             "prev_chunk_id": ids[i - 1] if i else None, "next_chunk_id": ids[i + 1] if i + 1 < len(ids) else None,
             "start_turn": span[0], "start_char": span[1], "end_turn": span[2], "end_char": span[3],
             "n_tokens": counts[i], "has_question": any(u.question for u in units[a:b]), "text": texts[i]}
            for i, ((a, b), span) in enumerate(zip(ranges, spans))]


def expand(conn, chunk_id, before=1, after=1):
    """A chunk merged with up to `before` chunks in front and `after` behind it: their whole span, no text twice."""
    session_id, idx = conn.execute("SELECT session_id, idx FROM chunks WHERE chunk_id = %s", (chunk_id,)).fetchone()
    first, last = conn.execute("""SELECT min(ARRAY[start_turn, start_char]), max(ARRAY[end_turn, end_char])
                                  FROM chunks WHERE session_id = %s AND idx BETWEEN %s AND %s""",
                               (session_id, idx - before, idx + after)).fetchone()
    turns = {i: (role, content) for i, role, content in conn.execute(
        "SELECT idx, role, content FROM turns WHERE session_id = %s AND idx BETWEEN %s AND %s",
        (session_id, first[0], last[0]))}
    return render(turns, first[0], first[1], last[0], last[1])


# ---------------------------------------------------------------- database

COLUMNS = ("chunk_id", "session_id", "idx", "prev_chunk_id", "next_chunk_id", "start_turn", "start_char",
           "end_turn", "end_char", "n_tokens", "has_question", "text")


def sessions(conn, only=None, skip=frozenset()):
    """Yield (session_id, {idx: (role, content)}) for the split's sessions, streamed in session order."""
    query = "SELECT session_id, idx, role, content FROM turns"
    params = ()
    if only is not None:
        query += " WHERE session_id = ANY(%s)"
        params = (list(only),)
    with conn.cursor(name="turns") as cur:
        cur.execute(query + " ORDER BY session_id, idx", params)
        current, turns = None, {}
        for sid, idx, role, content in cur:
            if sid != current:
                if current is not None and current not in skip:
                    yield current, turns
                current, turns = sid, {}
            turns[idx] = (role, content)
        if current is not None and current not in skip:
            yield current, turns


def connect(dsn, schema, **kwargs):
    """A connection whose search_path is the split's schema (then public, for pgvector's types)."""
    return psycopg.connect(dsn, options=f"-c search_path={schema},public", **kwargs)


def write(conn, dsn, schema, tokenizer, cfg, tag_tokens, batch):
    """Chunk the sessions without chunks and COPY them in, one transaction per batch of sessions. conn streams the
    turns; a second connection writes."""
    done = {r[0] for r in conn.execute("SELECT DISTINCT session_id FROM chunks")}
    rows, n_sessions, n_chunks, tokens, over = [], 0, 0, 0, 0
    with connect(dsn, schema) as writer:

        def flush():
            with writer.transaction(), writer.cursor() as cur, cur.copy(
                    f"COPY chunks ({', '.join(COLUMNS)}) FROM STDIN") as cp:
                for r in rows:
                    cp.write_row([r[c] for c in COLUMNS])
            rows.clear()

        for sid, turns in sessions(conn, skip=done):
            for r in chunk_session(sid, turns, tokenizer, cfg, tag_tokens):
                rows.append(r)
                n_chunks += 1
                tokens += r["n_tokens"]
                over += r["n_tokens"] > cfg.max_tokens
            n_sessions += 1
            if n_sessions % batch == 0:
                flush()
                print(f"  {n_sessions} sessions, {n_chunks} chunks", file=sys.stderr, end="\r")
        if rows:
            flush()
    print(f"  {n_sessions} sessions, {n_chunks} chunks, {tokens / max(n_chunks, 1):.0f} tokens on average, "
          f"{over} over {cfg.max_tokens}", file=sys.stderr)


def show(conn, tokenizer, cfg, tag_tokens, count, seed):
    """Print `count` chunks with a question from random sessions, with their spans and the overlap with the next."""
    ids = sorted(r[0] for r in conn.execute("SELECT DISTINCT session_id FROM turns"))
    random.Random(seed).shuffle(ids)
    shown = 0
    for i in range(0, len(ids), 20):
        for sid, turns in sessions(conn, only=ids[i:i + 20]):
            rows = chunk_session(sid, turns, tokenizer, cfg, tag_tokens)
            for k, r in enumerate(rows):
                if not r["has_question"] or shown == count:
                    continue
                shown += 1
                nxt = rows[k + 1] if k + 1 < len(rows) else None
                print(f"\n{'=' * 100}\n{r['chunk_id']}  ({r['idx'] + 1}/{len(rows)} in its session)  "
                      f"{r['n_tokens']} tokens  turns {r['start_turn']}:{r['start_char']} -> "
                      f"{r['end_turn']}:{r['end_char']}\n{'-' * 100}\n{r['text']}")
                if nxt:
                    print(f"{'-' * 100}\nnext chunk starts at turn {nxt['start_turn']}:{nxt['start_char']}: "
                          f"{nxt['text'][:160]!r}...")
            if shown == count:
                return


def main():
    ap = argparse.ArgumentParser(description="Chunk a split's sessions into its chunks table.")
    ap.add_argument("--split", required=True, help="dev, dev-large, test, final, ...: as loaded by db/load.py")
    ap.add_argument("--dsn", default=DSN, help="default: $DATABASE_URL, else POSTGRES_* from .env")
    ap.add_argument("--tokenizer", default="BAAI/bge-m3", help="Hugging Face tokenizer of the embedding model")
    ap.add_argument("--max-tokens", type=int, default=Config.max_tokens)
    ap.add_argument("--overlap", type=int, default=Config.overlap)
    ap.add_argument("--min-fill", type=float, default=Config.min_fill,
                    help="share of --max-tokens after which a chunk may end early, at a stronger boundary")
    ap.add_argument("--unit-cap", type=int, help="longest unit, in tokens (default: max-tokens - overlap)")
    ap.add_argument("--batch", type=int, default=200, help="sessions per transaction")
    ap.add_argument("--reset", action="store_true", help="delete the split's chunks first, embeddings included")
    ap.add_argument("--show", type=int, metavar="N", help="print N sample chunks with a question; write nothing")
    ap.add_argument("--seed", type=int, default=0, help="sample for --show")
    ap.add_argument("--expand", metavar="CHUNK_ID", help="print a stored chunk merged with its neighbours")
    ap.add_argument("--before", type=int, default=1)
    ap.add_argument("--after", type=int, default=1)
    args = ap.parse_args()
    cfg = Config(args.max_tokens, args.overlap, args.min_fill, args.unit_cap or args.max_tokens - args.overlap)
    if not 0 <= cfg.overlap < cfg.max_tokens or not 0 < cfg.unit_cap <= cfg.max_tokens:
        ap.error("need 0 <= --overlap < --max-tokens and 0 < --unit-cap <= --max-tokens")
    schema = args.split.replace("-", "_")
    with connect(args.dsn, schema) as conn:
        if not conn.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = %s", (schema,)).fetchone():
            ap.error(f"no schema {schema}: load the split first (db/load.py --split {args.split})")
        if args.expand:
            print(expand(conn, args.expand, args.before, args.after))
            return
        tokenizer = Tokenizer.from_pretrained(args.tokenizer)
        tag_tokens = 1 + max(len(tokenizer.encode(t, add_special_tokens=False).ids) for t in TAGS.values())
        if args.show:
            show(conn, tokenizer, cfg, tag_tokens, args.show, args.seed)
            return
        if args.reset:
            conn.execute("TRUNCATE chunks")
            conn.commit()
        write(conn, args.dsn, schema, tokenizer, cfg, tag_tokens, args.batch)
        conn.commit()
        conn.execute("ANALYZE chunks")
        conn.commit()


if __name__ == "__main__":
    main()
