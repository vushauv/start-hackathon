"""Standalone text embedder (BAAI/bge-small-en-v1.5, 384-dim, L2-normalised).

Usage:
  python embedder.py "some text" "another text"          # passages
  python embedder.py --query "What degree did I graduate with?"
  echo "text" | python embedder.py                        # one text per stdin line
  python embedder.py --in texts.txt --out vecs.json
Output: JSON [{"text": ..., "embedding": [384 floats]}]
"""
import argparse, json, sys

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model=MODEL, device=None, batch_size=128):
        import torch
        from sentence_transformers import SentenceTransformer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = SentenceTransformer(model, device=device)
        if device == "cuda":
            self.m.half()
        self.m.max_seq_length = 512
        self.device, self.bs = device, batch_size

    def encode(self, texts, query=False):
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        return self.m.encode(list(texts), batch_size=self.bs, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype("float32")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--query", action="store_true", help="add the bge query prefix (use for questions)")
    ap.add_argument("--in", dest="inp", help="file with one text per line")
    ap.add_argument("--out", help="write JSON here instead of stdout")
    ap.add_argument("--device", choices=["cuda", "cpu"])
    a = ap.parse_args()
    texts = list(a.texts)
    if a.inp:
        texts += [l.rstrip("\n") for l in open(a.inp, encoding="utf-8") if l.strip()]
    if not texts and not sys.stdin.isatty():
        texts += [l.rstrip("\n") for l in sys.stdin if l.strip()]
    if not texts:
        ap.error("no input text")
    vecs = Embedder(device=a.device).encode(texts, query=a.query)
    res = [{"text": t, "embedding": [round(float(x), 6) for x in v]} for t, v in zip(texts, vecs)]
    s = json.dumps(res)
    open(a.out, "w").write(s) if a.out else print(s)


if __name__ == "__main__":
    main()
