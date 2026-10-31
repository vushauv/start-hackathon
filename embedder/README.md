# embedder

Standalone text embedder for the memory challenge: `BAAI/bge-small-en-v1.5`, 384-dim, L2-normalised (cosine = dot product). It uses the GPU (fp16) if one is available, and otherwise the CPU.

## Setup
```
cd embedder
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Use
```
python embedder.py "user: I graduated with a CS degree"
python embedder.py --query "What degree did I graduate with?"   # questions get the bge query prefix
python embedder.py --in texts.txt --out vecs.json                # one text per line
```
Output: `[{"text": "...", "embedding": [384 floats]}]`

From Python:
```python
from embedder import Embedder
e = Embedder()
docs = e.encode(["user: ... assistant: ..."])
q = e.encode(["What degree did I graduate with?"], query=True)
scores = docs @ q[0]
```

## Why this setup
On a benchmark of 100 dev questions (20 per type), user/assistant turn-pair chunks of about 256 tokens embedded with this model, combined with BM25 using RRF, found 0.966 of the gold sessions in the top 5. Whole-session vectors scored worst (0.856). Embed chunks, not whole sessions, and put questions through `--query`.
