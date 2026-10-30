Vendored from https://github.com/RxAI-dev/memory-challenge at commit `f0a317dfa13e7797baae61711cef27a9805bedcc` (`main`).

| File | Role |
| --- | --- |
| `client.py` | WebSocket client. Reads `REACTOR_ACCESS_KEY` from the environment. |
| `predictions.py` | Converts client result lines into `predictions.jsonl`. |
| `score.py` | Official dev scorer (standard library only). |

These files are the participant kit, copied unchanged so `memtest` can call them. Do not put access keys in this tree.
