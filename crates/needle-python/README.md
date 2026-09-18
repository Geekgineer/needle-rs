<div align="center">
  <img src="https://raw.githubusercontent.com/Geekgineer/needle-rs/main/assets/banner.svg" alt="needle-rs" width="100%"/>
</div>

<div align="center">
  <a href="https://needle-rs.pages.dev"><b>Live demo</b></a> ·
  <a href="https://github.com/Geekgineer/needle-rs">GitHub</a> ·
  <a href="https://github.com/Geekgineer/needle-rs/blob/main/ARCHITECTURE.md">Architecture</a>
</div>

# needle-rs

Local tool calling with no server, no API key and no GPU. A compiled Rust
extension — importing it costs milliseconds, not the seconds a JAX or PyTorch
import takes, and it pulls in no Python ML dependencies at all.

This is the Python binding for
[needle-rs](https://github.com/Geekgineer/needle-rs), a pure-Rust runtime for
[Cactus Compute's](https://github.com/cactus-compute/needle) Needle tool-calling
models. All three model generations are supported, and output is verified token-exact
against the upstream JAX reference.

```bash
pip install needle-rs
```

## Needle v2

One `.cact` file carries the weights, the geometry and the tokenizer.

```python
from needle_rs import V2Engine

engine = V2Engine.load("weights/needle2.cact")

tools = """[{"name":"get_weather","description":"Get current weather for a city",
             "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                           "required":["city"]}}]"""
query = "What's the weather in Paris?"

out = engine.run(query, tools)
# <tool_call>[{"name":"get_weather","arguments":{"city":"Paris"}}]</tool_call>

# Gate execution on the model's confidence in the answer it just gave.
p = engine.confidence_for(query, tools, out)
if p is not None and p >= 0.5:
    ...   # act on the call
else:
    ...   # escalate
```

The confidence head scores a judgement already made, so `confidence_for` needs
the completion. Passing a bare query to the lower-level `confidence()` reads
near zero however answerable the query is.

Grammar-constrained decoding restricts the payload to the declared schema —
valid tool names and argument keys only:

```python
engine.generate(query, tools, max_new_tokens=96, temperature=0.0, constrain=True)
```

Temperature above zero samples, and `seed` makes that reproducible.

## Needle v1

```python
from needle_rs import NeedleEngine

engine = NeedleEngine.load("weights/needle.safetensors", "weights/vocab.txt")
result = engine.run("Book a flight from London to JFK tomorrow", tools)

engine.run_stream(query, tools, lambda tid, piece: print(piece, end="", flush=True))
engine.run_batch([(query1, tools1), (query2, tools2)])
```

v1 post-processes its output: the `<tool_call>` marker is stripped and your
original tool-name casing restored, so with `run_stream` the streamed pieces are
a progress view and the **returned** string is the answer.

## Tool retrieval

Needle 2 and Needle 1 carry a contrastive head for narrowing a large catalogue
before routing. Embeddings are L2-normalised, so similarity is a plain dot
product. **The published Needle 3 weights do not** — they export only a
confidence head, so `V3Engine` has no `retrieve_tools`. Needle 3's architecture
defines an embedding head; it is not in this checkpoint.

```python
engine.retrieve_tools(
    "What's the weather in Paris?",
    ["Get current weather for a city", "Book a flight", "Send an email"],
    top_k=2,
)
# [(0, 0.897), (2, 0.547)]  — (index, score), descending
```

## Weights

Weights are not bundled — download them once:

| Version | Files | Size | Source |
|---|---|---|---|
| v2 | `needle2.cact` | 13.7 MB | [`Cactus-Compute/needle2`](https://huggingface.co/Cactus-Compute/needle2) |
| v1 | `needle.safetensors` + `vocab.txt` | 22 MB + 122 KB | [`Abdalrahman/needle-rs-safetensors`](https://huggingface.co/Abdalrahman/needle-rs-safetensors) |

```python
from huggingface_hub import hf_hub_download
cact = hf_hub_download("Cactus-Compute/needle2", "needle2.cact")
```

## Notes

- Needle is a tool-calling router, not a chat model: one query plus tool
  definitions in, one JSON call out. It will not produce useful free-form text.
- Single-shot. No multi-turn dialogue and no reasoning over tool results — your
  application executes the call and decides what to do with the response.
- English-trained; multilingual behaviour is not evaluated upstream.
- Constrained decoding guarantees syntactic validity, not semantic correctness.

## Credit and license

This package is MIT. The **models** — architecture, training and weights — are
the work of [Cactus Compute](https://github.com/cactus-compute/needle) and carry
their own terms: Needle 3's weights are **Apache-2.0**, Needle 2's and Needle
1's are MIT, and the upstream repository is Apache-2.0. Check the licence on the
generation you ship. If you publish work using them, please cite Needle
([arXiv:2607.18363](https://arxiv.org/abs/2607.18363)); the entry is in the
[repository README](https://github.com/Geekgineer/needle-rs#citation).

This package is the runtime only.
