# HuggingFace Model Card

**Filename:** `README.md` in the `Abdalrahman/needle-rs-safetensors` HuggingFace repository.

Paste this content verbatim into the model card when uploading the converted weights.

---

# needle-rs-safetensors

> **Attribution:** This repository contains format-converted weights for
> **[Needle](https://github.com/cactus-compute/needle)** by
> [Cactus Compute](https://github.com/cactus-compute) (Henry Ndubuaku et al., 2026).
> All model training, architecture design, and original weights are their work,
> released under the MIT License.
>
> This repo packages the weights in SafeTensors INT4-packed format for use with the
> **[needle-rs](https://github.com/geekgineer/needle-rs)** Rust runtime.
> If you use this model, please also cite the original Needle repository.

## What is in this repository

| File | Size | Description |
|---|---|---|
| `needle.safetensors` | 22 MB | INT4 quantized weights + BF16 norms, SafeTensors format |
| `vocab.txt` | 120 KB | 8,192 SentencePiece vocabulary pieces (TSV: piece TAB score) |

## Model description

Needle is a 26M-parameter encoder-decoder transformer for single-shot tool/function calling,
trained by Cactus Compute. See the [original repository](https://github.com/cactus-compute/needle)
for full model details, training procedure, and research context.

**Architecture summary:**
- d_model=512, 12 encoder layers, 8 decoder layers
- 8 query heads / 4 KV heads (GQA, repeat=2)
- Vocabulary size: 8,192
- Max encoder length: 1,024 tokens

## Weight format

The SafeTensors file uses a custom `I4` dtype for quantized attention kernels:

- Group-wise INT4: `group_size=32`, scale=`max|w|/7`, nibble-packed
- Non-kernel parameters (norms, gates, embeddings) stored in BF16
- Model config (d_model, num_heads, etc.) embedded in `__metadata__`

This format is consumed directly by `needle-rs`. See
[ARCHITECTURE.md](https://github.com/geekgineer/needle-rs/blob/main/ARCHITECTURE.md)
for the full weight format specification.

## Usage

### With needle-rs CLI

```bash
# Download
huggingface-cli download Abdalrahman/needle-rs-safetensors \
  needle.safetensors vocab.txt --local-dir weights/

# Run
./needle-rs weights/needle.safetensors weights/vocab.txt \
  "What is the weather in Paris?" \
  '[{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}]'
```

### With needle-rs Rust API

```rust
use needle_infer::NeedleEngine;

let engine = NeedleEngine::load("weights/needle.safetensors", "weights/vocab.txt")?;
let result = engine.run("What is the weather in Paris?", tools_json);
println!("{}", result.text);
```

### With needle-rs WASM (browser)

```js
import init, { NeedleWasm } from "./pkg/needle_wasm.js";
await init();
const resp = await fetch("https://huggingface.co/Abdalrahman/needle-rs-safetensors/resolve/main/needle.safetensors");
const engine = NeedleWasm.load(new Uint8Array(await resp.arrayBuffer()), vocabText);
```

Live demo: [needle-rs.pages.dev](https://needle-rs.pages.dev)

## License

The original Needle model weights are released under the MIT License by Cactus Compute.
This repository (format conversion and packaging) is also MIT.

## Citation

If you use this model, please cite the original Needle paper/repository:

```bibtex
@software{needle2026,
  author  = {Ndubuaku, Henry and {Cactus Compute}},
  title   = {Needle: A 26M-Parameter Tool-Calling Transformer},
  year    = {2026},
  url     = {https://github.com/cactus-compute/needle},
  license = {MIT}
}
```

And optionally cite this runtime:

```bibtex
@software{needlers2026,
  author  = {Ibrahim, Abdalrahman},
  title   = {needle-rs: Pure-Rust + WASM Runtime for Needle},
  year    = {2026},
  url     = {https://github.com/geekgineer/needle-rs},
  license = {MIT}
}
```

## Tags

`rust` `wasm` `webassembly` `on-device` `function-calling` `tool-calling` `quantized` `int4` `safetensors` `no-server` `edge-ai`
