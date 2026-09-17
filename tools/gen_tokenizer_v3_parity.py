#!/usr/bin/env python3
"""Emit Needle 3 tokenizer parity vectors.

The v3 container embeds its own SentencePiece model, and it is NOT the v2
one — the two `tokenizer.model` files differ. This generates the ground
truth from real `sentencepiece` on v3's model, so needle-rs is checked
against the library rather than against upstream's Python re-implementation.

    hf download Cactus-Compute/needle3 tokenizer/tokenizer.model --local-dir weights/v3/
    PYTHONPATH=needle .venv-parity/bin/python tools/gen_tokenizer_v3_parity.py
"""

import json
import pathlib
import sys

from needle.model import export

CONTAINER = pathlib.Path("weights/needle3.cact")
SP_MODEL = pathlib.Path("weights/v3/tokenizer/tokenizer.model")
OUT = pathlib.Path("tests/tokenizer_v3_vectors.json")

CASES = [
    "What's the weather in Paris?",
    "Turn off the bedroom lights",
    "Book a flight from London to JFK tomorrow",
    "Email alice@example.com saying the build is green",
    "<|im_start|>user\n<tools>[]</tools>\nhi<|im_end|>\n<|im_start|>assistant\n",
    '{"name":"get_weather","arguments":{"city":"Paris"}}',
    "set(x) := {1, 2, 3}",
    "   leading and trailing   ",
    "naive cafe -- resume",
    "naïve café — résumé",
    "日本語のテキスト",
    "emoji 🌦️ inside",
    "a" * 200,
    "",
]


def main():
    if not CONTAINER.exists():
        sys.exit(f"missing {CONTAINER}")
    if not SP_MODEL.exists():
        sys.exit(
            f"missing {SP_MODEL} — hf download Cactus-Compute/needle3 "
            "tokenizer/tokenizer.model --local-dir weights/v3/"
        )

    blob = export.read_tokenizer_blob(str(CONTAINER))
    tok = export.parse_tokenizer_blob(blob)

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(SP_MODEL))

    n_pieces = len(tok["pieces"])
    if sp.get_piece_size() != n_pieces:
        sys.exit(
            f"piece-count mismatch: container {n_pieces} vs "
            f"{SP_MODEL} {sp.get_piece_size()} — wrong tokenizer file?"
        )

    cases = [{"text": t, "ids": [int(i) for i in sp.encode(t)]} for t in CASES]

    payload = {
        "pieces": n_pieces,
        "pad_id": tok["pad_id"],
        "blob_bytes": len(blob),
        "specials": [pc for pc in tok["pieces"][:32] if pc.startswith("<")],
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(
        f"wrote {OUT} — {len(cases)} cases, {n_pieces} pieces, "
        f"blob {len(blob)} bytes"
    )


if __name__ == "__main__":
    main()
