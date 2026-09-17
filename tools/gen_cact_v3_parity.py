#!/usr/bin/env python3
"""Emit Needle 3 container parity vectors from upstream's own reader.

Upstream is the spec: whatever `export.read_export` says the container
declares is what needle-rs must agree with, field for field. The directory
is nameless and positional, so a correct tensor in the wrong slot passes
every per-tensor check — agreeing on geometry and on every record is what
makes the canon checkable at all.

    PYTHONPATH=needle .venv-parity/bin/python tools/gen_cact_v3_parity.py
"""

import json
import pathlib
import struct
import sys

from needle.model import export

CONTAINER = pathlib.Path("weights/needle3.cact")
OUT = pathlib.Path("tests/cact_v3_vectors.json")


def main():
    if not CONTAINER.exists():
        sys.exit(
            f"missing {CONTAINER} — hf download Cactus-Compute/needle3 "
            "needle3.cact --local-dir weights/"
        )

    # These are private upstream names. Pinned reference: dd85774 "Needle 3
    # Live". If upstream renames them, fail with a diagnosis rather than a
    # bare AttributeError three frames down.
    for attr in ("_REC_FMT", "_HDR_FMT"):
        if not hasattr(export, attr):
            sys.exit(
                f"upstream export.py no longer defines {attr}; the vendored "
                "reference has moved past dd85774 and this generator needs "
                "updating alongside the Rust header parser"
            )

    meta, _tensors = export.read_export(str(CONTAINER))
    raw = CONTAINER.read_bytes()

    n_rec = meta["num_tensors"]
    rec_size = struct.calcsize(export._REC_FMT)
    hdr_size = struct.calcsize(export._HDR_FMT)
    codebook_len = len(meta["codebook"])
    dir_start = hdr_size + codebook_len * 4

    records = []
    for i in range(n_rec):
        off = dir_start + i * rec_size
        (dtype, ndim, _pad, s0, s1, s2, s3, offset, nbytes, group, bits) = struct.unpack_from(
            export._REC_FMT, raw, off
        )
        records.append(
            {
                "index": i,
                "dtype": dtype,
                "ndim": ndim,
                "shape": [s0, s1, s2, s3][:ndim],
                "offset": offset,
                "nbytes": nbytes,
                "group": group,
                "bits": bits,
            }
        )

    geometry = {}
    for k, v in meta.items():
        if k == "codebook":
            continue
        geometry[k] = list(v) if isinstance(v, tuple) else v
    geometry["rope_theta"] = float(geometry["rope_theta"])

    payload = {
        "container_bytes": len(raw),
        "header_bytes": hdr_size,
        "record_bytes": rec_size,
        "codebook_len": codebook_len,
        "geometry": geometry,
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(
        f"wrote {OUT} — {n_rec} records, header {hdr_size} B, "
        f"record {rec_size} B, container {len(raw)} B"
    )


if __name__ == "__main__":
    main()
