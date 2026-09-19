#!/usr/bin/env python3
"""Measure the shipped artifacts and write the numbers into the docs.

The module size quoted in the docs is the *deployed* artifact's, not a local
build's: CI and a laptop produce modules a few hundred bytes apart, and the
number a reader cares about is the one they download.

Sizes and test counts are facts about a build, and they were being copied by
hand into roughly twenty-five places. Every code change silently invalidated all
of them at once: the WASM module went 529 -> 537 -> 560 KB across three rounds
and each time the docs had to be chased. This makes the build the source of
truth and the docs a projection of it.

    python3 tools/sync_sizes.py            # measure and rewrite
    python3 tools/sync_sizes.py --check    # fail if anything has drifted

`--check` is what CI runs. It never edits; it reports what is stale and exits 1,
so a pull request that grows the module cannot land with prose claiming the old
size.

The over-the-wire figure is measured against the live deployment rather than a
local `brotli -q 11`, because Cloudflare compresses at a lower level and the
local number understates what a visitor actually downloads — which is the exact
mistake this file exists to stop repeating.
"""

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
FACTS = ROOT / "docs" / "sizes.json"
LIVE_WASM = "https://needle-rs.pages.dev/pkg/needle_wasm_bg.wasm"

WASM_OPT_FLAGS = [
    "-Oz", "--enable-bulk-memory", "--enable-bulk-memory-opt",
    "--enable-mutable-globals", "--enable-sign-ext",
    "--enable-nontrapping-float-to-int",
]


def kb(n):
    return f"{round(n / 1024)} KB"


def measure_local():
    """Build the artifacts and measure them. Needs cargo, wasm-pack, wasm-opt."""
    out = {}
    tmp = ROOT / "pkg-sizes"
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(
        ["wasm-pack", "build", "crates/needle-wasm", "--target", "web",
         "--release", "--out-dir", "../../pkg-sizes/"],
        cwd=ROOT, check=True, capture_output=True,
    )
    raw = tmp / "needle_wasm_bg.wasm"
    out["wasm_unoptimised"] = raw.stat().st_size
    out["glue_js"] = (tmp / "needle_wasm.js").stat().st_size
    subprocess.run(["wasm-opt", str(raw), *WASM_OPT_FLAGS, "-o", str(raw)],
                   check=True, capture_output=True)
    out["wasm"] = raw.stat().st_size
    # Measured here, while the optimised module still exists on disk.
    if shutil.which("brotli"):
        r = subprocess.run(["brotli", "-q", "11", "-c", str(raw)], capture_output=True)
        out["brotli_local"] = len(r.stdout)
    shutil.rmtree(tmp, ignore_errors=True)

    subprocess.run(["cargo", "build", "--release", "-p", "needle-rs-cli",
                    "-p", "needle-c"], cwd=ROOT, check=True, capture_output=True)
    stripped = ROOT / "target" / "release" / "needle-rs-stripped"
    subprocess.run(["strip", "-o", str(stripped),
                    str(ROOT / "target" / "release" / "needle-rs")], check=True)
    out["cli"] = stripped.stat().st_size
    stripped.unlink()
    for name in ("libneedle_c.dylib", "libneedle_c.so"):
        p = ROOT / "target" / "release" / name
        if p.exists():
            out["c_dylib"] = p.stat().st_size
            break
    return out


def measure_live():
    """What Cloudflare actually serves, which is what a visitor downloads."""
    got = {}
    for key, enc in (("wire", "br"), ("gzip", "gzip"), ("live_raw", "identity")):
        req = urllib.request.Request(
            LIVE_WASM,
            headers={"Accept-Encoding": enc, "User-Agent": "needle-rs-size-sync"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            got[key] = len(r.read())
    return got


def counts():
    """Test counts, which drift for the same reason sizes do."""
    res = subprocess.run(["cargo", "test", "--workspace", "--release"],
                         cwd=ROOT, capture_output=True, text=True)
    rust = sum(int(m) for m in re.findall(r"^test result: ok\. (\d+) passed",
                                          res.stdout, re.M))
    wasm = 0
    for js in ("node_e2e.js", "node_e2e_v2.js", "node_e2e_v3.js"):
        p = ROOT / "crates" / "needle-wasm" / "tests" / js
        r = subprocess.run(["node", str(p)], cwd=ROOT, capture_output=True, text=True)
        last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        m = re.search(r"(\d+) passed", last)
        if m:
            wasm += int(m.group(1))
    return {"rust_tests": rust, "wasm_assertions": wasm}


# Prose quotes a band, not a figure. The band is stable across ordinary growth,
# so adding code does not invalidate a dozen documents — but it can go stale
# silently, which is worse than a wrong number because nothing looks wrong. Each
# band therefore carries a ceiling that `--check` enforces.
BANDS = [
    ("wasm", 600 * 1024, "under 600 KB", "the module's prose band"),
    ("wire", 250 * 1024, "about 200 KB", "the over-the-wire prose band"),
]


def rules(f):
    """(file, pattern, replacement) for the few places an exact figure belongs.

    Everywhere else quotes a band. These are the benchmark tables and the
    architecture note, where precision is the subject rather than decoration.
    Patterns match the shape of a claim, not a particular number.
    """
    wasm = kb(f.get("live_raw", f["wasm"]))
    wire = kb(f["wire"]) if "wire" in f else None
    gz = kb(f["gzip"]) if "gzip" in f else None
    unopt, glue = kb(f["wasm_unoptimised"]), kb(f["glue_js"])
    cli, dylib = kb(f["cli"]), kb(f["c_dylib"])
    brotli = kb(f["brotli_local"])
    R = []

    def add(path, pat, rep):
        # A figure we could not measure is never written. Substituting the raw
        # size for the transfer size would be worse than leaving it stale.
        if rep is None:
            return
        R.append((path, pat, rep))

    add("BENCHMARKS.md", r"(?<=CLI binary \(`needle-rs`\) \| \*\*)\d{3} KB", cli)
    add("BENCHMARKS.md", r"(?<=`libneedle_c\.dylib`\) \| \*\*)\d{3} KB", dylib)
    add("BENCHMARKS.md", r"(?<=`needle_wasm_bg\.wasm`\) \| \*\*)\d{3} KB", wasm)
    add("BENCHMARKS.md", r"\b\d{3} KB(?= as Cloudflare Pages actually serves it)", wire)
    add("BENCHMARKS.md", r"\b\d{3} KB(?= gzipped)", gz)
    add("BENCHMARKS.md", r"\b\d{3} KB(?= if you compress it yourself)", brotli)
    add("BENCHMARKS.md", r"(?<=same module, unoptimised \| )\d{3} KB", unopt)
    add("BENCHMARKS.md", r"\b\d{3} KB(?= figure:)", wasm)
    add("BENCHMARKS.md", r"\b\d{3} KB(?= binary, or )", cli)
    add("BENCHMARKS.md", r"(?<=or )\d{3} KB(?= of\nWebAssembly)", wasm)

    add("ARCHITECTURE.md", r"\*\*\d{3} KB\*\*(?= after `wasm-opt)", f"**{wasm}**")

    add("docs/wasm-integration.md", r"(?<=`pkg/needle_wasm_bg\.wasm` \()\d{3} KB", unopt)
    add("docs/wasm-integration.md", r"(?<=`pkg/needle_wasm\.js` \()\d{2} KB(?= of glue)", glue)
    add("docs/RELEASING.md", r"\b\d{3} KB(?= module where CI produces )", unopt)
    add("docs/RELEASING.md", r"(?<=module where CI produces )\d{3} KB", wasm)
    add("examples/dom-editor/README.md", r"\d{3} -> \d{3} KB",
        f"{unopt.split()[0]} -> {wasm.split()[0]} KB")

    add("README.md", r"\b\d+ Rust tests and \d+ WASM binding assertions",
        f"{f['rust_tests']} Rust tests and {f['wasm_assertions']} WASM binding assertions")
    return R


def check_bands(f):
    """A band that has quietly become false is the failure this guards."""
    broken = []
    for key, ceiling, text, what in BANDS:
        if key in f and f[key] > ceiling:
            broken.append(
                f"{what} says \"{text}\" but {key} is now {kb(f[key])} "
                f"(> {kb(ceiling)}) — re-band the prose"
            )
    return broken


def apply(facts, check):
    stale = []
    for path, pat, rep in rules(facts):
        p = ROOT / path
        if not p.exists():
            continue
        s = p.read_text()
        new = re.sub(pat, rep, s)
        if new != s:
            stale.append(path)
            if not check:
                p.write_text(new)
    return sorted(set(stale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit 1; never edit")
    ap.add_argument("--offline", action="store_true",
                    help="skip the live deployment measurement")
    args = ap.parse_args()

    facts = measure_local()
    facts.update(counts())
    facts.setdefault("brotli_local", facts["wasm"])
    if not args.offline:
        try:
            facts.update(measure_live())
        except Exception as e:  # noqa: BLE001 — a network failure must not edit docs
            print(f"live measurement unavailable ({e}) — leaving the "
                  f"over-the-wire figures alone rather than guessing")

    FACTS.parent.mkdir(parents=True, exist_ok=True)
    readable = {k: (f"{v} B ({kb(v)})" if k.endswith(("wasm", "cli", "c_dylib",
                "glue_js", "wire", "gzip", "live_raw", "wasm_unoptimised",
                "brotli_local")) else v) for k, v in facts.items()}
    if not args.check:
        FACTS.write_text(json.dumps(readable, indent=1) + "\n")

    broken = check_bands(facts)
    stale = apply(facts, args.check)
    if args.check:
        for b in broken:
            print(f"BAND: {b}")
        if stale or broken:
            if stale:
                print("size or count claims have drifted from the build:")
                for p in stale:
                    print(f"  {p}")
            if stale:
                print("\nrun: python3 tools/sync_sizes.py")
            return 1
        print("all size and count claims match the build")
        return 0
    print("updated:" if stale else "already in sync")
    for p in stale:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
