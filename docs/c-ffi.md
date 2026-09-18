# C FFI Guide

How to call needle-rs from Python (ctypes), Go (cgo), Swift, or any language with
a C FFI.

## Build the shared library

```bash
cargo build --release -p needle-c
```

Produces:
- `target/release/libneedle_c.so` (Linux)
- `target/release/libneedle_c.dylib` (macOS)
- `target/release/needle_c.dll` (Windows)

The C header is at `crates/needle-c/include/needle.h`, which is the normative
reference — every declaration in it is link-checked against the built library.

## Three model generations, three surfaces

| | Needle v3 | Needle v2 | Needle v1 |
|---|---|---|---|
| Prefix | `needle_v3_*` | `needle_v2_*` | `needle_*` |
| Handle | `NeedleV3Handle` | `NeedleV2Handle` | `NeedleHandle` |
| Load | one `.cact` container | one `.cact` container | `.safetensors` + a vocab file |
| Free handle with | `needle_v3_free` | `needle_v2_free` | `needle_free` |
| Probe heads | confidence only | contrastive + confidence | contrastive only |
| Constrained decode | yes | yes | no |
| Reasoning | emits `<think>` first | no | no |
| Context | 8192 | 2048 | 1024 |

All three live in the same library and can be loaded side by side. They share
exactly two entry points, `needle_free_str` and `needle_last_error`. **Handles
are not interchangeable** — passing a `NeedleV2Handle` to `needle_free` is
undefined behaviour, not a caught error.

v2 and v3 both use the `.cact` extension. A container states its own generation
in its first word, so loading the wrong one **fails** rather than misreading the
header: `needle_v3_load` on a v2 container returns NULL.

## Two rules that cover every binding

1. **Every `char *` returned is yours to free**, with `needle_free_str`. That
   covers `needle_run`, `needle_v2_run`, `needle_v2_run_json`,
   `needle_v2_generate`, every `needle_v3_*` call that returns a string, and all
   three `_run_stream` calls. The `const char *` from
   `needle_last_error` is the exception: it is borrowed, thread-local, valid
   until the next `needle_*` call on that thread, and must not be freed.
2. **A NULL return means failure** — call `needle_last_error()` for the reason.
   One deliberate exception: `needle_v2_run_json` returns NULL — and
   `needle_v3_run_json` an empty string — with *no* error set when the output
   carried no `<tool_call>` markers at all. That is **not**
   the off-topic case: a query no tool fits returns the string `"[]"`, a
   deliberate abstention you should act on. Distinguish the two — NULL is a
   degenerate generation, `"[]"` is the model declining on purpose.

---

## API reference

### Needle v3

```c
// One container carries weights, geometry and tokenizer: no vocab argument.
NeedleV3Handle *needle_v3_load(const char *cact_path);
NeedleV3Handle *needle_v3_load_bytes(const uint8_t *data, size_t len);
void            needle_v3_free(NeedleV3Handle *handle);

// The full completion, reasoning included.
char *needle_v3_run(NeedleV3Handle *h, const char *query, const char *tools_json);
// Just the tool-call payload. "[]" is a considered abstention; "" means no
// <tool_call> markers were emitted at all. Do not collapse the two.
char *needle_v3_run_json(NeedleV3Handle *h, const char *query, const char *tools_json);
// The chain-of-thought inside a completion, or NULL. New in v3.
char *needle_v3_reasoning(NeedleV3Handle *h, const char *text);

char *needle_v3_generate(NeedleV3Handle *h, const char *query, const char *tools_json,
                         size_t max_new_tokens,   // 0 uses the default (256)
                         float  temperature,      // 0 is greedy
                         uint64_t seed,
                         bool   constrain,        // restrict the payload to the schema
                         bool   kv_int8);         // 8-bit key/value cache

char *needle_v3_run_stream(NeedleV3Handle *h, const char *query, const char *tools_json,
                           void (*cb)(const char *piece, void *userdata), void *userdata);

bool needle_v3_has_confidence(NeedleV3Handle *h);
bool needle_v3_confidence_for(NeedleV3Handle *h, const char *query,
                              const char *tools_json, const char *completion,
                              float *out);

size_t needle_v3_kv_bytes(NeedleV3Handle *h, size_t seq_len, bool kv_int8);
size_t needle_v3_max_seq_len(NeedleV3Handle *h);
```

Three things differ from v2 and will bite if you assume otherwise.

**v3 reasons before answering.** `needle_v3_run` returns a `<think>` block
followed by the call. Use `needle_v3_run_json` for the payload alone, or
`needle_v3_reasoning` to pull the reasoning out of a completion you already
have.

**`needle_v3_confidence_for` takes the completion, not the query.** The head
scores a finished judgement. On the shipped checkpoint a correct call scores
0.94, a wrong one 0.26 — and a bare query scores 0.80, which looks like a
confident answer and is not one. v2's head collapsed to near zero on a bare
query, so that misuse announced itself; v3's does not.

**There is no `needle_v3_retrieve_tools`.** v3 exports a confidence head and
nothing else, so retrieval has no v3 equivalent. The symbol is absent rather
than present-and-always-empty: a missing symbol is a compile error at the call
site, while one that always fails is a runtime mystery.

#### The key/value cache

v3's context is 8192 and its cache is correspondingly large, so the caller —
who usually owns the memory budget — can ask before committing:

```c
size_t f32  = needle_v3_kv_bytes(h, 8192, false);  // 42.0 MB
size_t int8 = needle_v3_kv_bytes(h, 8192, true);   // 11.2 MB
```

Passing `kv_int8 = true` to `needle_v3_generate` stores the cache at 8 bits —
the width the container declares in `kv_bits` — for roughly a quarter of the
memory at about 2% of the speed. It is not bit-identical to the default, so it
stays opt-in; on our tests it produces identical tool calls.

### Needle v2

```c
// One container carries weights, geometry and tokenizer: no vocab argument.
NeedleV2Handle *needle_v2_load(const char *cact_path);
NeedleV2Handle *needle_v2_load_bytes(const uint8_t *data, size_t len);

// Full model output, <tool_call> markers included.
char *needle_v2_run(NeedleV2Handle *h, const char *query, const char *tools_json);

// Just the payload: "[{...}]", or "[]" when no tool applies.
// NULL with no error set = no <tool_call> markers were emitted at all.
char *needle_v2_run_json(NeedleV2Handle *h, const char *query, const char *tools_json);

// temperature <= 0 is greedy; constrain != 0 restricts output to the schema.
char *needle_v2_generate(NeedleV2Handle *h, const char *query, const char *tools_json,
                         size_t max_new_tokens, float temperature,
                         uint64_t seed, int constrain);

typedef void (*NeedleStreamCallback)(uint32_t token_id, const char *piece, void *user_data);
char *needle_v2_run_stream(NeedleV2Handle *h, const char *query, const char *tools_json,
                           NeedleStreamCallback callback, void *user_data);

// Confidence: probability in (0,1) that `completion` is the right answer.
bool needle_v2_confidence_for(NeedleV2Handle *h, const char *query,
                              const char *tools_json, const char *completion,
                              float *out);
bool needle_v2_confidence(NeedleV2Handle *h, const char *text, float *out);  // raw logit

// Contrastive retrieval.
size_t needle_v2_contrastive_dim(NeedleV2Handle *h);  // 0 = no head
bool   needle_v2_encode_contrastive(NeedleV2Handle *h, const char *text,
                                    float *out, size_t dim);
size_t needle_v2_retrieve_tools(NeedleV2Handle *h, const char *query,
                                const char **tool_descs, size_t n_tools, size_t top_k,
                                size_t *out_indices, float *out_scores);

void needle_v2_free(NeedleV2Handle *h);
```

Tool schemas are compacted internally, so pretty-printed JSON and its minified
form give byte-identical output. The model was trained on compact schemas; before
this normalisation, indented input measurably degraded tool selection.

### Needle v1

```c
NeedleHandle *needle_load(const char *weights_path, const char *vocab_path);
NeedleHandle *needle_load_bytes(const uint8_t *weights_data, size_t weights_len,
                                const uint8_t *vocab_data,   size_t vocab_len);

char *needle_run(NeedleHandle *h, const char *query, const char *tools_json);
char *needle_run_stream(NeedleHandle *h, const char *query, const char *tools_json,
                        NeedleStreamCallback callback, void *user_data);

size_t needle_contrastive_dim(NeedleHandle *h);  // 0 = no head
bool   needle_encode_contrastive(NeedleHandle *h, const char *text, float *out, size_t dim);
size_t needle_retrieve_tools(NeedleHandle *h, const char *query,
                             const char **tool_descs, size_t n_tools, size_t top_k,
                             size_t *out_indices, float *out_scores);

void needle_free(NeedleHandle *h);
```

v1 post-processes its output: the `<tool_call>` marker is stripped and the
caller's original tool-name casing is restored. The tokens delivered to a
streaming callback are therefore a progress view, and the returned string is the
answer — **they differ**, and the returned string is the one to parse.

### Shared

```c
void        needle_free_str(char *s);      // frees any returned char*; NULL-safe
const char *needle_last_error(void);       // borrowed, thread-local, do not free
```

---

## Confidence gating

The confidence head scores a *judgement already made*: it is trained on the
formatted prompt followed by a completion. `needle_v2_confidence_for` assembles
that input for you — pass the string `needle_v2_run` returned.

```c
char *out = needle_v2_run(h, query, tools);
float p;
if (needle_v2_confidence_for(h, query, tools, out, &p) && p >= 0.5f) {
    execute(out);        // above your threshold: act
} else {
    escalate(query);     // below it: re-ask, or route to a larger model
}
needle_free_str(out);
```

Pick the threshold per product. Two caveats worth knowing before you gate on it:

- Passing a bare query to the lower-level `needle_v2_confidence` reads near zero
  however answerable the query is, because the completion is what is being
  scored. That is the intended behaviour of the primitive, not a bug.
- Upstream's published score is the *minimum* of this head and the decode
  probability of the call tokens. needle-rs exposes the head only; the
  composition is not replicated because the reference implementation is not
  published. Calibration also holds for the base model only.

---

## Python (ctypes)

`examples/python-via-cffi/infer.py` is a complete program covering both
generations. The essentials:

```python
import ctypes

lib = ctypes.CDLL("target/release/libneedle_c.so")   # .dylib on macOS

# Declare returned strings as c_void_p, NOT c_char_p. ctypes converts a
# c_char_p result into a Python bytes object and discards the original
# pointer, so needle_free_str would be handed Python-owned memory and abort.
lib.needle_v2_load.restype  = ctypes.c_void_p
lib.needle_v2_load.argtypes = [ctypes.c_char_p]
lib.needle_v2_run.restype   = ctypes.c_void_p
lib.needle_v2_run.argtypes  = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
lib.needle_free_str.argtypes = [ctypes.c_void_p]
lib.needle_v2_free.argtypes  = [ctypes.c_void_p]
lib.needle_last_error.restype = ctypes.c_char_p

def take(ptr):
    """Copy a returned string, then free the original."""
    if not ptr:
        return None
    text = ctypes.cast(ptr, ctypes.c_char_p).value.decode()
    lib.needle_free_str(ptr)
    return text

handle = lib.needle_v2_load(b"weights/needle2.cact")
if not handle:
    raise RuntimeError(lib.needle_last_error().decode())

tools = b'[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]'
print(take(lib.needle_v2_run(handle, b"What's the weather in Paris?", tools)))
lib.needle_v2_free(handle)
```

For v1, swap in `needle_load(weights, vocab)` / `needle_run` / `needle_free`.

---

## Go (cgo)

```go
package main

/*
#cgo LDFLAGS: -L../../target/release -lneedle_c -lm
#include "../../crates/needle-c/include/needle.h"
#include <stdlib.h>
*/
import "C"
import (
    "fmt"
    "unsafe"
)

func main() {
    path := C.CString("weights/needle2.cact")
    defer C.free(unsafe.Pointer(path))

    handle := C.needle_v2_load(path)
    if handle == nil {
        panic(C.GoString(C.needle_last_error()))
    }
    defer C.needle_v2_free(handle)

    q := C.CString("What's the weather in Paris?")
    t := C.CString(`[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]`)
    defer C.free(unsafe.Pointer(q))
    defer C.free(unsafe.Pointer(t))

    result := C.needle_v2_run(handle, q, t)
    if result == nil {
        panic(C.GoString(C.needle_last_error()))
    }
    defer C.needle_free_str(result)
    fmt.Println(C.GoString(result))

    // Gate on confidence before acting on the call.
    var p C.float
    if C.needle_v2_confidence_for(handle, q, t, result, &p) {
        fmt.Printf("confidence %.1f%%\n", float32(p)*100)
    }
}
```

`C.GoString` copies, so freeing `result` afterwards is correct.

---

## Swift

```swift
import Foundation

typealias NeedleV2Handle = OpaquePointer

@_silgen_name("needle_v2_load")
func needle_v2_load(_ path: UnsafePointer<CChar>) -> NeedleV2Handle?

@_silgen_name("needle_v2_run")
func needle_v2_run(_ h: NeedleV2Handle, _ query: UnsafePointer<CChar>,
                   _ tools: UnsafePointer<CChar>) -> UnsafeMutablePointer<CChar>?

@_silgen_name("needle_v2_confidence_for")
func needle_v2_confidence_for(_ h: NeedleV2Handle, _ query: UnsafePointer<CChar>,
                              _ tools: UnsafePointer<CChar>,
                              _ completion: UnsafePointer<CChar>,
                              _ out: UnsafeMutablePointer<Float>) -> Bool

@_silgen_name("needle_free_str")
func needle_free_str(_ s: UnsafeMutablePointer<CChar>?)

@_silgen_name("needle_v2_free")
func needle_v2_free(_ h: NeedleV2Handle)

@_silgen_name("needle_last_error")
func needle_last_error() -> UnsafePointer<CChar>?

guard let handle = needle_v2_load("weights/needle2.cact") else {
    fatalError(needle_last_error().map { String(cString: $0) } ?? "load failed")
}
defer { needle_v2_free(handle) }

let query = "What's the weather in Paris?"
let tools = #"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#

guard let raw = needle_v2_run(handle, query, tools) else {
    fatalError(needle_last_error().map { String(cString: $0) } ?? "inference failed")
}
let output = String(cString: raw)      // copies
needle_free_str(raw)
print(output)

var p: Float = 0
if needle_v2_confidence_for(handle, query, tools, output, &p) {
    print(String(format: "confidence %.1f%%", p * 100))
}
```

Build with `swiftc main.swift -L target/release -lneedle_c -o demo`.

---

## Thread safety

No handle type is `Send`: do not share one across threads. Create a handle per
thread, or add external locking. A handle holds a KV cache sized to the session,
so per-thread handles cost memory — for v2 see the KV-ring note in
[ARCHITECTURE.md](../ARCHITECTURE.md); for v3 ask `needle_v3_kv_bytes`, which is
8.8 MB at 512 positions and 42.0 MB at the full context, or 2.3 and 11.2 MB with
`kv_int8`.

`needle_last_error()` uses thread-local storage and is safe to call concurrently;
each thread sees only its own last error.
