# C FFI Guide

How to call needle-rs from Python (ctypes), Go (cgo), Swift, or any language with a C FFI.

## Build the shared library

```bash
cargo build --release -p needle-c
```

Produces:
- `target/release/libneedle_c.so` (Linux)
- `target/release/libneedle_c.dylib` (macOS)
- `target/release/needle_c.dll` (Windows)

The C header is at `crates/needle-c/include/needle.h`.

---

## API reference

```c
// Load model from disk. Returns null on failure; use needle_last_error().
NeedleHandle *needle_load(const char *weights_path, const char *vocab_path);

// Load model from in-memory buffers (for embedded use / no filesystem).
NeedleHandle *needle_load_bytes(
    const uint8_t *weights_data, size_t weights_len,
    const uint8_t *vocab_data,   size_t vocab_len);

// Run inference. Returns heap-allocated JSON string; free with needle_free_str().
char *needle_run(NeedleHandle *handle, const char *query, const char *tools_json);

// Run with per-token streaming callback.
typedef void (*NeedleStreamCallback)(uint32_t token_id, const char *piece, void *user_data);
char *needle_run_stream(
    NeedleHandle *handle, const char *query, const char *tools_json,
    NeedleStreamCallback callback, void *user_data);

// Contrastive embedding. Returns false if no head or dim mismatch.
bool   needle_encode_contrastive(NeedleHandle *handle, const char *text, float *out, size_t dim);
size_t needle_contrastive_dim(NeedleHandle *handle);  // 0 = no head

// Rank tool descriptions. Returns number of results written.
size_t needle_retrieve_tools(
    NeedleHandle *handle, const char *query,
    const char **tool_descs, size_t n_tools, size_t top_k,
    size_t *out_indices, float *out_scores);

// Free memory.
void needle_free_str(char *s);      // free string from needle_run / needle_run_stream
void needle_free(NeedleHandle *h);  // free handle from needle_load / needle_load_bytes

// Last error message (thread-local, valid until next needle_* call). Do not free.
const char *needle_last_error();
```

---

## Python (ctypes)

See `examples/python-via-cffi/infer.py` for a complete example. Quick version:

```python
import ctypes, platform, os

lib = ctypes.CDLL("target/release/libneedle_c.so")
lib.needle_load.restype  = ctypes.c_void_p
lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.needle_run.restype   = ctypes.c_char_p
lib.needle_run.argtypes  = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
lib.needle_free_str.argtypes = [ctypes.c_char_p]
lib.needle_free.argtypes     = [ctypes.c_void_p]

handle = lib.needle_load(b"weights/needle.safetensors", b"weights/vocab.txt")
tools  = b'[{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}]'
result = lib.needle_run(handle, b"What is the weather in Paris?", tools)
print(result.decode())
lib.needle_free(handle)
```

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
    w := C.CString("weights/needle.safetensors")
    v := C.CString("weights/vocab.txt")
    defer C.free(unsafe.Pointer(w))
    defer C.free(unsafe.Pointer(v))

    handle := C.needle_load(w, v)
    if handle == nil {
        panic(C.GoString(C.needle_last_error()))
    }
    defer C.needle_free(handle)

    q := C.CString("What is the weather in Paris?")
    t := C.CString(`[{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}]`)
    defer C.free(unsafe.Pointer(q))
    defer C.free(unsafe.Pointer(t))

    result := C.needle_run(handle, q, t)
    defer C.needle_free_str(result)
    fmt.Println(C.GoString(result))
}
```

---

## Swift

```swift
import Foundation

// Load the library (assumes libneedle_c.dylib is in the dynamic linker path)
typealias NeedleHandle = OpaquePointer

@_silgen_name("needle_load")
func needle_load(_ weights: UnsafePointer<CChar>, _ vocab: UnsafePointer<CChar>) -> NeedleHandle?

@_silgen_name("needle_run")
func needle_run(_ handle: NeedleHandle, _ query: UnsafePointer<CChar>, _ tools: UnsafePointer<CChar>) -> UnsafeMutablePointer<CChar>?

@_silgen_name("needle_free_str")
func needle_free_str(_ s: UnsafeMutablePointer<CChar>?)

@_silgen_name("needle_free")
func needle_free(_ handle: NeedleHandle)

let handle = needle_load("weights/needle.safetensors", "weights/vocab.txt")!
let tools = #"[{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}]"#
let result = needle_run(handle, "What is the weather in Paris?", tools)!
print(String(cString: result))
needle_free_str(result)
needle_free(handle)
```

---

## Thread safety

`NeedleHandle` is not `Send` — do not share a handle across threads. Create one handle per thread, or add external locking. `needle_last_error()` uses thread-local storage; it is safe to call from multiple threads simultaneously.
