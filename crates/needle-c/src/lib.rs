//! C ABI shared library for Needle.
//! Exposes a stable C interface for Python (ctypes/cffi), Go, Swift, etc.
//!
//! API:
//!   needle_load(weights_path, vocab_path)                  → *NeedleHandle
//!   needle_load_bytes(weights_data, weights_len, vocab_data, vocab_len) → *NeedleHandle
//!   needle_run(handle, query, tools_json)                  → *char  (free with needle_free_str)
//!   needle_run_stream(handle, query, tools_json, cb, ud)   → *char  (free with needle_free_str)
//!   needle_encode_contrastive(handle, text, out, dim)      → bool
//!   needle_contrastive_dim(handle)                         → usize
//!   needle_retrieve_tools(handle, query, descs, n, k, idx, scores) → usize
//!   needle_free_str(s)
//!   needle_free(handle)
//!   needle_last_error()                                    → *const char

use needle_infer::NeedleEngine;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

std::thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
}

fn set_last_error(msg: impl std::fmt::Display) {
    let s = CString::new(msg.to_string())
        .unwrap_or_else(|_| CString::new("(error message contained null byte)").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(s));
}

fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Opaque handle to a loaded NeedleEngine.
pub struct NeedleHandle {
    engine: NeedleEngine,
}

/// Load a Needle model from disk.
/// Returns null on failure. Caller must free with `needle_free`.
///
/// # Safety
/// `weights_path` and `vocab_path` must be non-null, valid, null-terminated UTF-8 C strings.
#[no_mangle]
pub unsafe extern "C" fn needle_load(
    weights_path: *const c_char,
    vocab_path: *const c_char,
) -> *mut NeedleHandle {
    let weights = match CStr::from_ptr(weights_path).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let vocab = match CStr::from_ptr(vocab_path).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    clear_last_error();
    match NeedleEngine::load(weights, vocab) {
        Ok(engine) => Box::into_raw(Box::new(NeedleHandle { engine })),
        Err(e) => {
            set_last_error(format!("load error: {e}"));
            ptr::null_mut()
        }
    }
}

/// Run inference. Returns a heap-allocated JSON string. Caller must free with `needle_free_str`.
/// Returns null on error.
///
/// # Safety
/// `handle` must be null or a valid handle returned by `needle_load`/`needle_load_bytes`.
/// `query` and `tools_json` must be non-null, valid, null-terminated UTF-8 C strings.
#[no_mangle]
pub unsafe extern "C" fn needle_run(
    handle: *mut NeedleHandle,
    query: *const c_char,
    tools_json: *const c_char,
) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let h = &*handle;

    let query_str = match CStr::from_ptr(query).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let tools_str = match CStr::from_ptr(tools_json).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    clear_last_error();
    let result = h.engine.run(query_str, tools_str);

    match CString::new(result.text) {
        Ok(cs) => cs.into_raw(),
        Err(e) => {
            set_last_error(format!("output contained null byte: {e}"));
            ptr::null_mut()
        }
    }
}

/// Free a string returned by `needle_run` or `needle_run_stream`. No-op on null.
///
/// # Safety
/// `s` must be null or a pointer previously returned by `needle_run`/`needle_run_stream`.
#[no_mangle]
pub unsafe extern "C" fn needle_free_str(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Free a NeedleHandle returned by `needle_load` or `needle_load_bytes`. No-op on null.
///
/// # Safety
/// `handle` must be null or a pointer previously returned by `needle_load`/`needle_load_bytes`.
#[no_mangle]
pub unsafe extern "C" fn needle_free(handle: *mut NeedleHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Load a Needle model from in-memory byte buffers.
///
/// `weights_data`/`weights_len`: raw bytes of the SafeTensors file.
/// `vocab_data`/`vocab_len`: UTF-8 bytes of the vocabulary text file.
///
/// Returns null on failure; use `needle_last_error()` to retrieve the message.
/// Caller must free the returned handle with `needle_free`.
///
/// # Safety
/// `weights_data` must be a valid pointer to at least `weights_len` bytes.
/// `vocab_data` must be a valid pointer to at least `vocab_len` bytes of UTF-8 text.
#[no_mangle]
pub unsafe extern "C" fn needle_load_bytes(
    weights_data: *const u8,
    weights_len: usize,
    vocab_data: *const u8,
    vocab_len: usize,
) -> *mut NeedleHandle {
    if weights_data.is_null() || vocab_data.is_null() {
        set_last_error("null pointer passed to needle_load_bytes");
        return ptr::null_mut();
    }
    let weights = std::slice::from_raw_parts(weights_data, weights_len).to_vec();
    let vocab_str = match std::str::from_utf8(std::slice::from_raw_parts(vocab_data, vocab_len)) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("vocab is not valid UTF-8: {e}"));
            return ptr::null_mut();
        }
    };

    clear_last_error();
    match NeedleEngine::from_bytes(weights, vocab_str) {
        Ok(engine) => Box::into_raw(Box::new(NeedleHandle { engine })),
        Err(e) => {
            set_last_error(format!("load error: {e}"));
            ptr::null_mut()
        }
    }
}

/// C callback type for streaming: called once per generated token.
/// `token_id`: raw token ID.
/// `piece`:    null-terminated UTF-8 text for this token (e.g., " Paris").
/// `user_data`: caller-supplied context pointer.
pub type NeedleStreamCallback =
    unsafe extern "C" fn(token_id: u32, piece: *const c_char, user_data: *mut c_void);

/// Run inference with streaming. Returns the final post-processed output (same as `needle_run`).
///
/// `callback` fires for each generated token in decode order (including `<tool_call>`).
/// `user_data` is passed through to each callback invocation unchanged.
///
/// Returns null on error; use `needle_last_error()`.
/// Caller must free the returned string with `needle_free_str`.
///
/// # Safety
/// `handle` must be null or a valid handle returned by `needle_load`/`needle_load_bytes`.
/// `query` and `tools_json` must be non-null, valid, null-terminated UTF-8 C strings.
/// `user_data` may be null or any pointer the callback understands; the library does not dereference it.
#[no_mangle]
pub unsafe extern "C" fn needle_run_stream(
    handle: *mut NeedleHandle,
    query: *const c_char,
    tools_json: *const c_char,
    callback: Option<NeedleStreamCallback>,
    user_data: *mut c_void,
) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let h = &*handle;

    let query_str = match CStr::from_ptr(query).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let tools_str = match CStr::from_ptr(tools_json).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    clear_last_error();
    let result = h
        .engine
        .run_stream(query_str, tools_str, |token_id, piece| {
            if let Some(cb) = callback {
                if let Ok(cs) = CString::new(piece) {
                    cb(token_id, cs.as_ptr(), user_data);
                }
            }
        });

    match CString::new(result.text) {
        Ok(cs) => cs.into_raw(),
        Err(e) => {
            set_last_error(format!("output contained null byte: {e}"));
            ptr::null_mut()
        }
    }
}

/// Encode text into a L2-normalized contrastive embedding.
///
/// `text`:  null-terminated UTF-8 input string.
/// `out`:   caller-allocated float32 buffer of at least `needle_contrastive_dim(handle)` elements.
/// `dim`:   size of `out` in float32 elements (must equal `needle_contrastive_dim(handle)`).
///
/// Returns true on success, false if the model has no contrastive head or dimensions mismatch.
///
/// # Safety
/// `handle` must be null or a valid handle. `text` must be a valid null-terminated UTF-8 string.
/// `out` must point to a valid, writable buffer of at least `dim` `f32` values.
#[no_mangle]
pub unsafe extern "C" fn needle_encode_contrastive(
    handle: *mut NeedleHandle,
    text: *const c_char,
    out: *mut f32,
    dim: usize,
) -> bool {
    if handle.is_null() || out.is_null() {
        return false;
    }
    let h = &*handle;
    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    match h.engine.encode_contrastive(text_str) {
        Some(emb) if emb.len() == dim => {
            let out_slice = std::slice::from_raw_parts_mut(out, dim);
            out_slice.copy_from_slice(&emb);
            true
        }
        _ => false,
    }
}

/// Return the contrastive embedding dimension (0 if no contrastive head loaded).
///
/// # Safety
/// `handle` must be null or a valid handle returned by `needle_load`/`needle_load_bytes`.
#[no_mangle]
pub unsafe extern "C" fn needle_contrastive_dim(handle: *mut NeedleHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    (*handle).engine.contrastive_dim()
}

/// Rank tool descriptions by contrastive similarity to a query.
///
/// Encodes `query` and each `tool_descs[i]` with the contrastive head, computes
/// dot-product similarity (both embeddings are L2-normalized), and writes the top-k
/// results into caller-allocated `out_indices` and `out_scores` buffers.
///
/// Returns the number of results written (min(top_k, n_tools)), or 0 if the model
/// has no contrastive head or any pointer argument is null.
///
/// `tool_descs`: array of `n_tools` null-terminated UTF-8 strings.
/// `out_indices`: caller buffer of at least `top_k` `size_t` values (result indices).
/// `out_scores`:  caller buffer of at least `top_k` `f32` values (cosine scores, desc order).
///
/// # Safety
/// `handle` must be null or a valid handle. `query` must be a valid null-terminated UTF-8 string.
/// `tool_descs` must be a valid array of `n_tools` non-null, null-terminated UTF-8 string pointers.
/// `out_indices` must point to a writable buffer of at least `top_k` `usize` values.
/// `out_scores` must point to a writable buffer of at least `top_k` `f32` values.
#[no_mangle]
pub unsafe extern "C" fn needle_retrieve_tools(
    handle: *mut NeedleHandle,
    query: *const c_char,
    tool_descs: *const *const c_char,
    n_tools: usize,
    top_k: usize,
    out_indices: *mut usize,
    out_scores: *mut f32,
) -> usize {
    if handle.is_null()
        || query.is_null()
        || tool_descs.is_null()
        || out_indices.is_null()
        || out_scores.is_null()
        || n_tools == 0
        || top_k == 0
    {
        return 0;
    }
    let h = &*handle;
    let query_str = match CStr::from_ptr(query).to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let mut descs: Vec<&str> = Vec::with_capacity(n_tools);
    for i in 0..n_tools {
        let ptr = *tool_descs.add(i);
        if ptr.is_null() {
            return 0;
        }
        match CStr::from_ptr(ptr).to_str() {
            Ok(s) => descs.push(s),
            Err(_) => return 0,
        }
    }

    let results = h.engine.retrieve_tools(query_str, &descs, top_k);
    let n = results.len();
    for (i, (idx, score)) in results.into_iter().enumerate() {
        *out_indices.add(i) = idx;
        *out_scores.add(i) = score;
    }
    n
}

/// Return the last error message as a null-terminated C string, or null if none.
/// The returned pointer is valid until the next call to any needle_* function on this thread.
/// Do NOT free this pointer.
#[no_mangle]
pub extern "C" fn needle_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|cs| cs.as_ptr())
            .unwrap_or(ptr::null())
    })
}
