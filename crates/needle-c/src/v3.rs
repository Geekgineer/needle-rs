//! C ABI for the Needle 3 path.
//!
//! ```text
//! needle_v3_load(cact_path)                                  -> *NeedleV3Handle
//! needle_v3_load_bytes(data, len)                            -> *NeedleV3Handle
//! needle_v3_run(h, query, tools_json)                        -> *char
//! needle_v3_run_json(h, query, tools_json)                   -> *char  (payload only)
//! needle_v3_reasoning(h, text)                               -> *char  (may be null)
//! needle_v3_generate(h, query, tools, max, temp, seed, constrain) -> *char
//! needle_v3_run_stream(h, query, tools, cb, userdata)        -> *char
//! needle_v3_has_confidence(h)                                -> bool
//! needle_v3_confidence_for(h, query, tools, completion, out) -> bool
//! needle_v3_kv_bytes(h, seq_len)                             -> usize
//! needle_v3_max_seq_len(h)                                   -> usize
//! needle_v3_free(h)
//! ```
//!
//! Strings are freed with `needle_free_str` and errors read with
//! `needle_last_error`, both shared with the v1 and v2 surfaces.
//!
//! There is deliberately no `needle_v3_retrieve_tools` or
//! `needle_v3_encode_contrastive`: v3 exports a confidence head and nothing
//! else, so those would have to return empty on every call. An absent symbol
//! is a compile error at the call site; a present one that always fails is a
//! runtime mystery.

use crate::{clear_last_error, set_last_error};
use needle_infer::v3_engine::{V3Engine, V3Options, DEFAULT_MAX_NEW_TOKENS};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

/// Opaque handle to a loaded v3 engine.
pub struct NeedleV3Handle {
    engine: V3Engine,
}

unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok()
}

fn out_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => {
            set_last_error("result contained a null byte");
            ptr::null_mut()
        }
    }
}

/// Load a v3 model from a `.cact` file. Null on failure; free with
/// `needle_v3_free`.
///
/// # Safety
/// `cact_path` must be a valid, null-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn needle_v3_load(cact_path: *const c_char) -> *mut NeedleV3Handle {
    clear_last_error();
    let Some(path) = cstr(cact_path) else {
        set_last_error("cact_path was null or not UTF-8");
        return ptr::null_mut();
    };
    match V3Engine::load(path) {
        Ok(engine) => Box::into_raw(Box::new(NeedleV3Handle { engine })),
        Err(e) => {
            set_last_error(&format!("failed to load {path}: {e}"));
            ptr::null_mut()
        }
    }
}

/// Load a v3 model from bytes already in memory.
///
/// # Safety
/// `data` must point to at least `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn needle_v3_load_bytes(data: *const u8, len: usize) -> *mut NeedleV3Handle {
    clear_last_error();
    if data.is_null() {
        set_last_error("data was null");
        return ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(data, len).to_vec();
    match V3Engine::from_bytes(bytes) {
        Ok(engine) => Box::into_raw(Box::new(NeedleV3Handle { engine })),
        Err(e) => {
            set_last_error(&format!("failed to load container: {e}"));
            ptr::null_mut()
        }
    }
}

unsafe fn handle<'a>(h: *mut NeedleV3Handle) -> Option<&'a NeedleV3Handle> {
    if h.is_null() {
        set_last_error("handle was null");
        return None;
    }
    Some(&*h)
}

/// The full completion, reasoning included. Free with `needle_free_str`.
///
/// # Safety
/// `handle` must come from `needle_v3_load`; the strings must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn needle_v3_run(
    handle_ptr: *mut NeedleV3Handle,
    query: *const c_char,
    tools_json: *const c_char,
) -> *mut c_char {
    clear_last_error();
    let (Some(h), Some(q), Some(t)) = (handle(handle_ptr), cstr(query), cstr(tools_json)) else {
        return ptr::null_mut();
    };
    out_string(h.engine.run(q, t))
}

/// Just the tool-call payload.
///
/// Returns `"[]"` when the model considered the tools and declined — a
/// decision, not a failure — and an empty string when it emitted no
/// `<tool_call>` markers at all. Callers that collapse the two turn a
/// considered "no" into an error.
///
/// # Safety
/// As [`needle_v3_run`].
#[no_mangle]
pub unsafe extern "C" fn needle_v3_run_json(
    handle_ptr: *mut NeedleV3Handle,
    query: *const c_char,
    tools_json: *const c_char,
) -> *mut c_char {
    clear_last_error();
    let (Some(h), Some(q), Some(t)) = (handle(handle_ptr), cstr(query), cstr(tools_json)) else {
        return ptr::null_mut();
    };
    out_string(h.engine.run_json(q, t).unwrap_or_default())
}

/// The chain-of-thought inside a completion, or null if there is none.
///
/// New in v3: v2 answered directly.
///
/// # Safety
/// `handle` must come from `needle_v3_load`; `text` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn needle_v3_reasoning(
    handle_ptr: *mut NeedleV3Handle,
    text: *const c_char,
) -> *mut c_char {
    clear_last_error();
    let (Some(_h), Some(s)) = (handle(handle_ptr), cstr(text)) else {
        return ptr::null_mut();
    };
    match V3Engine::reasoning(s) {
        Some(r) => out_string(r.to_string()),
        None => ptr::null_mut(),
    }
}

/// Generate with explicit settings. `max_new_tokens == 0` uses the default.
///
/// # Safety
/// As [`needle_v3_run`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn needle_v3_generate(
    handle_ptr: *mut NeedleV3Handle,
    query: *const c_char,
    tools_json: *const c_char,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
    constrain: bool,
) -> *mut c_char {
    clear_last_error();
    let (Some(h), Some(q), Some(t)) = (handle(handle_ptr), cstr(query), cstr(tools_json)) else {
        return ptr::null_mut();
    };
    let opts = V3Options {
        max_new_tokens: if max_new_tokens == 0 {
            DEFAULT_MAX_NEW_TOKENS
        } else {
            max_new_tokens
        },
        temperature,
        seed,
        system: None,
        constrain,
    };
    out_string(h.engine.generate(q, t, &opts).text)
}

/// Generate, invoking `cb(piece, userdata)` with each decoded delta.
///
/// The callback receives decoded text, not raw tokenizer pieces:
/// concatenating every delta reproduces the returned string exactly.
///
/// # Safety
/// As [`needle_v3_run`]; `cb` must be callable with a null-terminated string.
#[no_mangle]
pub unsafe extern "C" fn needle_v3_run_stream(
    handle_ptr: *mut NeedleV3Handle,
    query: *const c_char,
    tools_json: *const c_char,
    cb: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    userdata: *mut c_void,
) -> *mut c_char {
    clear_last_error();
    let (Some(h), Some(q), Some(t)) = (handle(handle_ptr), cstr(query), cstr(tools_json)) else {
        return ptr::null_mut();
    };
    let res = h
        .engine
        .generate_with(q, t, &V3Options::default(), |_id, piece| {
            if let Some(f) = cb {
                if let Ok(c) = CString::new(piece) {
                    f(c.as_ptr(), userdata);
                }
            }
        });
    out_string(res.text)
}

/// Whether this container carries a confidence head.
///
/// # Safety
/// `handle` must come from `needle_v3_load`.
#[no_mangle]
pub unsafe extern "C" fn needle_v3_has_confidence(handle_ptr: *mut NeedleV3Handle) -> bool {
    clear_last_error();
    handle(handle_ptr).is_some_and(|h| h.engine.confidence.is_some())
}

/// How confident the model is in a completion it produced.
///
/// **Pass the completion, not the query.** The head scores a finished
/// judgement. On the shipped checkpoint a correct call scores 0.93, a wrong
/// one 0.26, and a bare query 0.80 — which looks like a confident answer and
/// is not one. v2 collapsed to near zero on a bare query, so that misuse
/// announced itself; v3's does not.
///
/// Writes a probability in `(0, 1)` to `out` and returns true.
///
/// # Safety
/// As [`needle_v3_run`]; `out` must point to a writable `float`.
#[no_mangle]
pub unsafe extern "C" fn needle_v3_confidence_for(
    handle_ptr: *mut NeedleV3Handle,
    query: *const c_char,
    tools_json: *const c_char,
    completion: *const c_char,
    out: *mut f32,
) -> bool {
    clear_last_error();
    let (Some(h), Some(q), Some(t), Some(c)) = (
        handle(handle_ptr),
        cstr(query),
        cstr(tools_json),
        cstr(completion),
    ) else {
        return false;
    };
    if out.is_null() {
        set_last_error("out was null");
        return false;
    }
    match h.engine.confidence_for(q, t, c) {
        Some(p) => {
            *out = p;
            true
        }
        None => {
            set_last_error("this container exports no confidence head");
            false
        }
    }
}

/// Key/value cache bytes for a session of `seq_len` positions.
///
/// Exposed because the caller is often the one with the memory budget — an
/// embedded target deciding whether a session fits at all.
///
/// # Safety
/// `handle` must come from `needle_v3_load`.
#[no_mangle]
pub unsafe extern "C" fn needle_v3_kv_bytes(
    handle_ptr: *mut NeedleV3Handle,
    seq_len: usize,
) -> usize {
    clear_last_error();
    handle(handle_ptr).map_or(0, |h| h.engine.model.cfg.kv_bytes(seq_len, 4))
}

/// Context limit in tokens.
///
/// # Safety
/// `handle` must come from `needle_v3_load`.
#[no_mangle]
pub unsafe extern "C" fn needle_v3_max_seq_len(handle_ptr: *mut NeedleV3Handle) -> usize {
    clear_last_error();
    handle(handle_ptr).map_or(0, |h| h.engine.model.cfg.max_seq_len)
}

/// Release a handle.
///
/// # Safety
/// `handle` must come from `needle_v3_load` and not have been freed.
#[no_mangle]
pub unsafe extern "C" fn needle_v3_free(handle_ptr: *mut NeedleV3Handle) {
    if !handle_ptr.is_null() {
        drop(Box::from_raw(handle_ptr));
    }
}
