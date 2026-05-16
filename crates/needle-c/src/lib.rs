//! C ABI shared library for Needle.
//! Exposes a stable C interface for Python (ctypes/cffi), Go, Swift, etc.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use needle_infer::NeedleEngine;

/// Opaque handle to a loaded NeedleEngine.
pub struct NeedleHandle {
    engine: NeedleEngine,
}

/// Load a Needle model from disk.
/// Returns null on failure. Caller must free with `needle_free`.
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

    match NeedleEngine::load(weights, vocab) {
        Ok(engine) => Box::into_raw(Box::new(NeedleHandle { engine })),
        Err(e) => {
            eprintln!("[needle-c] load error: {e}");
            ptr::null_mut()
        }
    }
}

/// Run inference. Returns a heap-allocated JSON string. Caller must free with `needle_free_str`.
/// Returns null on error.
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

    // TODO: proper tokenization
    let query_ids: Vec<u32> = vec![needle_infer::tokenizer::BOS_ID];
    let tools_ids: Vec<u32> = vec![];

    let result = h.engine.run(&query_ids, &tools_ids, tools_str);

    match CString::new(result.text) {
        Ok(cs) => cs.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string returned by `needle_run`.
#[no_mangle]
pub unsafe extern "C" fn needle_free_str(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Free a NeedleHandle returned by `needle_load`.
#[no_mangle]
pub unsafe extern "C" fn needle_free(handle: *mut NeedleHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}
