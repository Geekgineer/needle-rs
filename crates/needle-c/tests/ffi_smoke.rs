//! FFI smoke test: exercises the C ABI lifecycle — load → run → free.
//!
//! Tests null-safety, error paths (bad paths, null pointers), and the
//! full happy-path load → run → free_str → free when weights are present.
//!
//! Run: cargo test -p needle-c ffi -- --nocapture

use needle_c::{
    needle_contrastive_dim, needle_encode_contrastive, needle_free, needle_free_str,
    needle_last_error, needle_load, needle_load_bytes, needle_retrieve_tools, needle_run,
    needle_run_stream,
};
use std::ffi::{CStr, CString};

const WEIGHTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../weights/needle.safetensors"
);
const VOCAB: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/vocab.txt");

fn weights_available() -> bool {
    std::path::Path::new(WEIGHTS).exists() && std::path::Path::new(VOCAB).exists()
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// Null / missing-file paths must return null with a queryable error — never crash.
#[test]
fn test_ffi_bad_paths_return_null() {
    unsafe {
        let bad = cstr("/nonexistent/path.safetensors");
        let vocab = cstr("/nonexistent/vocab.txt");
        let handle = needle_load(bad.as_ptr(), vocab.as_ptr());
        assert!(handle.is_null(), "expected null for missing files");

        // needle_last_error() must return a non-null, non-empty error string.
        let err_ptr = needle_last_error();
        assert!(
            !err_ptr.is_null(),
            "expected error message after failed load"
        );
        let err = CStr::from_ptr(err_ptr)
            .to_str()
            .expect("error must be valid UTF-8");
        assert!(!err.is_empty(), "error string must not be empty");
        eprintln!("last error after bad load: {err:?}");
    }
}

/// needle_run on a null handle must return null — never crash.
#[test]
fn test_ffi_null_handle_run_returns_null() {
    unsafe {
        let q = cstr("hello");
        let t = cstr("[]");
        let result = needle_run(std::ptr::null_mut(), q.as_ptr(), t.as_ptr());
        assert!(result.is_null(), "expected null for null handle");
    }
}

/// needle_free on null must be a no-op — never crash.
#[test]
fn test_ffi_free_null_is_safe() {
    unsafe {
        needle_free(std::ptr::null_mut());
        needle_free_str(std::ptr::null_mut());
    }
}

/// Full happy-path: load → run → inspect result → free_str → free.
/// Only runs when weights are present (same skip pattern as e2e tests).
#[test]
fn test_ffi_load_run_free() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }

    let w = cstr(WEIGHTS);
    let v = cstr(VOCAB);

    let handle = unsafe { needle_load(w.as_ptr(), v.as_ptr()) };
    assert!(
        !handle.is_null(),
        "needle_load returned null — check weights"
    );

    let query = cstr("What's the weather in Paris?");
    let tools = cstr(
        r#"[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"}}}]"#,
    );

    let result_ptr = unsafe { needle_run(handle, query.as_ptr(), tools.as_ptr()) };
    assert!(!result_ptr.is_null(), "needle_run returned null");

    let result_str = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap() };
    eprintln!("FFI output: {result_str:?}");

    assert!(
        result_str.contains("get_weather"),
        "expected tool name in output"
    );
    assert!(
        result_str.contains("location"),
        "expected arg key in output"
    );

    unsafe {
        needle_free_str(result_ptr);
        needle_free(handle);
    }
}

/// needle_load_bytes: load from in-memory buffers and verify result matches needle_load.
#[test]
fn test_ffi_load_bytes() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }

    let weights_bytes = std::fs::read(WEIGHTS).expect("weights read failed");
    let vocab_bytes = std::fs::read(VOCAB).expect("vocab read failed");

    let handle = unsafe {
        needle_load_bytes(
            weights_bytes.as_ptr(),
            weights_bytes.len(),
            vocab_bytes.as_ptr(),
            vocab_bytes.len(),
        )
    };
    assert!(!handle.is_null(), "needle_load_bytes returned null");

    let query = cstr("What's the weather in Paris?");
    let tools = cstr(
        r#"[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"}}}]"#,
    );

    let result_ptr = unsafe { needle_run(handle, query.as_ptr(), tools.as_ptr()) };
    assert!(
        !result_ptr.is_null(),
        "needle_run after load_bytes returned null"
    );

    let text = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap().to_string() };
    eprintln!("load_bytes output: {text:?}");
    assert!(
        text.contains("get_weather"),
        "expected tool name in output from load_bytes"
    );

    unsafe {
        needle_free_str(result_ptr);
        needle_free(handle);
    }
}

/// needle_load_bytes with null pointers must return null, not crash.
#[test]
fn test_ffi_load_bytes_null_safe() {
    unsafe {
        let h1 = needle_load_bytes(std::ptr::null(), 0, std::ptr::null(), 0);
        assert!(h1.is_null(), "expected null for null weights pointer");
    }
}

/// needle_run_stream: streaming callback fires per-token; final text matches needle_run.
#[test]
fn test_ffi_run_stream() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }

    let w = cstr(WEIGHTS);
    let v = cstr(VOCAB);
    let handle = unsafe { needle_load(w.as_ptr(), v.as_ptr()) };
    assert!(!handle.is_null());

    let query = cstr("What's the weather in Paris?");
    let tools = cstr(
        r#"[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"}}}]"#,
    );

    // Track how many times the callback fires.
    let mut call_count: u32 = 0;
    let call_count_ptr = &mut call_count as *mut u32 as *mut std::ffi::c_void;

    unsafe extern "C" fn on_token(
        _id: u32,
        _piece: *const std::ffi::c_char,
        ud: *mut std::ffi::c_void,
    ) {
        let count = &mut *(ud as *mut u32);
        *count += 1;
    }

    let stream_ptr = unsafe {
        needle_run_stream(
            handle,
            query.as_ptr(),
            tools.as_ptr(),
            Some(on_token),
            call_count_ptr,
        )
    };
    let direct_ptr = unsafe { needle_run(handle, query.as_ptr(), tools.as_ptr()) };

    assert!(!stream_ptr.is_null());
    assert!(!direct_ptr.is_null());

    let stream_text = unsafe { CStr::from_ptr(stream_ptr).to_str().unwrap() };
    let direct_text = unsafe { CStr::from_ptr(direct_ptr).to_str().unwrap() };

    eprintln!("stream output: {stream_text:?}  (callback fired {call_count} times)");
    assert_eq!(
        stream_text, direct_text,
        "stream and direct outputs must match"
    );
    assert!(call_count > 0, "streaming callback must fire at least once");

    unsafe {
        needle_free_str(stream_ptr);
        needle_free_str(direct_ptr);
        needle_free(handle);
    }
}

/// needle_encode_contrastive and needle_contrastive_dim: dimension reported correctly.
#[test]
fn test_ffi_encode_contrastive() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }

    let w = cstr(WEIGHTS);
    let v = cstr(VOCAB);
    let handle = unsafe { needle_load(w.as_ptr(), v.as_ptr()) };
    assert!(!handle.is_null());

    let dim = unsafe { needle_contrastive_dim(handle) };
    eprintln!("contrastive_dim: {dim}");

    if dim == 0 {
        eprintln!("SKIP L2-norm check: no contrastive head in these weights");
        unsafe { needle_free(handle) };
        return;
    }

    let mut emb = vec![0.0f32; dim];
    let text = cstr("What's the weather in Paris?");
    let ok = unsafe { needle_encode_contrastive(handle, text.as_ptr(), emb.as_mut_ptr(), dim) };
    assert!(ok, "needle_encode_contrastive returned false");

    let sq_norm: f32 = emb.iter().map(|x| x * x).sum();
    assert!(
        (sq_norm - 1.0).abs() < 1e-4,
        "embedding not L2-normalized: ||v||²={sq_norm:.6}"
    );
    eprintln!("L2-norm: {sq_norm:.6}  (expected ~1.0)");

    // Wrong dim must return false, not crash.
    let ok_wrong =
        unsafe { needle_encode_contrastive(handle, text.as_ptr(), emb.as_mut_ptr(), dim + 1) };
    assert!(!ok_wrong, "expected false for wrong dim");

    unsafe { needle_free(handle) };
}

/// needle_retrieve_tools: null safety + (when head present) sorted results.
#[test]
fn test_ffi_retrieve_tools() {
    // Null-safety: null handle must return 0, not crash.
    unsafe {
        let q = cstr("query");
        let desc0 = cstr("weather tool");
        let descs: Vec<*const std::ffi::c_char> = vec![desc0.as_ptr()];
        let mut indices = [0usize; 2];
        let mut scores = [0.0f32; 2];
        let n = needle_retrieve_tools(
            std::ptr::null_mut(),
            q.as_ptr(),
            descs.as_ptr(),
            1,
            1,
            indices.as_mut_ptr(),
            scores.as_mut_ptr(),
        );
        assert_eq!(n, 0, "null handle must return 0");
    }

    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }

    let w = cstr(WEIGHTS);
    let v = cstr(VOCAB);
    let handle = unsafe { needle_load(w.as_ptr(), v.as_ptr()) };
    assert!(!handle.is_null());

    let dim = unsafe { needle_contrastive_dim(handle) };
    if dim == 0 {
        eprintln!("SKIP ranking check: no contrastive head in these weights");
        unsafe { needle_free(handle) };
        return;
    }

    // With a contrastive head, results must be sorted descending and in range.
    let query = cstr("What's the weather in Paris?");
    let desc0 = cstr("Get current weather conditions for a city");
    let desc1 = cstr("Search the web for information");
    let desc2 = cstr("Send an email to a recipient");
    let descs: Vec<*const std::ffi::c_char> = vec![desc0.as_ptr(), desc1.as_ptr(), desc2.as_ptr()];
    let mut indices = [0usize; 2];
    let mut scores = [0.0f32; 2];

    let n = unsafe {
        needle_retrieve_tools(
            handle,
            query.as_ptr(),
            descs.as_ptr(),
            3,
            2,
            indices.as_mut_ptr(),
            scores.as_mut_ptr(),
        )
    };
    assert_eq!(n, 2, "expected 2 results for top_k=2");
    assert!(
        scores[0] >= scores[1],
        "results must be sorted by descending score"
    );
    for i in 0..n {
        assert!(indices[i] < 3, "result index must be in range");
        assert!(
            scores[i] >= -1.0 && scores[i] <= 1.0 + 1e-4,
            "score must be in [-1, 1]"
        );
    }
    eprintln!(
        "retrieve_tools: [{}: {:.4}], [{}: {:.4}]",
        indices[0], scores[0], indices[1], scores[1]
    );

    unsafe { needle_free(handle) };
}
