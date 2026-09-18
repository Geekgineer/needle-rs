//! Exercises the `needle_v3_*` C ABI through the same pointers a C caller uses.
//!
//! Requires `weights/needle3.cact`; the null-argument and lifecycle checks run
//! regardless, since they must not need a model.
//!
//! The header is verified separately by compiling `examples/c/v3_smoke.c`
//! against it — a declaration that does not match the library is the failure
//! this cannot see.
//!
//! Run: cargo test -p needle-c --release --test v3_ffi_smoke -- --nocapture

use needle_c::v3::*;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");
const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;
const QUERY: &str = "What's the weather in Paris?";

fn have_weights() -> bool {
    if std::path::Path::new(CACT).exists() {
        return true;
    }
    eprintln!("skipping: missing {CACT}");
    false
}

unsafe fn load() -> *mut NeedleV3Handle {
    let p = CString::new(CACT).unwrap();
    let h = needle_v3_load(p.as_ptr());
    assert!(!h.is_null(), "load failed");
    h
}

unsafe fn take(s: *mut c_char) -> String {
    assert!(!s.is_null(), "expected a string, got null");
    let out = CStr::from_ptr(s).to_string_lossy().into_owned();
    needle_c::needle_free_str(s);
    out
}

#[test]
fn run_and_payload() {
    if !have_weights() {
        return;
    }
    unsafe {
        let h = load();
        let q = CString::new(QUERY).unwrap();
        let t = CString::new(TOOLS).unwrap();

        let text = take(needle_v3_run(h, q.as_ptr(), t.as_ptr()));
        println!("run -> {text}");
        assert!(text.contains("get_weather"));

        let payload = take(needle_v3_run_json(h, q.as_ptr(), t.as_ptr()));
        println!("run_json -> {payload}");
        assert!(payload.starts_with('['));
        assert!(payload.contains("Paris"));

        // v3 reasons; v2 did not. Null is allowed, a wrong type is not.
        let txt = CString::new(text.as_str()).unwrap();
        let r = needle_v3_reasoning(h, txt.as_ptr());
        if !r.is_null() {
            println!("reasoning -> {}", take(r));
        }

        needle_v3_free(h);
    }
}

#[test]
fn confidence_scores_a_completion() {
    if !have_weights() {
        return;
    }
    unsafe {
        let h = load();
        let q = CString::new(QUERY).unwrap();
        let t = CString::new(TOOLS).unwrap();
        assert!(needle_v3_has_confidence(h), "v3 exports a confidence head");

        let good = CString::new(
            r#"<tool_call>[{"name":"get_weather","arguments":{"city":"Paris"}}]</tool_call>"#,
        )
        .unwrap();
        let bad = CString::new(
            r#"<tool_call>[{"name":"get_weather","arguments":{"city":"Berlin"}}]</tool_call>"#,
        )
        .unwrap();

        let (mut pg, mut pb) = (0.0f32, 0.0f32);
        assert!(needle_v3_confidence_for(
            h,
            q.as_ptr(),
            t.as_ptr(),
            good.as_ptr(),
            &mut pg
        ));
        assert!(needle_v3_confidence_for(
            h,
            q.as_ptr(),
            t.as_ptr(),
            bad.as_ptr(),
            &mut pb
        ));
        println!("confidence: Paris {pg:.4}, Berlin {pb:.4}");
        assert!((0.0..=1.0).contains(&pg));
        assert!(pg > pb, "the right city should outscore the wrong one");

        // A null out pointer must fail rather than write through it.
        assert!(!needle_v3_confidence_for(
            h,
            q.as_ptr(),
            t.as_ptr(),
            good.as_ptr(),
            ptr::null_mut()
        ));

        needle_v3_free(h);
    }
}

extern "C" fn collect(piece: *const c_char, userdata: *mut c_void) {
    unsafe {
        let buf = &mut *(userdata as *mut String);
        if !piece.is_null() {
            buf.push_str(&CStr::from_ptr(piece).to_string_lossy());
        }
    }
}

#[test]
fn streaming_reproduces_the_return_value() {
    if !have_weights() {
        return;
    }
    unsafe {
        let h = load();
        let q = CString::new(QUERY).unwrap();
        let t = CString::new(TOOLS).unwrap();

        let mut buf = String::new();
        let text = take(needle_v3_run_stream(
            h,
            q.as_ptr(),
            t.as_ptr(),
            Some(collect),
            &mut buf as *mut String as *mut c_void,
        ));
        // Deltas are decoded text, so concatenation must be exact.
        assert_eq!(buf, text, "streamed deltas diverged from the return value");
        needle_v3_free(h);
    }
}

#[test]
fn memory_reporting_is_available_before_a_session() {
    if !have_weights() {
        return;
    }
    unsafe {
        let h = load();
        let max = needle_v3_max_seq_len(h);
        let short = needle_v3_kv_bytes(h, 512, false);
        let long = needle_v3_kv_bytes(h, 4096, false);
        let long_q = needle_v3_kv_bytes(h, 4096, true);
        println!(
            "max_seq_len {max}, kv_bytes 512 -> {:.1} MB, 4096 -> {:.1} MB (int8 {:.1} MB)",
            short as f64 / 1048576.0,
            long as f64 / 1048576.0,
            long_q as f64 / 1048576.0
        );
        assert_eq!(max, 8192);
        // The caller often owns the memory budget, so this has to grow with
        // the session rather than report a fixed reservation.
        assert!(short < long, "cache cost should grow with the session");
        // And the int8 flag has to change the answer, or a caller budgeting
        // for a quantised session is given the f32 number.
        assert!(
            long_q * 3 < long,
            "int8 estimate {long_q} is not below f32 {long}"
        );
        needle_v3_free(h);
    }
}

#[test]
fn a_ladder_rung_loads_at_a_shallower_depth() {
    if !have_weights() {
        return;
    }
    unsafe {
        let p = CString::new(CACT).unwrap();
        let full = load();
        assert_eq!(needle_v3_num_layers(full), 20);

        let rung = needle_v3_load_with_depth(p.as_ptr(), 8);
        assert!(!rung.is_null(), "an 8-block rung should load");
        assert_eq!(needle_v3_num_layers(rung), 8);

        // The whole point of a rung is that it costs less to run.
        let full_kv = needle_v3_kv_bytes(full, 512, false);
        let rung_kv = needle_v3_kv_bytes(rung, 512, false);
        println!(
            "  20L {:.1} MB vs 8L {:.1} MB of cache at 512 tokens",
            full_kv as f64 / 1048576.0,
            rung_kv as f64 / 1048576.0
        );
        assert!(rung_kv < full_kv, "the rung should need less cache");

        let q = CString::new(QUERY).unwrap();
        let t = CString::new(TOOLS).unwrap();
        let payload = take(needle_v3_run_json(rung, q.as_ptr(), t.as_ptr()));
        println!("  8L run_json -> {payload}");
        assert!(payload.contains("get_weather"));

        // Outside 2..=num_layers is a failure, not a clamp.
        assert!(needle_v3_load_with_depth(p.as_ptr(), 99).is_null());
        assert!(needle_v3_load_with_depth(p.as_ptr(), 1).is_null());

        needle_v3_free(rung);
        needle_v3_free(full);
    }
}

#[test]
fn null_arguments_do_not_crash() {
    unsafe {
        // Must hold with no model loaded at all.
        needle_v3_free(ptr::null_mut());
        assert!(needle_v3_load(ptr::null()).is_null());
        assert!(needle_v3_load_bytes(ptr::null(), 0).is_null());
        assert!(!needle_v3_has_confidence(ptr::null_mut()));
        assert_eq!(needle_v3_kv_bytes(ptr::null_mut(), 128, false), 0);
        assert_eq!(needle_v3_kv_bytes(ptr::null_mut(), 128, true), 0);
        assert_eq!(needle_v3_max_seq_len(ptr::null_mut()), 0);
        assert_eq!(needle_v3_num_layers(ptr::null_mut()), 0);
        assert!(needle_v3_load_with_depth(ptr::null(), 8).is_null());
        assert!(needle_v3_run(ptr::null_mut(), ptr::null(), ptr::null()).is_null());

        let missing = CString::new("definitely-not-here.cact").unwrap();
        assert!(needle_v3_load(missing.as_ptr()).is_null());
    }
}

#[test]
fn a_v2_container_is_refused() {
    let v2 = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle2.cact");
    if !std::path::Path::new(v2).exists() {
        eprintln!("skipping: no needle2.cact");
        return;
    }
    unsafe {
        // Both generations are `.cact`. Loading the wrong one must fail rather
        // than read a 120-byte header as 196 and produce wrong geometry.
        let p = CString::new(v2).unwrap();
        assert!(
            needle_v3_load(p.as_ptr()).is_null(),
            "a v2 container must not load as v3"
        );
    }
}
