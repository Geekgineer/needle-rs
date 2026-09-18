use needle_infer::cact::{TAG, TAG_V3};
use needle_infer::v2_engine::{GenerateOptions, V2Engine};
use needle_infer::v3_engine::{KvPrecision, V3Engine, V3Options};
use needle_infer::NeedleEngine;
use std::env;
use std::io::Write as IoWrite;
use std::process;

const USAGE: &str = "\
Usage:
  needle-rs [OPTIONS] <model.cact> <query> <tools_json>          (Needle 3 or 2)
  needle-rs [OPTIONS] <weights.safetensors> <vocab.txt> <query> <tools_json>   (v1)

The model version is taken from the file itself, not its name: a `.cact`
container states its generation in its first word and carries its own geometry
and tokenizer, so it needs no vocabulary argument. A `.safetensors` file is the
v1 format and needs one.

Arguments:
  model       Path to a .cact container (v3 or v2) or .safetensors weights (v1)
  vocab       v1 only: vocabulary text file, one piece per line
  query       User query string
  tools       JSON array of tool definitions

Options:
  --stream            Print each token to stderr as generated; result to stdout
  --max-tokens <N>    Generation cap (v3 default 256, v2 default 128)
  --temperature <T>   0 for greedy, higher to sample (v3, v2; default 0)
  --seed <N>          Sampling seed (v3, v2; default 0)
  --system <TEXT>     System message (v3, v2)
  --json              Print only the tool-call payload, not the full text (v3, v2)
  --constrain         Restrict the tool-call payload to the declared schema (v3, v2)
  --kv-int8           Store the KV cache as int8: ~1/4 the memory (v3)
  --prefill-chunk <N> Positions per batched-prefill chunk; 0 prefills one at a
                      time (v2 only, default 64)
  --help              Print this message

Examples:
  needle-rs weights/needle2.cact \\
    \"What's the weather in Paris?\" \\
    '[{\"name\":\"get_weather\",\"parameters\":{\"city\":{\"type\":\"string\"}}}]'

  needle-rs --stream --max-tokens 64 weights/needle2.cact \\
    \"Book a flight\" '[{\"name\":\"book_flight\",\"parameters\":{}}]'
";

struct Opts {
    stream: bool,
    json_only: bool,
    constrain: bool,
    kv_int8: bool,
    prefill_chunk: Option<usize>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    seed: Option<u64>,
    system: Option<String>,
    positional: Vec<String>,
}

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    process::exit(1);
}

/// Consume the value that follows a flag, so a missing one gives one message.
fn take(raw: &[String], i: &mut usize, name: &str) -> String {
    *i += 1;
    raw.get(*i)
        .cloned()
        .unwrap_or_else(|| fail(&format!("{name} needs a value\n\n{USAGE}")))
}

fn parse_args() -> Opts {
    let raw: Vec<String> = env::args().skip(1).collect();
    let mut o = Opts {
        stream: false,
        json_only: false,
        constrain: false,
        kv_int8: false,
        prefill_chunk: None,
        max_tokens: None,
        temperature: None,
        seed: None,
        system: None,
        positional: Vec::new(),
    };
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--stream" => o.stream = true,
            "--json" => o.json_only = true,
            "--constrain" => o.constrain = true,
            "--kv-int8" => o.kv_int8 = true,
            "--prefill-chunk" => {
                let v = take(&raw, &mut i, "--prefill-chunk");
                o.prefill_chunk = Some(
                    v.parse()
                        .unwrap_or_else(|_| fail("--prefill-chunk must be an integer")),
                );
            }
            "--max-tokens" => {
                let v = take(&raw, &mut i, "--max-tokens");
                o.max_tokens = Some(
                    v.parse()
                        .unwrap_or_else(|_| fail("--max-tokens must be an integer")),
                );
            }
            "--temperature" => {
                let v = take(&raw, &mut i, "--temperature");
                o.temperature = Some(
                    v.parse()
                        .unwrap_or_else(|_| fail("--temperature must be a number")),
                );
            }
            "--seed" => {
                let v = take(&raw, &mut i, "--seed");
                o.seed = Some(
                    v.parse()
                        .unwrap_or_else(|_| fail("--seed must be an integer")),
                );
            }
            "--system" => o.system = Some(take(&raw, &mut i, "--system")),
            "--help" | "-h" => {
                print!("{USAGE}");
                process::exit(0);
            }
            arg if arg.starts_with("--") => fail(&format!("Unknown option: {arg}\n\n{USAGE}")),
            arg => o.positional.push(arg.to_string()),
        }
        i += 1;
    }
    o
}

fn main() {
    let o = parse_args();
    if o.positional.is_empty() {
        fail(USAGE);
    }
    let model = o.positional[0].clone();

    // Both v2 and v3 ship as `.cact`, so the generation comes from the
    // container's own tag rather than the file name. Guessing from the
    // extension would load a v3 model into the v2 engine and misread a
    // 196-byte header as 120.
    if model.ends_with(".cact") {
        match container_tag(&model) {
            Some(TAG_V3) => run_v3(&o, &model),
            Some(TAG) => run_v2(&o, &model),
            Some(other) => fail(&format!(
                "{model}: not a Needle container (tag {other:#010x})"
            )),
            None => fail(&format!("{model}: could not read the container header")),
        }
    } else {
        run_v1(&o, &model);
    }
}

/// Read the first word of a container, which states its generation.
fn container_tag(path: &str) -> Option<u32> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut w = [0u8; 4];
    f.read_exact(&mut w).ok()?;
    Some(u32::from_le_bytes(w))
}

fn run_v3(o: &Opts, model: &str) {
    if o.positional.len() < 3 {
        fail(&format!(
            "a .cact model takes <query> and <tools_json> (its tokenizer is embedded)\n\n{USAGE}"
        ));
    }
    let (query, tools) = (&o.positional[1], &o.positional[2]);

    if o.prefill_chunk.is_some() {
        eprintln!("note: --prefill-chunk applies to Needle 2 only; ignoring");
    }

    let engine =
        V3Engine::load(model).unwrap_or_else(|e| fail(&format!("Failed to load {model}: {e}")));
    let opts = V3Options {
        // v3 reasons before answering, so it needs more room than v2's 128.
        max_new_tokens: o
            .max_tokens
            .unwrap_or(needle_infer::v3_engine::DEFAULT_MAX_NEW_TOKENS),
        temperature: o.temperature.unwrap_or(0.0),
        seed: o.seed.unwrap_or(0),
        system: o.system.clone(),
        constrain: o.constrain,
        kv_precision: if o.kv_int8 {
            KvPrecision::Int8
        } else {
            KvPrecision::F32
        },
    };

    let result = if o.stream {
        let stderr = std::io::stderr();
        let r = engine.generate_with(query, tools, &opts, |_id, piece| {
            let mut h = stderr.lock();
            let _ = write!(h, "{piece}");
            let _ = h.flush();
        });
        eprintln!();
        r
    } else {
        engine.generate(query, tools, &opts)
    };

    if o.json_only {
        match needle_infer::v3_engine::extract_tool_call(&result.text) {
            Some(tc) => println!("{tc}"),
            None => {
                eprintln!("no <tool_call> in the output");
                process::exit(2);
            }
        }
    } else {
        println!("{}", result.text);
    }
}

fn run_v2(o: &Opts, model: &str) {
    if o.positional.len() < 3 {
        fail(&format!(
            "a .cact model takes <query> and <tools_json> (its tokenizer is embedded)\n\n{USAGE}"
        ));
    }
    let (query, tools) = (&o.positional[1], &o.positional[2]);

    let engine =
        V2Engine::load(model).unwrap_or_else(|e| fail(&format!("Failed to load {model}: {e}")));
    let opts = GenerateOptions {
        max_new_tokens: o.max_tokens.unwrap_or(128),
        temperature: o.temperature.unwrap_or(0.0),
        seed: o.seed.unwrap_or(0),
        system: o.system.clone(),
        constrain: o.constrain,
        prefill_chunk: o.prefill_chunk.unwrap_or(needle_core::v2::DEFAULT_CHUNK),
    };

    let result = if o.stream {
        let stderr = std::io::stderr();
        let r = engine.generate(query, tools, &opts, |_id, piece| {
            let mut h = stderr.lock();
            let _ = write!(h, "{piece}");
            let _ = h.flush();
        });
        eprintln!();
        r
    } else {
        engine.generate(query, tools, &opts, |_, _| {})
    };

    if o.json_only {
        match &result.tool_call {
            Some(tc) => println!("{tc}"),
            None => {
                eprintln!("no <tool_call> in the output");
                process::exit(2);
            }
        }
    } else {
        println!("{}", result.text);
    }
}

fn run_v1(o: &Opts, model: &str) {
    if o.positional.len() < 4 {
        fail(&format!(
            "v1 .safetensors weights also need <vocab.txt>; a .cact model does not\n\n{USAGE}"
        ));
    }
    for (flag, set) in [
        ("--max-tokens", o.max_tokens.is_some()),
        ("--temperature", o.temperature.is_some()),
        ("--seed", o.seed.is_some()),
        ("--system", o.system.is_some()),
        ("--json", o.json_only),
        ("--constrain", o.constrain),
        ("--kv-int8", o.kv_int8),
        ("--prefill-chunk", o.prefill_chunk.is_some()),
    ] {
        if set {
            fail(&format!("{flag} is v2 only; {model} is a v1 model"));
        }
    }
    let (vocab, query, tools) = (&o.positional[1], &o.positional[2], &o.positional[3]);

    let engine = NeedleEngine::load(model, vocab)
        .unwrap_or_else(|e| fail(&format!("Failed to load model: {e}")));

    let result = if o.stream {
        let stderr = std::io::stderr();
        let r = engine.run_stream(query, tools, |_id, piece| {
            let mut h = stderr.lock();
            let _ = write!(h, "{piece}");
            let _ = h.flush();
        });
        eprintln!();
        r
    } else {
        engine.run(query, tools)
    };

    println!("{}", result.text);
}
