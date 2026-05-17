use needle_infer::NeedleEngine;
use std::env;
use std::io::Write as IoWrite;
use std::process;

const USAGE: &str = "\
Usage: needle-rs [OPTIONS] <weights.safetensors> <vocab.txt> <query> <tools_json>

Arguments:
  weights     Path to .safetensors weight file
  vocab       Path to vocabulary text file (one piece per line)
  query       User query string
  tools       JSON array of tool definitions

Options:
  --stream    Print each token to stderr as it is generated; final JSON to stdout
  --help      Print this message

Examples:
  needle-rs weights/needle.safetensors weights/vocab.txt \\
    \"What's the weather in Paris?\" \\
    '[{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"location\":{\"type\":\"string\"}}}]'

  needle-rs --stream weights/needle.safetensors weights/vocab.txt \\
    \"Book a flight\" '[{\"name\":\"book_flight\",\"description\":\"Book flight\",\"parameters\":{}}]'
";

fn main() {
    let raw_args: Vec<String> = env::args().collect();

    // Parse flags before positional args.
    let mut stream = false;
    let mut positional: Vec<&str> = Vec::new();

    let mut i = 1;
    while i < raw_args.len() {
        match raw_args[i].as_str() {
            "--stream" => stream = true,
            "--help" | "-h" => {
                print!("{USAGE}");
                process::exit(0);
            }
            arg if arg.starts_with('-') => {
                eprintln!("Unknown option: {arg}");
                eprintln!("{USAGE}");
                process::exit(1);
            }
            arg => positional.push(arg),
        }
        i += 1;
    }

    if positional.len() < 4 {
        eprintln!("{USAGE}");
        process::exit(1);
    }

    let weights_path = positional[0];
    let vocab_path = positional[1];
    let query = positional[2];
    let tools_json = positional[3];

    let engine = match NeedleEngine::load(weights_path, vocab_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            process::exit(1);
        }
    };

    let result = if stream {
        let stderr = std::io::stderr();
        engine.run_stream(query, tools_json, |_id, piece| {
            let mut h = stderr.lock();
            let _ = write!(h, "{piece}");
            let _ = h.flush();
        })
    } else {
        engine.run(query, tools_json)
    };

    if stream {
        eprintln!(); // newline after streamed tokens
    }
    println!("{}", result.text);
}
