use std::env;
use std::process;
use needle_infer::NeedleEngine;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        eprintln!("Usage: needle-rs <weights.safetensors> <vocab.txt> <query> <tools_json>");
        eprintln!("  weights: path to .safetensors weight file");
        eprintln!("  vocab:   path to vocabulary text file (one piece per line)");
        eprintln!("  query:   the user query string");
        eprintln!("  tools:   JSON array of tool definitions");
        process::exit(1);
    }

    let weights_path = &args[1];
    let vocab_path   = &args[2];
    let query        = &args[3];
    let tools_json   = &args[4];

    let engine = match NeedleEngine::load(weights_path, vocab_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            process::exit(1);
        }
    };

    let result = engine.run(query, tools_json);
    println!("{}", result.text);
}
