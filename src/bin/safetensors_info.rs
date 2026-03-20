/// Print all tensor keys, shapes, and dtypes from a safetensors file.
///
/// Usage:
///   cargo run --bin safetensors_info --release -- model.safetensors

use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Inspect a safetensors file")]
struct Args {
    /// Path to the safetensors file.
    path: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let bytes = std::fs::read(&args.path)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;

    let mut keys: Vec<_> = st.tensors().into_iter().collect();
    keys.sort_by(|a, b| a.0.cmp(&b.0));

    let mut total_params = 0usize;
    let mut total_bytes = 0usize;

    println!("{:<70} {:>12}  {:>6}  {:>10}", "Key", "Shape", "Dtype", "Params");
    println!("{}", "─".repeat(102));

    for (name, view) in &keys {
        let shape: Vec<usize> = view.shape().to_vec();
        let numel: usize = shape.iter().product();
        let dtype = format!("{:?}", view.dtype());
        let shape_str = format!("{:?}", shape);
        println!("{:<70} {:>12}  {:>6}  {:>10}", name, shape_str, dtype, numel);
        total_params += numel;
        total_bytes += view.data().len();
    }

    println!("{}", "─".repeat(102));
    println!("{} tensors, {:.2}M parameters, {:.2} MB on disk",
        keys.len(),
        total_params as f64 / 1e6,
        total_bytes as f64 / 1e6,
    );

    Ok(())
}
