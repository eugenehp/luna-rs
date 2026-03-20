/// download_weights — fetch LUNA model weights from HuggingFace.
///
/// Uses the `hf-hub` crate (same cache layout as Python `huggingface_hub`).
/// Cached files are returned instantly with no network traffic.
///
/// Usage:
///   cargo run --bin download_weights --release --features hf-download
///   cargo run --bin download_weights --release --features hf-download -- --variant large
///   cargo run --bin download_weights --release --features hf-download -- --all

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Download LUNA weights from HuggingFace (thorir/LUNA)")]
struct Args {
    /// HuggingFace repo ID.
    #[arg(long, default_value = "thorir/LUNA")]
    repo: String,

    /// Model variant to download: base, large, or huge.
    #[arg(long, default_value = "base")]
    variant: String,

    /// Download all three variants.
    #[arg(long)]
    all: bool,
}

const VARIANTS: &[(&str, &str)] = &[
    ("base",  "LUNA_base.safetensors"),
    ("large", "LUNA_large.safetensors"),
    ("huge",  "LUNA_huge.safetensors"),
];

fn main() -> Result<()> {
    let args = Args::parse();

    let variants: Vec<(&str, &str)> = if args.all {
        VARIANTS.to_vec()
    } else {
        let filename = VARIANTS.iter()
            .find(|(v, _)| *v == args.variant)
            .map(|&(_, f)| f)
            .ok_or_else(|| anyhow::anyhow!("Unknown variant '{}'. Use: base, large, huge", args.variant))?;
        vec![(&*args.variant, filename)]
    };

    for (variant, filename) in &variants {
        print!("Downloading LUNA-{variant} ({filename}) … ");
        let path = download(&args.repo, filename)?;
        println!("{}", path.display());
    }

    Ok(())
}

#[cfg(feature = "hf-download")]
fn download(repo: &str, filename: &str) -> Result<std::path::PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    let api = ApiBuilder::new().with_progress(true).build()?;
    Ok(api.model(repo.to_string()).get(filename)?)
}

#[cfg(not(feature = "hf-download"))]
fn download(_repo: &str, _filename: &str) -> Result<std::path::PathBuf> {
    anyhow::bail!("Compile with --features hf-download to enable weight downloading")
}
