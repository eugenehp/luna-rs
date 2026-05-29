//! RLX-backed LUNA inference (`rlx::Graph` + `rlx::Session`).
//!
//! Burn-backed types live at the crate root when `--features burn` is
//! enabled. Enable this module with `--features rlx`.

pub mod encoder;
pub mod graph;
pub mod io;
pub mod prepare;
pub mod rope_helpers;
pub mod weights;

pub use encoder::{EpochEmbedding, LunaEncoder, RunEpochOpts};
pub use io::{load_edf, load_fif, save_epochs, PreprocInfo, RlxEpoch};
