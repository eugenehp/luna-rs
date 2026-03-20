/// Model and runtime configuration for LUNA inference.
///
/// `ModelConfig` mirrors the Python LUNA hyperparameters.
/// Field names match the HuggingFace `config.json` `"model"` sub-object.

// ── ModelConfig ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    /// Patch size in time-samples (default 40).
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,

    /// Number of learned cross-attention queries (default 4).
    #[serde(default = "default_num_queries")]
    pub num_queries: usize,

    /// Per-query / per-channel embedding dimension (default 64).
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,

    /// Number of Rotary Transformer encoder blocks (default 8).
    #[serde(default = "default_depth")]
    pub depth: usize,

    /// Number of attention heads per transformer block.
    /// Actual head count in the temporal encoder is `num_heads * num_queries`
    /// because the effective dim is `embed_dim * num_queries`.
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,

    /// MLP expansion ratio inside transformer blocks (default 4.0).
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,

    /// Number of output classes.  0 = reconstruction (pre-training).
    #[serde(default)]
    pub num_classes: usize,

    /// Drop-path rate for stochastic depth (default 0.0).
    #[serde(default)]
    pub drop_path: f64,

    /// Layer normalisation epsilon (default 1e-5).
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
}

fn default_patch_size()  -> usize { 40 }
fn default_num_queries() -> usize { 4 }
fn default_embed_dim()   -> usize { 64 }
fn default_depth()       -> usize { 8 }
fn default_num_heads()   -> usize { 2 }
fn default_mlp_ratio()   -> f64   { 4.0 }
fn default_norm_eps()    -> f64   { 1e-5 }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            patch_size:  default_patch_size(),
            num_queries: default_num_queries(),
            embed_dim:   default_embed_dim(),
            depth:       default_depth(),
            num_heads:   default_num_heads(),
            mlp_ratio:   default_mlp_ratio(),
            num_classes: 0,
            drop_path:   0.0,
            norm_eps:    default_norm_eps(),
        }
    }
}

impl ModelConfig {
    /// Effective hidden dimension after query concatenation: `embed_dim * num_queries`.
    pub fn hidden_dim(&self) -> usize {
        self.embed_dim * self.num_queries
    }

    /// FFN hidden dimension inside transformer blocks.
    pub fn ffn_hidden_dim(&self) -> usize {
        (self.hidden_dim() as f64 * self.mlp_ratio) as usize
    }

    /// Attention head dimension in the temporal encoder.
    pub fn head_dim(&self) -> usize {
        self.hidden_dim() / (self.num_heads * self.num_queries)
    }

    /// Total number of attention heads in the temporal encoder.
    pub fn total_heads(&self) -> usize {
        self.num_heads * self.num_queries
    }
}

// ── DataConfig ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DataConfig {
    /// Sampling rate after resampling (Hz).
    pub sample_rate: f32,
    /// Epoch duration in seconds.
    pub epoch_dur: f32,
    /// Bounding box for channel position normalisation (metres).
    pub xyz_min: [f32; 3],
    pub xyz_max: [f32; 3],
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            sample_rate: 256.0,
            epoch_dur:   5.0,
            xyz_min: [-0.12, -0.12, -0.12],
            xyz_max: [ 0.12,  0.12,  0.12],
        }
    }
}

impl DataConfig {
    /// Number of time samples per epoch.
    pub fn epoch_samples(&self) -> usize {
        (self.sample_rate * self.epoch_dur) as usize
    }
}
