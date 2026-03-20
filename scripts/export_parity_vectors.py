#!/usr/bin/env python3
"""Export test vectors for Python↔Rust numerical parity verification.

Downloads LUNA-Base weights, runs a forward pass on synthetic input,
and saves input + output tensors to safetensors for Rust comparison.

Usage:
    python scripts/export_parity_vectors.py [--output tests/vectors/parity.safetensors]

Requires: torch, safetensors, huggingface_hub, mne
    pip install torch safetensors huggingface_hub mne
"""

import argparse
import os
import sys
import types
import numpy as np
import torch

# ── Mock torcheeg (only need SEED_CHANNEL_LIST constant) ─────────────────────
_torcheeg = types.ModuleType('torcheeg')
_torcheeg_ds = types.ModuleType('torcheeg.datasets')
_torcheeg_const = types.ModuleType('torcheeg.datasets.constants')
_torcheeg_const.SEED_CHANNEL_LIST = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ',
    'PO4','PO6','PO8','CB1','O1','OZ','O2','CB2'
]
_torcheeg.datasets = _torcheeg_ds
_torcheeg_ds.constants = _torcheeg_const
sys.modules['torcheeg'] = _torcheeg
sys.modules['torcheeg.datasets'] = _torcheeg_ds
sys.modules['torcheeg.datasets.constants'] = _torcheeg_const

# ── Mock mamba_ssm (LUNA.py imports it but we only load pretrained weights) ──
_mamba = types.ModuleType('mamba_ssm')
class _FakeMamba(torch.nn.Module):
    def __init__(self, **kwargs): super().__init__()
    def forward(self, x, **kwargs): return x
_mamba.Mamba = _FakeMamba
sys.modules['mamba_ssm'] = _mamba

# ── Mock rotary_embedding_torch ──────────────────────────────────────────────
# We need a real working RotaryEmbedding for the forward pass
try:
    import rotary_embedding_torch
except ImportError:
    print("Installing rotary_embedding_torch...", file=sys.stderr)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rotary_embedding_torch",
                          "--break-system-packages", "-q"])
    import rotary_embedding_torch

def main():
    parser = argparse.ArgumentParser(description="Export LUNA parity test vectors")
    parser.add_argument("--output", default="tests/vectors/parity.safetensors")
    parser.add_argument("--biofoundation-path", default=None)
    args = parser.parse_args()

    # Find BioFoundation repo
    bf_path = args.biofoundation_path
    if bf_path is None:
        for c in [os.path.expanduser("~/Desktop/BioFoundation"),
                  os.path.join(os.path.dirname(__file__), "..", "..", "BioFoundation")]:
            if os.path.isdir(c) and os.path.isfile(os.path.join(c, "models", "LUNA.py")):
                bf_path = os.path.abspath(c)
                break
    if not bf_path or not os.path.isdir(bf_path):
        print("ERROR: Cannot find BioFoundation repo.", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, bf_path)
    print(f"BioFoundation: {bf_path}")

    from models.LUNA import LUNA
    from models.modules.channel_embeddings import get_channel_indices, get_channel_locations

    # Download weights
    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download("thorir/LUNA", "LUNA_base.safetensors")
    print(f"Weights: {weights_path}")

    # Build model
    model = LUNA(patch_size=40, num_queries=4, embed_dim=64, depth=8, num_heads=2,
                 mlp_ratio=4.0, num_classes=0, drop_path=0.0)
    from safetensors.torch import load_file
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # ── Deterministic synthetic input ────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)

    CHN_ORDER = [
        "FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
        "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "A1-T3", "T4-A2",
    ]
    n_ch, n_t = len(CHN_ORDER), 1280

    # Synthetic signal: deterministic sine waves
    signal = np.zeros((n_ch, n_t), dtype=np.float32)
    for ch in range(n_ch):
        t = np.arange(n_t, dtype=np.float32) / 256.0
        signal[ch] = (np.sin(2*np.pi*(9+ch*0.3)*t)*20e-6
                     + np.sin(2*np.pi*(18+ch*0.5)*t)*5e-6
                     + np.sin(2*np.pi*(5+ch*0.2)*t)*15e-6)

    x_signal = torch.from_numpy(signal).unsqueeze(0)  # [1, 22, 1280]

    # Channel locations from MNE standard_1005
    channel_locations = np.stack(get_channel_locations(CHN_ORDER), axis=0)
    channel_locations = torch.from_numpy(channel_locations).float().unsqueeze(0)

    # Channel name indices
    channel_indices = get_channel_indices(CHN_ORDER)
    channel_names = torch.tensor(channel_indices).unsqueeze(0)

    # Channel-wise z-score
    eps = 1e-8
    mean = x_signal.mean(dim=2, keepdim=True)
    std = x_signal.std(dim=2, keepdim=True)
    x_normalized = (x_signal - mean) / (std + eps)

    # IMPORTANT: pass mask=None (not zeros tensor) to skip noise injection.
    # Python's prepare_tokens adds randn*0.02 noise when `mask is not None`,
    # even if the mask is all-zeros. At inference, mask should be None.
    mask = None

    # Forward — need to patch the model to accept None mask
    # The Python LUNA.forward requires a positional `mask` arg, so we pass None
    with torch.no_grad():
        # Call forward directly — LUNA.forward(x_signal, mask, channel_locations, channel_names)
        result = model(x_normalized, mask, channel_locations, channel_names)
        # With mask=None, returns (x_reconstructed, x_original, attention_scores)
        x_reconstructed, x_original, attention_scores = result

    print(f"\nInput:      {list(x_normalized.shape)}, mean={x_normalized.mean():.6f}, std={x_normalized.std():.6f}")
    print(f"Output:     {list(x_reconstructed.shape)}, mean={x_reconstructed.mean():.6f}, std={x_reconstructed.std():.6f}")
    print(f"Attn:       {list(attention_scores.shape)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    from safetensors.torch import save_file
    save_file({
        "input_signal":         x_signal.squeeze(0),           # [22, 1280]
        "input_normalized":     x_normalized.squeeze(0),       # [22, 1280]
        "channel_locations":    channel_locations.squeeze(0),   # [22, 3]
        "channel_names":        channel_names.squeeze(0).int(),# [22]
        "output_reconstructed": x_reconstructed.squeeze(0),    # [22, 1280]
        "attention_scores":     attention_scores,               # [32, 4, 22]
    }, args.output)

    print(f"\n✓ Saved → {args.output}")

if __name__ == "__main__":
    main()
