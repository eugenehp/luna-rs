#!/usr/bin/env python3
"""Export intermediate tensors for debugging parity."""
import sys, types, torch, numpy as np

_tc = types.ModuleType('torcheeg'); _td = types.ModuleType('torcheeg.datasets')
_tdc = types.ModuleType('torcheeg.datasets.constants')
_tdc.SEED_CHANNEL_LIST = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']
_tc.datasets = _td; _td.constants = _tdc
sys.modules['torcheeg'] = _tc; sys.modules['torcheeg.datasets'] = _td; sys.modules['torcheeg.datasets.constants'] = _tdc
_m = types.ModuleType('mamba_ssm')
class _FM(torch.nn.Module):
    def __init__(self, **kw): super().__init__()
    def forward(self, x, **kw): return x
_m.Mamba = _FM; sys.modules['mamba_ssm'] = _m

sys.path.insert(0, '/home/user/Desktop/BioFoundation')
from models.LUNA import LUNA, nerf_positional_encoding
from safetensors.torch import load_file, save_file
import os

weights_path = '/home/user/.cache/huggingface/hub/models--thorir--LUNA/snapshots/8bdf01e4fafecf7146bfb2117a65520c04c96611/LUNA_base.safetensors'
model = LUNA(patch_size=40, num_queries=4, embed_dim=64, depth=8, num_heads=2, mlp_ratio=4.0, num_classes=0)
model.load_state_dict(load_file(weights_path), strict=True)
model.eval()

vecs = load_file('tests/vectors/parity.safetensors')
x_norm = vecs['input_normalized'].unsqueeze(0)
chan_locs = vecs['channel_locations'].unsqueeze(0)

with torch.no_grad():
    # Patch embed
    x_patched = model.patch_embed(x_norm)
    freq_embed = model.freq_embed(x_norm)
    
    # Channel location normalization
    channel_min = torch.min(chan_locs, dim=1, keepdim=True)[0]
    channel_max = torch.max(chan_locs, dim=1, keepdim=True)[0]
    normed_locs = (chan_locs - channel_min) / (channel_max - channel_min + 1e-8)
    
    # NeRF encoding
    nerf_enc = nerf_positional_encoding(normed_locs, 64)
    
    # Channel location embedder
    chan_loc_emb = model.channel_location_embedder(nerf_enc)

os.makedirs('tests/vectors', exist_ok=True)
save_file({
    'patch_embed_output': x_patched.squeeze(0),    # [704, 64]
    'freq_embed_output': freq_embed.squeeze(0),     # [704, 64]
    'normed_locs': normed_locs.squeeze(0),          # [22, 3]
    'nerf_encoded': nerf_enc.squeeze(0),            # [22, 64]
    'chan_loc_emb': chan_loc_emb.squeeze(0),         # [22, 64]
}, 'tests/vectors/intermediates.safetensors')
print('Saved tests/vectors/intermediates.safetensors')
for k in ['patch_embed_output', 'freq_embed_output', 'normed_locs', 'nerf_encoded', 'chan_loc_emb']:
    t = eval(k.replace('output', 'output') if 'output' not in k else k)
    # Just print from the saved dict
print('Done')
