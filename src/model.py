from __future__ import annotations

import torch
from monai.networks.nets import SwinUNETR


def build_swinunetr(cfg: dict) -> torch.nn.Module:
    m = cfg["model"]
    model = SwinUNETR(
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        patch_size=2,
        depths=tuple(m.get("depths", [2, 2, 2, 2])),
        num_heads=tuple(m.get("num_heads", [3, 6, 12, 24])),
        feature_size=m["feature_size"],
        norm_name="instance",
        drop_rate=m.get("dropout_rate", 0.0),
        attn_drop_rate=m.get("attn_drop_rate", 0.0),
        dropout_path_rate=m.get("dropout_path_rate", 0.0),
        use_checkpoint=m.get("use_checkpoint", True),
    )
    return model
