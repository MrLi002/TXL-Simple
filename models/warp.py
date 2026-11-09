"""Keep the upstream sampling convention for checkpoint/output compatibility."""

import torch
from torch import nn
from torch.nn import functional as F


class BackwardWarp(nn.Module):
    def __init__(self):
        super().__init__()
        self._grids = {}

    def forward(self, img, flow):
        _, _, h, w = img.shape
        key = (h, w, flow.device, flow.dtype)
        if key not in self._grids:
            y, x = torch.meshgrid(
                torch.arange(h, device=flow.device, dtype=flow.dtype),
                torch.arange(w, device=flow.device, dtype=flow.dtype),
                indexing="ij",
            )
            # Bound the cache when evaluating many image resolutions.
            if len(self._grids) >= 8:
                self._grids.clear()
            self._grids[key] = (x, y)
        x, y = self._grids[key]
        grid = torch.stack(
            (2 * ((x + flow[:, 0]) / w - 0.5), 2 * ((y + flow[:, 1]) / h - 0.5)),
            dim=-1,
        )
        return F.grid_sample(img, grid, align_corners=False)
