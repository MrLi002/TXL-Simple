"""Residual blocks from the upstream Large model; parameter names are preserved."""

import torch
from torch import nn
from torch.nn import functional as F



class ResBlock(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True))
        self.conv_block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x_ = self.conv_in(x)
        x_res = self.conv_block(x_)
        return x_res + x_


class ResBlockIF(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True))
        self.conv_block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )
        self.conv_block1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x_ = self.conv_in(x)
        x_res = self.conv_block(x_) + x_
        x_res1 = self.conv_block1(x_res) + x_res
        return x_res1


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