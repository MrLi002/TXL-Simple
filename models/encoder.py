"""Image/event encoders from TimeLens-XL-Large."""

import torch
from torch import nn


class EventsEncoder(nn.Module):
    def __init__(self, voxel_bins=128):
        super().__init__()
        self.e_channel = voxel_bins
        self.step = 8
        self.pre_conv = ResBlock(voxel_bins + 6, voxel_bins)
        self.ct = nn.Parameter(torch.ones(1, voxel_bins, 1, 1) * 0.2)
        self.down0 = nn.Sequential(nn.Conv2d(8, 64, 3, 2, 1), nn.LeakyReLU(0.2))
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(0.2))

    def forward(self, im0, im1, events, interp_ratio):
        current = self.e_channel // interp_ratio
        pre = self.pre_conv(torch.cat((im0, events, im1), 1)) * self.ct
        groups = pre.split(current, 1)
        batch, _, h, w = groups[0].shape
        if current > self.step:
            if current % self.step:
                raise ValueError("Channels per segment must be divisible by 8 when compressing")
            stacked = torch.stack(groups, 1)
            stacked = stacked.reshape(batch, interp_ratio, current // 8, 8, h, w).sum(2)
            groups = [stacked[:, i] for i in range(interp_ratio)]
        elif current < self.step:
            zeros = pre.new_zeros(batch, self.step - current, h, w)
            groups = [torch.cat((group, zeros), 1) for group in groups]
        encoded = self.down1(self.down0(torch.cat(groups, 0)))
        return torch.stack(encoded.split(batch, 0), 1)


class ResBlock(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True), nn.LeakyReLU(0.2)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        x_ = self.conv_in(x)
        x_res = self.conv_block(x)
        return x_res + x_


class ImageEncoder(nn.Module):
    def __init__(self, img_ch, base_channel):
        super().__init__()
        self.img_ch = img_ch
        self.preblock = ResBlock(img_ch, base_channel)
        self.down0 = nn.Conv2d(base_channel, 2 * base_channel, 3, 2, 1, bias=True)
        self.down1 = nn.Conv2d(2 * base_channel, 4 * base_channel, 3, 2, 1, bias=True)

    def forward(self, img):
        n, _, h, w = img.shape
        img = torch.cat(img.split(self.img_ch, 1), 0)
        preconv = self.preblock(img)
        im_half = self.down0(preconv)
        im_quater = self.down1(im_half)
        return (im_quater[:n], im_quater[n:], preconv[:n], preconv[n:])
