"""TimeLens-XL-Large recurrent flow decoder and image fusion network."""

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ResBlock, ResBlockIF, BackwardWarp


class FlowDecoder(nn.Module):
    def __init__(self, in_channel, base_channel, conv_base_channel):
        super().__init__()
        self.block_1 = ResBlock(in_channel, base_channel)
        self.block_3 = ResBlock(base_channel * 2 + 2, conv_base_channel)
        self.block_4 = ResBlock(base_channel * 4 + 2, conv_base_channel)
        self.conv_out2 = nn.Conv2d(conv_base_channel, 2, 3, 1, 1, bias=True)
        self.conv_out3 = nn.Conv2d(conv_base_channel, 2, 3, 1, 1, bias=True)
        self.backwarp = BackwardWarp()

    def forward(self, events, im1, last_flow, last_warp_res, prev_econv):
        encoder_in = torch.cat((last_warp_res, events, prev_econv), 1)
        block_1 = self.block_1(encoder_in)
        im_tiplus = block_1
        border_flow_in = torch.cat((last_flow, im1, im_tiplus), 1)
        block_3 = self.block_3(border_flow_in)
        flow0 = self.conv_out2(block_3) + last_flow
        warp_res = self.backwarp(im1, flow0)
        final_flow_in = torch.cat((im_tiplus, im1, flow0, warp_res, block_3), 1)
        block_4 = self.block_4(final_flow_in)
        flow = self.conv_out3(block_4) + flow0
        out_warp_res = self.backwarp(im1, flow)
        flow_large = F.interpolate(flow, scale_factor=4, mode="bilinear") * 4
        return (out_warp_res, flow, flow_large, im_tiplus)


class IFBlock(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.bwarp = BackwardWarp()
        self.resConv = ResBlockIF(in_channel, base_channel)
        self.out_conv = nn.Conv2d(base_channel, 5, 3, 1, 1, bias=True)

    def forward(self, i0, i1, i0tf, i1tf, ft0, ft1, m, t, scale, forward_feat=None, backward_feat=None):
        data_in = torch.cat((i0, i1, i0tf, i1tf), 1)
        if m is not None:
            data_in = torch.cat((data_in, m), 1)
        else:
            data_in = torch.cat((data_in, t), 1)
        data_in = torch.cat((data_in, ft0, ft1), 1)
        data_in_lowres = F.interpolate(data_in, scale_factor=scale, mode="bilinear")
        if forward_feat is not None:
            data_in_lowres = torch.cat((data_in_lowres, forward_feat, backward_feat), 1)
        after_conv_lowres = self.out_conv(self.resConv(data_in_lowres))
        after_conv_highres = F.interpolate(after_conv_lowres, scale_factor=1 / scale, mode="bilinear")
        Ft0 = ft0 + after_conv_highres[:, 1:3]
        Ft1 = ft1 + after_conv_highres[:, 3:]
        Mask = after_conv_highres[:, :1] if m is None else after_conv_highres[:, :1] + m
        I0tf = self.bwarp(i0, Ft0)
        I1tf = self.bwarp(i1, Ft1)
        return (Ft0, Ft1, I0tf, I1tf, Mask)


class MaskNet(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.sig = nn.Sigmoid()
        self.quater_block = IFBlock(in_channel + 2 * base_channel, base_channel)
        self.half_block = IFBlock(in_channel, base_channel // 2)
        self.channel_squeeze0 = nn.Conv2d(128, 32, 1, 1, 0, bias=True)
        self.channel_squeeze1 = nn.Conv2d(128, 32, 1, 1, 0, bias=True)
        self.refine_blocks = ResBlockIF(3 + 128, 64)
        self.conv_out = nn.Conv2d(64, 3, 1, 1, bias=True)

    def forward(self, im0, im1, i0tf, i1tf, t, f_t0, f_t1, forward_feat, backward_feat, im0_feat_full, im1_feat_full):
        qft0, qft1, qi0tf, qi1tf, qm = self.quater_block(im0, im1, i0tf, i1tf, f_t0, f_t1, None, t, 0.25, forward_feat, backward_feat)
        hft0, hft1, hi0tf, hi1tf, hm = self.half_block(im0, im1, qi0tf, qi1tf, qft0, qft1, qm, t, 0.5)
        mask = self.sig(hm)
        fuse_out = hi0tf * mask + (1 - mask) * hi1tf
        forward_feat_ld = self.channel_squeeze0(forward_feat)
        backward_feat_ld = self.channel_squeeze1(backward_feat)
        forward_feat_hd, backward_feat_hd = (F.interpolate(forward_feat_ld, scale_factor=4, mode="bilinear"), F.interpolate(backward_feat_ld, scale_factor=4, mode="bilinear"))
        refine_in = torch.cat((forward_feat_hd, backward_feat_hd, fuse_out, im0_feat_full, im1_feat_full), 1)
        refine_out = self.conv_out(self.refine_blocks(refine_in)) + fuse_out
        return (refine_out, hft0, hft1)
