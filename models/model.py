"""Only the TimeLens-XL-Large model, exposed through an OF-AF-style interface."""

import torch
from torch import nn

from .encoder import EventsEncoder, ImageEncoder
from .flow_decoder import FlowDecoder, MaskNet


class MyModel(nn.Module):
    def __init__(self, args=None, *, voxel_bins=128, interp_ratio=16):
        super().__init__()
        if args is not None:
            voxel_bins = getattr(args, "voxel_bins", voxel_bins)
            interp_ratio = getattr(args, "interp_ratio", interp_ratio)
        self.voxel_bins = int(voxel_bins)
        self.interp_ratio = int(interp_ratio)
        self.events_encoder = EventsEncoder(self.voxel_bins)
        self.im_encoder = ImageEncoder(3, 32)
        self.decoder_0 = FlowDecoder(384, 128, 128)
        self.decoder_1 = FlowDecoder(384, 128, 128)
        self.masknet = MaskNet(17, 128)

    def _indices(self, times, ratio, batch, device):
        if times is None:
            return torch.arange(1, ratio, device=device)
        times = torch.as_tensor(times, device=device)
        if times.ndim == 2:
            if times.shape[0] != batch:
                raise ValueError("interp_times batch size does not match images")
            if not torch.allclose(times, times[:1].expand_as(times)):
                raise ValueError("All samples in a batch must use the same query times")
            times = times[0]
        if times.ndim != 1 or times.numel() == 0:
            raise ValueError("interp_times must be a nonempty [T] or [B,T] array")
        nodes = times * ratio
        if not torch.allclose(nodes, nodes.round(), atol=1e-5, rtol=0):
            raise ValueError(
                "TimeLens-XL queries must lie on the interp_ratio grid; no arbitrary-time flow interpolation is added"
            )
        indices = nodes.round().long()
        if torch.any(indices < 1) or torch.any(indices >= ratio):
            raise ValueError("Query times must be strictly between 0 and 1")
        return indices

    def forward(self, im0, im1, event, interp_times=None, *, interp_ratio=None, return_aux=False):
        ratio = self.interp_ratio if interp_ratio is None else int(interp_ratio)
        if ratio < 2 or self.voxel_bins % ratio:
            raise ValueError("interp_ratio must be >= 2 and divide voxel_bins")
        if im0.ndim != 4 or im0.shape != im1.shape or im0.shape[1] != 3:
            raise ValueError("im0 and im1 must both have shape [B,3,H,W]")
        n, _, height, width = im0.shape
        if height % 4 or width % 4:
            raise ValueError("Image dimensions must be divisible by 4; pad/crop before calling the model")
        if event.shape != (n, self.voxel_bins, height, width):
            raise ValueError("event must have shape [B,voxel_bins,H,W]")
        indices = self._indices(interp_times, ratio, n, im0.device)

        e_out = self.events_encoder(im0, im1, event, ratio)
        a0, a1, full0, full1 = self.im_encoder(torch.cat((im0, im1), 1))
        n, _, _, h, w = e_out.shape
        flow0, flow1 = a0.new_zeros(n, 2, h, w), a1.new_zeros(n, 2, h, w)
        warped0, warped1, state0, state1 = a0, a1, a0, a1
        left, right = {}, {}
        # Each decoder is shared across time, as in the upstream Large model.
        for step in range(ratio - 1):
            warped0, flow0, large0, state0 = self.decoder_0(e_out[:, step], a0, flow0, warped0, state0)
            warped1, flow1, large1, state1 = self.decoder_1(e_out[:, ratio - step - 1], a1, flow1, warped1, state1)
            left[step + 1] = (large0, state0)
            right[ratio - step - 1] = (large1, state1)
        predictions, final0, final1 = [], [], []
        ones = im0.new_ones(n, 1, height, width)
        for node in indices.tolist():
            ft0, feat0 = left[node]
            ft1, feat1 = right[node]
            i0t = self.decoder_0.backwarp(im0, ft0)
            i1t = self.decoder_0.backwarp(im1, ft1)
            prediction, refined0, refined1 = self.masknet(im0, im1, i0t, i1t, (ratio - node) / ratio * ones, ft0, ft1, feat0, feat1, full0, full1)
            predictions.append(prediction)
            if return_aux:
                final0.append(refined0)
                final1.append(refined1)
        predictions = torch.stack(predictions, 1)
        if return_aux:
            return {
                "frames": predictions,
                "flow_t0": torch.stack(final0, 1),
                "flow_t1": torch.stack(final1, 1),
                "node_indices": indices,
            }
        return predictions


TimeLensXLLarge = MyModel
