"""Run the Large model on two RGB images plus a prepared full-interval voxel."""

import argparse
from pathlib import Path

import numpy as np
import torch

from models.model import MyModel
from util.checkpoint import load_weights
from util.image import pad_inputs, read_image, write_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--im0", required=True)
    p.add_argument("--im1", required=True)
    p.add_argument("--voxel", required=True, help="NPY [C,H,W] or NPZ with key data/voxel")
    p.add_argument("--interp_ratio", type=int, default=16)
    p.add_argument("--voxel_bins", type=int, default=128)
    p.add_argument("--times", nargs="+", type=float)
    p.add_argument("--save_dir", default="results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    im0, im1 = read_image(args.im0), read_image(args.im1)
    h, w = im0.shape[-2:]
    if Path(args.voxel).suffix.lower() == ".npz":
        with np.load(args.voxel, allow_pickle=False) as data:
            key = "data" if "data" in data else "voxel"
            array = np.asarray(data[key], dtype=np.float32)
    else:
        array = np.load(args.voxel, allow_pickle=False).astype(np.float32)
    if array.shape != (args.voxel_bins, h, w) or im0.shape != im1.shape:
        raise ValueError("Images/voxel must have matching spatial sizes and expected channels")
    im0, im1, events = pad_inputs(im0, im1, torch.from_numpy(array))
    model = (
        MyModel(voxel_bins=args.voxel_bins, interp_ratio=args.interp_ratio).to(args.device).eval()
    )
    load_weights(model, args.checkpoint)
    with torch.inference_mode():
        outputs = model(
            im0[None].to(args.device),
            im1[None].to(args.device),
            events[None].to(args.device),
            args.times,
            return_aux=True,
        )
    for frame, node in zip(outputs["frames"][0], outputs["node_indices"].tolist()):
        write_image(frame[:, :h, :w], Path(args.save_dir) / f"frame_{node:03d}.png")


if __name__ == "__main__":
    main()
