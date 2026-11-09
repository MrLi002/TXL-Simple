from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def read_image(path):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1)


def write_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        (tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    )
    Image.fromarray(array).save(path)


def pad_inputs(im0, im1, voxel, multiple=32):
    h, w = im0.shape[-2:]
    padding = (0, (-w) % multiple, 0, (-h) % multiple)
    return (
        F.pad(im0, padding, mode="replicate"),
        F.pad(im1, padding, mode="replicate"),
        F.pad(voxel, padding),
    )
