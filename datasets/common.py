"""A scene is images/*.png plus chronologically ordered interval events/*.npz."""

import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.events import BAD_INTERVALS, read_voxels
from util.image import pad_inputs, read_image


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


class VFIData(Dataset):
    def __init__(
        self,
        root,
        dataset,
        *,
        ratios=(4,),
        voxel_bins=128,
        training=True,
        crop_size=256,
        image_dir="images",
        event_dir="events",
        coordinate_mode="legacy",
        spatial_mode="official",
        scenes=None,
    ):
        self.dataset, self.ratios = dataset, tuple(ratios)
        if not self.ratios or any(r < 2 or voxel_bins % r for r in self.ratios):
            raise ValueError("Each data ratio must be >= 2 and divide voxel_bins")
        self.voxel_bins, self.training = voxel_bins, training
        self.crop_size, self.coordinate_mode = crop_size, coordinate_mode
        self.spatial_mode = spatial_mode
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        scene_dirs = (
            [root]
            if (root / image_dir).is_dir()
            else sorted(
                (p for p in root.iterdir() if p.is_dir() and (p / image_dir).is_dir()),
                key=natural_key,
            )
        )
        if scenes:
            scene_dirs = [p for p in scene_dirs if p.name in scenes]
        self.samples = {r: [] for r in self.ratios}
        for scene in scene_dirs:
            images = sorted(
                (
                    p
                    for p in (scene / image_dir).iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ),
                key=natural_key,
            )
            events = sorted((scene / event_dir).glob("*.npz"), key=natural_key)
            for ratio in self.ratios:
                stride = 1 if training else ratio
                for start in range(0, len(images) - ratio, stride):
                    intervals = events[start : start + ratio]
                    if len(intervals) != ratio:
                        continue
                    bad = BAD_INTERVALS.get(scene.name, set()) if dataset == "bsergb" else set()
                    if any(p.name in bad for p in intervals):
                        continue
                    self.samples[ratio].append(
                        (scene.name, images[start : start + ratio + 1], intervals)
                    )
        if any(not items for items in self.samples.values()):
            empty = [r for r, items in self.samples.items() if not items]
            raise ValueError(
                f"No usable samples for ratios {empty} under {root}; expected {image_dir}/ and {event_dir}/ in each scene"
            )
        if not training and len(self.ratios) != 1:
            raise ValueError("Evaluate one data ratio at a time")
        self.base_ratio = min(self.ratios)

    def __len__(self):
        return len(self.samples[self.base_ratio])

    def __getitem__(self, index):
        # Match the original mixed-ratio sampling probabilities when all five
        # official ratios are selected. Mixed batches require batch_size=1.
        weights = [0.1, 0.1, 0.2, 0.3, 0.3] if self.ratios == (2, 4, 8, 16, 32) else None
        ratio = (
            random.choices(self.ratios, weights=weights, k=1)[0]
            if self.training
            else self.base_ratio
        )
        scene, images, intervals = self.samples[ratio][min(index, len(self.samples[ratio]) - 1)]
        frames = torch.stack([read_image(p) for p in images])
        h, w = frames.shape[-2:]
        voxel = torch.from_numpy(
            read_voxels(intervals, self.voxel_bins, h, w, self.dataset, self.coordinate_mode)
        )
        if self.training:
            size = self.crop_size
            if not size or size % 4 or size > min(h, w):
                raise ValueError("crop_size must fit the image and be divisible by 4")
            margin = 20 if self.dataset == "bsergb" and min(h, w) >= size + 40 else 0
            top = random.randint(margin, h - size - margin)
            left = random.randint(margin, w - size - margin)
            frames, voxel = (
                frames[..., top : top + size, left : left + size],
                voxel[..., top : top + size, left : left + size],
            )
        elif self.spatial_mode == "official":
            # Preserve upstream evaluation crop (BS-ERGB removes one 32px block).
            oh, ow = (
                ((h // 32 - 1) * 32, (w // 32 - 1) * 32)
                if self.dataset == "bsergb"
                else (h // 32 * 32, w // 32 * 32)
            )
            if min(oh, ow) <= 0:
                raise ValueError("Image too small for official crop; use spatial_mode='pad'")
            top, left = ((h - oh) // 2, (w - ow) // 2) if self.dataset == "bsergb" else (0, 0)
            frames, voxel = (
                frames[..., top : top + oh, left : left + ow],
                voxel[..., top : top + oh, left : left + ow],
            )
        im0, im1, targets = frames[0], frames[-1], frames[1:-1]
        output_shape = targets.shape[-2:]
        if not self.training and self.spatial_mode == "pad":
            im0, im1, voxel = pad_inputs(im0, im1, voxel)
        return {
            "im0": im0,
            "im1": im1,
            "events": voxel,
            "targets": targets,
            "times": torch.arange(1, ratio, dtype=torch.float32) / ratio,
            "data_ratio": ratio,
            "scene": scene,
            "start_name": images[0].stem,
            "target_names": [p.stem for p in images[1:-1]],
            "output_h": output_shape[0],
            "output_w": output_shape[1],
        }
