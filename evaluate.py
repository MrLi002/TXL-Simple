"""PSNR/SSIM evaluation for the single Large model; optional upstream LPIPS."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from models.model import MyModel
from util.checkpoint import load_weights
from util.image import write_image


def main(default_dataset=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["bsergb", "gopro"], default=default_dataset or "bsergb")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_root", required=True)
    p.add_argument(
        "--skips", nargs="+", type=int, help="Number of GT frames between endpoints, e.g. 1 3"
    )
    p.add_argument(
        "--interp_ratio", type=int, default=16, help="Internal model grid; 0 follows skip+1"
    )
    p.add_argument("--voxel_bins", type=int, default=128)
    p.add_argument("--save_dir", default="eval/timelensxl_large")
    p.add_argument("--save_pred", action="store_true")
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--num_worker", type=int, default=0)
    p.add_argument("--image_dir", default="images")
    p.add_argument("--event_dir", default="events")
    p.add_argument("--coordinate_mode", choices=["legacy", "pixel"], default="legacy")
    p.add_argument("--spatial_mode", choices=["official", "pad"], default="official")
    p.add_argument("--scenes", nargs="+")
    p.add_argument("--max_samples", type=int, default=0, help="Per skip; 0 evaluates all")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    skips = args.skips or ([1, 3] if args.dataset == "bsergb" else [7, 15])
    device = torch.device(args.device)
    model = (
        MyModel(voxel_bins=args.voxel_bins, interp_ratio=args.interp_ratio or 16).to(device).eval()
    )
    load_weights(model, args.checkpoint)
    perceptual = None
    if args.lpips:
        import pyiqa

        perceptual = pyiqa.create_metric("lpips", device=device)
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for skip in skips:
            data_ratio = skip + 1
            ratio = args.interp_ratio or data_ratio
            if ratio % data_ratio:
                raise ValueError("Internal model grid must contain every GT time")
            data = get_dataset(
                args.dataset,
                args.test_root,
                ratios=[data_ratio],
                voxel_bins=args.voxel_bins,
                training=False,
                image_dir=args.image_dir,
                event_dir=args.event_dir,
                coordinate_mode=args.coordinate_mode,
                spatial_mode=args.spatial_mode,
                scenes=args.scenes,
            )
            loader = DataLoader(data, batch_size=1, num_workers=args.num_worker)
            for sample_index, batch in enumerate(tqdm(loader, desc=f"skip={skip}")):
                if args.max_samples and sample_index >= args.max_samples:
                    break
                im0, im1, events, times = [
                    batch[k].to(device) for k in ("im0", "im1", "events", "times")
                ]
                h, w = int(batch["output_h"][0]), int(batch["output_w"][0])
                predictions = model(im0, im1, events, times, interp_ratio=ratio)[..., :h, :w].clamp(
                    0, 1
                )
                for target_index in range(predictions.shape[1]):
                    pred, target = predictions[0, target_index], batch["targets"][0, target_index]
                    array = pred.cpu().permute(1, 2, 0).numpy()
                    reference = target.permute(1, 2, 0).numpy()
                    window = min(7, h, w)
                    window = window if window % 2 else window - 1
                    if window < 3:
                        raise ValueError("Image too small for SSIM")
                    scene, start = batch["scene"][0], batch["start_name"][0]
                    name = batch["target_names"][target_index][0]
                    row = {
                        "skip": skip,
                        "scene": scene,
                        "start": start,
                        "target": name,
                        "psnr": float(peak_signal_noise_ratio(reference, array, data_range=1)),
                        "ssim": float(
                            structural_similarity(
                                reference, array, data_range=1, channel_axis=2, win_size=window
                            )
                        ),
                    }
                    if perceptual is not None:
                        row["lpips"] = float(perceptual(pred[None], target[None].to(device)))
                    rows.append(row)
                    if args.save_pred:
                        write_image(pred, output / str(skip) / scene / f"{start}_{name}.png")
    if not rows:
        raise RuntimeError("No frames evaluated")
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_keys = [k for k in ("psnr", "ssim", "lpips") if k in rows[0]]

    def summarize(group):
        return {
            "frames": len(group),
            **{k: float(np.mean([row[k] for row in group])) for k in metric_keys},
        }

    summary = {"args": vars(args), "overall": summarize(rows), "by_skip": {}, "by_scene": {}}
    for skip in skips:
        group = [row for row in rows if row["skip"] == skip]
        if group:
            summary["by_skip"][str(skip)] = summarize(group)
    for scene in sorted({row["scene"] for row in rows}):
        summary["by_scene"][scene] = summarize([row for row in rows if row["scene"] == scene])
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
