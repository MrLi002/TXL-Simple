"""Direct training entry point; no registry, BaseModel, or other architectures."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from models.model import MyModel
from util.checkpoint import load_weights, restore_rng, rng_state


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)



def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["gopro", "bsergb"], default="gopro")
    p.add_argument("--data_root", required=True)
    p.add_argument("--save_dir", default="train/timelensxl_large")
    p.add_argument("--epoch", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eta_min", type=float, default=1e-7)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_worker", type=int, default=8)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--voxel_bins", type=int, default=128)
    p.add_argument("--data_ratios", nargs="+", type=int)
    p.add_argument("--interp_ratio", type=int, default=16, help="Internal model grid; 0 follows each sample's data ratio")
    p.add_argument("--image_dir", default="images")
    p.add_argument("--event_dir", default="events")
    p.add_argument("--coordinate_mode", choices=["legacy", "pixel"], default="legacy")
    weights = p.add_mutually_exclusive_group()
    weights.add_argument("--weights_path", help="Official/simplified weights, weights only")
    weights.add_argument("--resume", help="Resume a checkpoint from this simplified trainer")
    p.add_argument("--only_weights", action="store_true", help="Accepted for OF-AF command compatibility; weights_path always loads weights only")
    p.add_argument("--lpips_weight", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=0.01)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    return p


def main():
    args = parser().parse_args()
    if args.epoch < 1 or args.batch_size < 1:
        raise ValueError("epoch and batch_size must be positive")
    ratios = args.data_ratios or ([4] if args.dataset == "bsergb" else [16])
    if len(ratios) > 1 and args.batch_size != 1:
        raise ValueError("Mixed data ratios require batch_size=1")
    if args.interp_ratio and any(args.interp_ratio % r for r in ratios):
        raise ValueError("Internal interp_ratio must be divisible by every data ratio; use --interp_ratio 0 for mixed GOPRO")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    data = get_dataset(args.dataset, args.data_root, ratios=ratios, voxel_bins=args.voxel_bins, crop_size=args.crop_size, image_dir=args.image_dir, event_dir=args.event_dir, coordinate_mode=args.coordinate_mode)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    model = MyModel(voxel_bins=args.voxel_bins, interp_ratio=args.interp_ratio or 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch * len(loader), eta_min=args.eta_min)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, saved_rng = 0, [], None

    if args.weights_path:
        load_weights(model, args.weights_path)

    if args.resume:
        checkpoint = load_weights(model, args.resume)
        required = {"optimizer", "scheduler", "scaler", "rng", "epoch", "args"}
        if not required.issubset(checkpoint):
            raise ValueError("This checkpoint supports weights-only loading; use --weights_path")
        for key in ("dataset", "voxel_bins", "interp_ratio", "epoch", "data_ratios", "batch_size", "crop_size", "lpips_weight", "grad_clip", "amp", "coordinate_mode", "image_dir", "event_dir"):
            if checkpoint["args"].get(key) != vars(args)[key]:
                raise ValueError(f"Resume requires unchanged {key}; use --weights_path to start a new run")
        if checkpoint.get("steps_per_epoch") != len(loader):
            raise ValueError("Dataset/batch length changed; use weights-only loading")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        saved_rng = checkpoint["rng"]
        start = checkpoint["epoch"] + 1
        history = checkpoint.get("history", [])

    perceptual = None
    if args.lpips_weight:
        import pyiqa
        perceptual = pyiqa.create_metric("lpips", as_loss=True, device=device)
        perceptual.eval()
        for parameter in perceptual.parameters():
            parameter.requires_grad_(False)

    # Metric construction can consume RNG state; restore only after all modules exist.
    if saved_rng is not None:
        restore_rng(saved_rng)

    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"TimeLens-XL-Large: {sum(p.numel() for p in model.parameters()):,} parameters; {len(data)} samples")

    for epoch in range(start, args.epoch):
        model.train()
        total, count = 0.0, 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epoch - 1}"):
            im0, im1, events, targets, times = [batch[k].to(device) for k in ("im0", "im1", "events", "targets", "times")]
            ratio = args.interp_ratio or int(batch["data_ratio"][0])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                pred = model(im0, im1, events, times, interp_ratio=ratio)
                loss = ((pred - targets).square() + 1e-6).sqrt().mean()
                if perceptual is not None:
                    loss = loss + args.lpips_weight * perceptual(pred.flatten(0, 1).float(), targets.flatten(0, 1).float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.detach())
            count += 1

        mean = total / count
        history.append({"epoch": epoch, "loss": mean, "lr": optimizer.param_groups[0]["lr"]})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "epoch": epoch, "args": vars(args), "steps_per_epoch": len(loader), "history": history}
        torch.save(state, output / f"epoch{epoch}.pth")
        temp = output / "last.tmp"
        torch.save(state, temp)
        temp.replace(output / "last.pth")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch} loss={mean:.6f}")


if __name__ == "__main__":
    main()






# def parser():
#     p = argparse.ArgumentParser(description=__doc__)
#     p.add_argument("--dataset", choices=["gopro", "bsergb"], default="gopro")
#     p.add_argument("--data_root", required=True)
#     p.add_argument("--save_dir", default="train/timelensxl_large")
#     p.add_argument("--epoch", type=int, default=20)
#     p.add_argument("--lr", type=float, default=1e-4)
#     p.add_argument("--eta_min", type=float, default=1e-7)
#     p.add_argument("--batch_size", type=int, default=1)
#     p.add_argument("--num_worker", type=int, default=8)
#     p.add_argument("--crop_size", type=int, default=256)
#     p.add_argument("--voxel_bins", type=int, default=128)
#     p.add_argument("--data_ratios", nargs="+", type=int)
#     p.add_argument(
#         "--interp_ratio",
#         type=int,
#         default=16,
#         help="Internal model grid; 0 follows each sample's data ratio",
#     )
#     p.add_argument("--image_dir", default="images")
#     p.add_argument("--event_dir", default="events")
#     p.add_argument("--coordinate_mode", choices=["legacy", "pixel"], default="legacy")
#     weights = p.add_mutually_exclusive_group()
#     weights.add_argument("--weights_path", help="Official/simplified weights, weights only")
#     weights.add_argument("--resume", help="Resume a checkpoint from this simplified trainer")
#     p.add_argument(
#         "--only_weights",
#         action="store_true",
#         help="Accepted for OF-AF command compatibility; weights_path always loads weights only",
#     )
#     p.add_argument("--lpips_weight", type=float, default=0.1)
#     p.add_argument("--grad_clip", type=float, default=0.01)
#     p.add_argument("--amp", action="store_true")
#     p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     p.add_argument("--seed", type=int, default=1234)
#     return p
#
#
# def main():
#     args = parser().parse_args()
#     if args.epoch < 1 or args.batch_size < 1:
#         raise ValueError("epoch and batch_size must be positive")
#     ratios = args.data_ratios or ([4] if args.dataset == "bsergb" else [16])
#     if len(ratios) > 1 and args.batch_size != 1:
#         raise ValueError("Mixed data ratios require batch_size=1")
#     if args.interp_ratio and any(args.interp_ratio % r for r in ratios):
#         raise ValueError(
#             "Internal interp_ratio must be divisible by every data ratio; use --interp_ratio 0 for mixed GOPRO"
#         )
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)
#     device = torch.device(args.device)
#     data = get_dataset(
#         args.dataset,
#         args.data_root,
#         ratios=ratios,
#         voxel_bins=args.voxel_bins,
#         crop_size=args.crop_size,
#         image_dir=args.image_dir,
#         event_dir=args.event_dir,
#         coordinate_mode=args.coordinate_mode,
#     )
#     loader = DataLoader(
#         data,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_worker,
#         pin_memory=device.type == "cuda",
#         worker_init_fn=seed_worker,
#     )
#     model = MyModel(voxel_bins=args.voxel_bins, interp_ratio=args.interp_ratio or 16).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch * len(loader), eta_min=args.eta_min)
#     amp = args.amp and device.type == "cuda"
#     scaler = torch.amp.GradScaler("cuda", enabled=amp)
#     start, history, saved_rng = 0, [], None
#     if args.weights_path:
#         load_weights(model, args.weights_path)
#     if args.resume:
#         checkpoint = load_weights(model, args.resume)
#         required = {"optimizer", "scheduler", "scaler", "rng", "epoch", "args"}
#         if not required.issubset(checkpoint):
#             raise ValueError("This checkpoint supports weights-only loading; use --weights_path")
#         for key in (
#             "dataset",
#             "voxel_bins",
#             "interp_ratio",
#             "epoch",
#             "data_ratios",
#             "batch_size",
#             "crop_size",
#             "lpips_weight",
#             "grad_clip",
#             "amp",
#             "coordinate_mode",
#             "image_dir",
#             "event_dir",
#         ):
#             if checkpoint["args"].get(key) != vars(args)[key]:
#                 raise ValueError(f"Resume requires unchanged {key}; use --weights_path to start a new run")
#         if checkpoint.get("steps_per_epoch") != len(loader):
#             raise ValueError("Dataset/batch length changed; use weights-only loading")
#         optimizer.load_state_dict(checkpoint["optimizer"])
#         scheduler.load_state_dict(checkpoint["scheduler"])
#         scaler.load_state_dict(checkpoint["scaler"])
#         saved_rng = checkpoint["rng"]
#         start = checkpoint["epoch"] + 1
#         history = checkpoint.get("history", [])
#     perceptual = None
#     if args.lpips_weight:
#         import pyiqa
#
#         perceptual = pyiqa.create_metric("lpips", as_loss=True, device=device)
#         perceptual.eval()
#         for parameter in perceptual.parameters():
#             parameter.requires_grad_(False)
#     # Metric construction can consume RNG state; restore only after all modules exist.
#     if saved_rng is not None:
#         restore_rng(saved_rng)
#     output = Path(args.save_dir)
#     output.mkdir(parents=True, exist_ok=True)
#     (output / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
#     print(f"TimeLens-XL-Large: {sum(p.numel() for p in model.parameters()):,} parameters; {len(data)} samples")
#     for epoch in range(start, args.epoch):
#         model.train()
#         total, count = 0.0, 0
#         for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epoch - 1}"):
#             im0, im1, events, targets, times = [batch[k].to(device) for k in ("im0", "im1", "events", "targets", "times")]
#             ratio = args.interp_ratio or int(batch["data_ratio"][0])
#             optimizer.zero_grad(set_to_none=True)
#             with torch.autocast(device_type=device.type, enabled=amp):
#                 pred = model(im0, im1, events, times, interp_ratio=ratio)
#                 loss = ((pred - targets).square() + 1e-6).sqrt().mean()
#                 if perceptual is not None:
#                     loss = loss + args.lpips_weight * perceptual(pred.flatten(0, 1).float(), targets.flatten(0, 1).float())
#             if not torch.isfinite(loss):
#                 raise FloatingPointError("Non-finite loss")
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()
#             total += float(loss.detach())
#             count += 1
#         mean = total / count
#         history.append({"epoch": epoch, "loss": mean, "lr": optimizer.param_groups[0]["lr"]})
#         state = {
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "scheduler": scheduler.state_dict(),
#             "scaler": scaler.state_dict(),
#             "rng": rng_state(),
#             "epoch": epoch,
#             "args": vars(args),
#             "steps_per_epoch": len(loader),
#             "history": history,
#         }
#         torch.save(state, output / f"epoch{epoch}.pth")
#         temp = output / "last.tmp"
#         torch.save(state, temp)
#         temp.replace(output / "last.pth")
#         (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
#         print(f"epoch={epoch} loss={mean:.6f}")
#
#
# if __name__ == "__main__":
#     main()

