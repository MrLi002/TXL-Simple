"""Dataset preprocessing, separate from the model's learned temporal grouping."""

import numpy as np

# Known corrupt BS-ERGB intervals excluded by the upstream TimeLens-XL loader.
BAD_INTERVALS = {
    "basket_09": {"000031.npz", "000032.npz", "000033.npz", "000034.npz"},
    "may29_rooftop_handheld_02": {"000017.npz", "000070.npz"},
    "may29_rooftop_handheld_03": {"000306.npz"},
    "may29_rooftop_handheld_05": {"000121.npz"},
}


def bsergb_interval(path, bins, height, width, coordinate_mode="legacy"):
    """Same time bins, polarity, and bilinear spatial accumulation as upstream.

    BS-ERGB uses calibrated fixed-point coordinates. Pixel-coordinate exports
    must explicitly select coordinate_mode='pixel'. Timestamps are intentionally
    cast to float32 as in the original loader for preprocessing compatibility.
    """
    with np.load(path, allow_pickle=False) as data:
        x, y, t, p = [
            np.asarray(data[key], dtype=np.float32) for key in ("x", "y", "timestamp", "polarity")
        ]
    if not (len(x) == len(y) == len(t) == len(p)):
        raise ValueError(f"Unequal event array lengths: {path}")
    grid = np.zeros((bins, height, width), dtype=np.float32)
    if len(t) == 0:
        return grid
    if coordinate_mode == "legacy":
        x = x / (19968 * width / height) * (width - 1)
        y = y / 19968 * (height - 1)
    elif coordinate_mode != "pixel":
        raise ValueError("coordinate_mode must be legacy or pixel")
    if np.any(np.diff(t) < 0):
        raise ValueError(f"Events must be sorted by timestamp: {path}")
    if not np.isin(p, [0, 1]).all():
        raise ValueError(f"BS-ERGB polarity must be 0/1: {path}")
    step = (float(t[-1]) - float(t[0]) + 1) / bins
    indices = np.floor((t - t[0]) / step).astype(np.int64).clip(0, bins - 1)
    low_x, low_y = x.astype(np.int64), y.astype(np.int64)
    wx, wy, polarity = x - low_x, y - low_y, p * 2 - 1
    # Interleave the four contributions per event to retain accumulation order.
    xx = np.stack((low_x, low_x, low_x + 1, low_x + 1), 1)
    yy = np.stack((low_y, low_y + 1, low_y, low_y + 1), 1)
    values = (
        np.stack(((1 - wx) * (1 - wy), (1 - wx) * wy, wx * (1 - wy), wx * wy), 1)
        * polarity[:, None]
    )
    tt = np.broadcast_to(indices[:, None], xx.shape)
    valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    np.add.at(grid, (tt[valid], yy[valid], xx[valid]), values[valid])
    return grid


def read_voxels(paths, voxel_bins, height, width, dataset, coordinate_mode="legacy"):
    """Concatenate per-frame-interval voxels, matching TimeLens-XL preprocessing."""
    if not paths or voxel_bins % len(paths):
        raise ValueError("voxel_bins must be divisible by the number of frame intervals")
    per_interval = voxel_bins // len(paths)
    arrays = []
    for path in paths:
        if dataset == "bsergb":
            array = bsergb_interval(path, per_interval, height, width, coordinate_mode)
        else:
            # TimeLens-XL GOPRO/Adobe precomputed files use the 'data' key.
            with np.load(path, allow_pickle=False) as data:
                if "data" not in data:
                    raise ValueError(
                        f"{path}: expected TimeLens-XL precomputed voxel key 'data'; raw OF-AF events require separate conversion"
                    )
                array = np.asarray(data["data"], dtype=np.float32)
            if array.ndim == 2:
                array = array[None]
            if array.ndim != 3 or array.shape[-2:] != (height, width):
                raise ValueError(f"Invalid voxel shape in {path}: {array.shape}")
            channels = array.shape[0]
            if channels > per_interval:
                if channels % per_interval:
                    raise ValueError(f"Cannot sum {channels} channels to {per_interval}: {path}")
                array = array.reshape(per_interval, channels // per_interval, height, width).sum(1)
            elif channels < per_interval:
                array = np.pad(array, ((0, per_interval - channels), (0, 0), (0, 0)))
        arrays.append(array)
    return np.concatenate(arrays, axis=0)
