"""Optional numerical regression against a local ORIGINAL TimeLens-XL checkout.

Usage: python tests/compare_upstream.py --upstream_root /path/to/TimeLens-XL
The reference source stays external to this simplified project. Only the unused
timer/warp imports are stubbed. The original CUDA-only grid allocation is mapped
to CPU for this CPU regression; network computations are unchanged.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import MyModel


class Timer:
    def __init__(self, *args):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def compare(arch_dir, *, flat=False):
    package = types.ModuleType("_original_tlx")
    package.__path__ = [str(arch_dir)]
    sys.modules[package.__name__] = package
    warp = types.ModuleType("_original_tlx.warp_layer")
    warp.warp = None  # Imported but never called by the upstream Large decoder.
    sys.modules[warp.__name__] = warp
    timer = types.ModuleType("tools.TimeTracker")
    timer.Ttracker = Timer
    sys.modules[timer.__name__] = timer
    for name, filename in [
        ("encoder_module", "encoder_module.py"),
        ("flow_decoder", "flow_decoder.py"),
        ("architecture", "architecture.py" if flat else "XXNet_final_attenfusion_arch.py"),
    ]:
        spec = importlib.util.spec_from_file_location("_original_tlx." + name, arch_dir / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    original_class = sys.modules["_original_tlx.architecture"].FinalBidirectionAttenfusion
    torch.set_num_threads(1)
    torch.manual_seed(123)
    original = original_class(
        base_channel=32, pos_e=0.2, neg_e=0.2, echannel=128, img_chn=6, num_decoder=8
    ).eval()
    simplified = MyModel().eval()
    simplified.load_state_dict(original.state_dict(), strict=True)
    print(f"Strict state_dict match: {len(original.state_dict())} tensors")
    results = []
    with torch.no_grad(), patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self):
        for ratio, batch in [(2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (4, 2)]:
            # Reset upstream mask cache: its cache key omits batch size.
            original.init_mask_dict.clear()
            im0, im1 = torch.rand(batch, 3, 8, 12), torch.rand(batch, 3, 8, 12)
            events = torch.randn(batch, 128, 8, 12)
            expected = original(torch.cat((im0, im1), 1), events, ratio)[0]
            actual = simplified(im0, im1, events, interp_ratio=ratio)
            error = float((expected - actual).abs().max())
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            results.append({"ratio": ratio, "batch": batch, "max_abs_error": error})
            print(results[-1])
        ratio = 16
        end_time = range(3, 15, 4)  # Official internal x16 / real x4 schedule.
        original.init_mask_dict.clear()
        expected = original(torch.cat((im0, im1), 1), events, ratio, end_time)[0]
        actual = simplified(im0, im1, events, [0.25, 0.5, 0.75], interp_ratio=16)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        print(
            {
                "internal_ratio": 16,
                "real_ratio": 4,
                "max_abs_error": float((expected - actual).abs().max()),
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream_root", required=True)
    parser.add_argument("--flat", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(args.upstream_root)
    compare(root if args.flat else root / "models" / "Expv8_large" / "archs", flat=args.flat)
