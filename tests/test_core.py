"""Run: python -m unittest discover -s tests -v"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets import get_dataset
from models.model import MyModel
from util.checkpoint import load_weights
from util.events import read_voxels


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.model = MyModel().eval()

    def test_rates_and_grid_queries(self):
        image = torch.rand(1, 3, 8, 12)
        events = torch.rand(1, 128, 8, 12)
        for ratio in (2, 4, 8, 16, 32):
            with self.subTest(ratio=ratio), torch.no_grad():
                full = self.model(image, image, events, interp_ratio=ratio)
                chosen = self.model(image, image, events, [0.5], interp_ratio=ratio)
                self.assertEqual(full.shape, (1, ratio - 1, 3, 8, 12))
                torch.testing.assert_close(chosen[:, 0], full[:, ratio // 2 - 1], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "grid"):
            self.model(image, image, events, [0.3], interp_ratio=16)

    def test_training_gradients(self):
        image = torch.rand(1, 3, 8, 8)
        self.model.zero_grad(set_to_none=True)
        pred = self.model(image, image, torch.rand(1, 128, 8, 8), [0.5], interp_ratio=2)
        loss = (pred - image[:, None]).square().mean()
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.model.zero_grad(set_to_none=True)

    def test_official_checkpoint_prefixes_and_strictness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "official.pt"
            torch.save(
                {"model_state": {"net.module." + k: v for k, v in self.model.state_dict().items()}},
                path,
            )
            target = MyModel()
            load_weights(target, path)
            for key, value in self.model.state_dict().items():
                torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)
            state = self.model.state_dict()
            state.pop(next(iter(state)))
            torch.save({"model": state}, path)
            with self.assertRaises(RuntimeError):
                load_weights(target, path)

    def test_dataset_intervals_padding_and_names(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene"
            (scene / "images").mkdir(parents=True)
            (scene / "events").mkdir()
            for i in range(5):
                Image.fromarray(np.full((9, 11, 3), i * 20, np.uint8)).save(
                    scene / "images" / f"{i:06d}.png"
                )
            for i in range(4):
                np.savez(
                    scene / "events" / f"{i:06d}.npz", data=np.full((32, 9, 11), i, np.float32)
                )
            sample = get_dataset(
                "gopro", directory, ratios=[4], training=False, spatial_mode="pad"
            )[0]
            self.assertEqual(sample["events"].shape, (128, 32, 32))
            self.assertEqual(sample["targets"].shape, (3, 3, 9, 11))
            self.assertEqual(sample["target_names"], ["000001", "000002", "000003"])
            torch.testing.assert_close(sample["times"], torch.tensor([0.25, 0.5, 0.75]))
            self.assertEqual(sample["events"][96, 0, 0], 3)

    def test_bsergb_polarity_and_empty_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{i}.npz" for i in range(2)]
            np.savez(paths[0], x=[1, 1], y=[1, 1], timestamp=[0, 10], polarity=[1, 0])
            np.savez(paths[1], x=[], y=[], timestamp=[], polarity=[])
            voxel = read_voxels(paths, 8, 4, 4, "bsergb", "pixel")
            self.assertEqual(voxel[0, 1, 1], 1)
            self.assertEqual(voxel[3, 1, 1], -1)
            self.assertEqual(np.abs(voxel[4:]).sum(), 0)


if __name__ == "__main__":
    unittest.main()
