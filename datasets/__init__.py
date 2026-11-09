from .bsergb import BSERGB
from .gopro import GOPRO


def get_dataset(name, root, **kwargs):
    if name not in {"bsergb", "gopro"}:
        raise ValueError("Only bsergb and gopro loaders are included")
    return {"bsergb": BSERGB, "gopro": GOPRO}[name](root, **kwargs)
