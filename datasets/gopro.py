from .common import VFIData


class GOPRO(VFIData):
    """Loads the original TimeLens-XL precomputed 'data' voxel NPZ format."""

    def __init__(self, root, **kwargs):
        super().__init__(root, "gopro", **kwargs)
