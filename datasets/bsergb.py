from .common import VFIData


class BSERGB(VFIData):
    def __init__(self, root, **kwargs):
        super().__init__(root, "bsergb", **kwargs)
