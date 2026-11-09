# Source attribution

This project is a structural simplification of OpenImagingLab/TimeLens-XL,
specifically models/Expv8_large. The network design and copied network layer
implementations belong to the original authors.

- Repository: https://github.com/OpenImagingLab/TimeLens-XL
- Source revision inspected: 414bc65fa6f1b8a45836c3233b37f7b4753cf3e9
- Paper: TimeLens-XL: Real-time Event-based Video Frame Interpolation with Large Motion (ECCV 2024).

Network layers retain their upstream state_dict names. The directory layout,
model wrapper, checkpoint handling, training/evaluation entry points and compact
dataset adapters were reorganized for this simplified project. No new license
is assigned to the upstream code. Consult the upstream repository/authors for
usage and redistribution terms.

OF-AF (https://github.com/Sudadaaaa/OF-AF) is the requested layout reference;
its BiFEB, Query, BiW and softsplat algorithms are not included.
