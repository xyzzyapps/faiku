"""Split Fly.code into odor bins and vision bins. Not MaleCNS anatomy.

Odor-only (previous fly): 8 KC odor channels, no MLP.
Vision fly: 4 odor + 4 vision.
"""

N_ODOR = 4
N_VISION = 4
N_CHANNELS = N_ODOR + N_VISION
N_KC = N_CHANNELS  # code width; odor-only still uses 8 bins, all odor


def layout(vision: bool) -> tuple[int, int, int]:
    """(n_odor, n_vision, n_code)."""
    if vision:
        return N_ODOR, N_VISION, N_CHANNELS
    return N_CHANNELS, 0, N_CHANNELS
