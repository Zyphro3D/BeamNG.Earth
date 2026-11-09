"""France: IGN BD ORTHO's colours for a game's ground (BNE_IGN_LOOK=off leaves them).

Seen from a car, what reads well from above does not: its shadows are blue-green (47, 64, 67 under woods)
and its concrete yards and pale bare ground nearly white (230, 235, 232), which looks like snow at ground level. The
blue is taken out of the shadows, where it is a cast (a dark green tree keeps its colour), and the highlights are
compressed. Pure.
"""
import os

import numpy as np

ON = os.environ.get("BNE_IGN_LOOK", "on").lower() != "off"
SOURCE = "france"           # imagery_regional.REGIONAL's name for IGN BD ORTHO
SHADE_FULL, SHADE_NONE = 50.0, 95.0         # mean of RGB: a full shadow, no shadow
CAST_MIN, CAST_FULL = 2.0, 16.0             # blue above red: no cast, a full one
SHADE_TINT = np.array((0.96, 1.04, 0.82))   # a shadow on soil and grass, on its own brightness
SHADE_MIX = 0.85
KNEE = 165.0                # mean of RGB above which brightness is compressed
KNEE_SLOPE = 0.45
PALE_TINT = np.array((1.0, 0.985, 0.95))    # a pale surface: concrete rather than snow


def adjust(rgb):
    """A new uint8 (h, w, 3) image: rgb with the shadows' cast and the highlights corrected."""
    a = np.asarray(rgb, np.float32)
    lum = a.mean(axis=2)
    shade = np.clip((SHADE_NONE - lum) / (SHADE_NONE - SHADE_FULL), 0.0, 1.0)
    cast = np.clip((a[..., 2] - a[..., 0] - CAST_MIN) / (CAST_FULL - CAST_MIN), 0.0, 1.0)
    mix = (shade * cast * SHADE_MIX)[..., None]
    out = a * (1.0 - mix) + (lum[..., None] * SHADE_TINT) * mix
    over = np.maximum(lum - KNEE, 0.0)
    scale = (KNEE + over * KNEE_SLOPE) / np.maximum(KNEE + over, 1.0)
    pale = (over > 0)[..., None]
    out = out * scale[..., None]
    out = np.where(pale, out * PALE_TINT, out)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
