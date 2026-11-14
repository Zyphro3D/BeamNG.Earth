#!/usr/bin/env python3
"""build_race_kerbs.py: the race track kerbs' material (server/racekerbs.py: bne_race_kerb), drawn here, no source
images.

    cd server && python ../tools/build_race_kerbs.py

Writes mod/.../art/shapes/beamngEarth/race_kerbs/kerb_b.color.dds and kerb_nm.normal.dds (BC7, mips) and
race_kerbs.materials.json beside them. The texture's across (u) is the kerb's width, its along (v) one red and one
white stripe (racekerbs.STRIPE_M each: the chunks' texture length is two): painted concrete with a little wear, its
outer edge rounded down (the normal map) as a kerb's profile is.
"""
import json
import os
import sys

import numpy as np

PX = 256
RED, WHITE = (178, 24, 30), (226, 226, 222)
WEAR = 0.06                 # the paint's grain (share of its colour)
VFS = "/levels/beamng_earth/art/shapes/beamngEarth/race_kerbs"


def colour(px=PX, seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((px, px, 3), np.float32)
    img[: px // 2] = RED
    img[px // 2:] = WHITE
    grain = rng.normal(0.0, WEAR, (px, px, 1)).astype(np.float32)
    k = np.ones(5, np.float32) / 5          # (a soft grain, not a pixel's noise)
    for ax in (0, 1):
        grain = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, grain)
    img = img * (1.0 + grain)
    u = np.linspace(0.0, 1.0, px, dtype=np.float32)[None, :, None]
    img *= 1.0 - 0.10 * np.clip((u - 0.85) / 0.15, 0.0, 1.0)     # (the rounded outer edge a little darker)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def normal(px=PX):
    u = np.linspace(0.0, 1.0, px, dtype=np.float32)
    slope = np.where(u > 0.8, -(u - 0.8) / 0.2 * 0.6, 0.0)         # (the outer fifth rounds down)
    nx = np.repeat(slope[None, :], px, axis=0)
    ny = np.zeros_like(nx)
    nz = np.sqrt(np.clip(1.0 - nx * nx, 0.0, 1.0))
    return np.clip((np.stack([nx, ny, nz], -1) * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def build(out_dir):
    import bc7
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kerb_b.color.dds"), "wb") as fh:
        fh.write(bc7.dds(colour(), srgb=True))
    with open(os.path.join(out_dir, "kerb_nm.normal.dds"), "wb") as fh:
        fh.write(bc7.dds(normal(), srgb=False))
    name = "bne_race_kerb"
    mats = {name: {"name": name, "mapTo": name, "class": "Material", "version": 1.5, "annotation": "ASPHALT",
                   "groundType": "ASPHALT",
                   "Stages": [{"baseColorMap": f"{VFS}/kerb_b.color.dds", "normalMap": f"{VFS}/kerb_nm.normal.dds",
                               "roughnessFactor": 0.55}, {}, {}, {}]}}
    with open(os.path.join(out_dir, "race_kerbs.materials.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(mats, fh, indent=2)
    return sorted(mats)


def main():
    sys.path.insert(0, os.getcwd())
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "mod", "beamng_earth", "levels", "beamng_earth", "art", "shapes", "beamngEarth", "race_kerbs")
    print("matériaux :", ", ".join(build(out)))
    print("écrit dans", out)


if __name__ == "__main__":
    main()
