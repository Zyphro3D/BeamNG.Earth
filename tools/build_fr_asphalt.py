#!/usr/bin/env python3
"""build_fr_asphalt.py: the carriageways' asphalt (drawn as meshes, BNGE_ROADS_IN_GROUND=0) from CC0 photographs.

    cd server && python ../tools/build_fr_asphalt.py SRC_DIR

SRC_DIR holds the downloaded sets, one folder each with albedo.jpg, normal.jpg (OpenGL green), roughness.jpg and,
for the base, ao.jpg:
  aerial_asphalt_01   Poly Haven (CC0), 4K: ~30 m of a street seen from above, the base of every asphalt
  asphalt_01          Poly Haven (CC0): even town asphalt, the detail of "city"
  Asphalt026C         ambientCG (CC0): cracked and patched, the detail of "worn"
  Asphalt025C         ambientCG (CC0): dark and even, the detail of "fresh"
(ambientCG's files are renamed: <set>_2K-JPG_Color.jpg -> albedo.jpg, _NormalGL -> normal, _Roughness ->
roughness, _AmbientOcclusion -> ao.)

Writes mod/.../art/shapes/beamngEarth/fr_asphalt/*.dds (BC7, mips) and fr_asphalt.materials.json beside it. The
method is EarthNG's (github.com/Alezx311/earthng, MIT, Copyright (c) 2026 Alezx311: earthng/fast_art.road_sheets and
fast_style's road materials): one base over the road's repeat (ROAD_M), its purple aerial cast halved and its mean
set to an asphalt's albedo, and per look a high-pass detail over DETAIL_M that the game adds to it.
Sizes are kept small for a modest graphics card: base colour and normal BASE_PX, its roughness and occlusion DATA_PX,
the details DETAIL_PX (~17 MB in all).
"""
import json
import os
import sys

import numpy as np

ROAD_M, DETAIL_M = 30.0, 2.0
BASE_PX, DATA_PX, DETAIL_PX = 2048, 1024, 512
# linear albedo (France's asphalt in game: 0.12, EarthNG's brightest, read pale and pinkish beside the stock roads,
# 2026-11-13 A/B), roughness, the share of the aerial's purple cast kept
BASE_MEAN, BASE_ROUGH, BASE_SATURATION = 0.06, 0.8, 0.15
DETAIL_SIGMA = 12                                              # texels (at DETAIL_PX) a detail's slopes are kept within
LOOKS = {"city": ("asphalt_01", 0.08), "worn": ("Asphalt026C", 0.10), "fresh": ("Asphalt025C", 0.05)}
VFS = "/levels/beamng_earth/art/shapes/beamngEarth/fr_asphalt"


def _load(path, mode, size):
    from PIL import Image
    return np.asarray(Image.open(path).convert(mode).resize((size, size), Image.LANCZOS), np.float32) / 255.0


def _blur_wrap(a, sigma):
    """A Gaussian blur of a tiling image (it wraps at its edges), by FFT."""
    h, w = a.shape
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    k = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fx * fx + fy * fy))
    return np.real(np.fft.ifft2(np.fft.fft2(a) * k)).astype(np.float32)


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _u8(a):
    a = np.clip(np.asarray(a, np.float32), 0.0, 1.0)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return (a * 255.0 + 0.5).astype(np.uint8)


def _normal(path, size, sigma=None):
    """A normal map for the game (DirectX green: OpenGL's flipped), its slopes wider than sigma texels taken out."""
    n = _load(path, "RGB", size) * 2.0 - 1.0
    nx, ny = n[..., 0], n[..., 1]
    if sigma:
        nx, ny = nx - _blur_wrap(nx, sigma), ny - _blur_wrap(ny, sigma)
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
    return np.stack([nx, -ny, nz], -1) * 0.5 + 0.5


def _detail(a, sigma):
    """A high-pass of a around 0.5 (the game adds 2 * detail - 1 to the base)."""
    hp = a - _blur_wrap(a, sigma)
    return 0.5 + 0.5 * np.clip(hp / (4.0 * float(hp.std()) + 1e-6), -1.0, 1.0)


def build(src, out_dir):
    import bc7
    os.makedirs(out_dir, exist_ok=True)

    def write(name, img, srgb):
        with open(os.path.join(out_dir, name + ".dds"), "wb") as fh:
            fh.write(bc7.dds(_u8(img), srgb=srgb))
        return f"{VFS}/{name}.dds"
    base = os.path.join(src, "aerial_asphalt_01")
    lin = _srgb_to_linear(_load(os.path.join(base, "albedo.jpg"), "RGB", BASE_PX))
    lum = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lin = lum[..., None] + (lin - lum[..., None]) * BASE_SATURATION
    maps = {
        "baseColorMap": write("base_b.color", _linear_to_srgb(lin * (BASE_MEAN / float(lin.mean()))), True),
        "normalMap": write("base_nm.normal", _normal(os.path.join(base, "normal.jpg"), BASE_PX), False),
    }
    ra = _load(os.path.join(base, "roughness.jpg"), "L", DATA_PX)
    maps["roughnessMap"] = write("base_r.data", np.clip(ra - ra.mean() + BASE_ROUGH, 0.05, 1.0), False)
    ao = _load(os.path.join(base, "ao.jpg"), "L", DATA_PX)
    maps["ambientOcclusionMap"] = write("base_ao.data", 1.0 - (1.0 - ao) * 0.5, False)   # (cracks' shade, softened)
    materials = {}
    for look, (setname, grain) in LOOKS.items():
        d = os.path.join(src, setname)
        lum = _srgb_to_linear(_load(os.path.join(d, "albedo.jpg"), "L", DETAIL_PX))
        stage = dict(maps)
        stage.update({
            "detailMap": write(f"{look}_dc.data", _detail(lum / max(float(lum.mean()), 1e-4), DETAIL_SIGMA), False),
            "detailNormalMap": write(f"{look}_dn.normal", _normal(os.path.join(d, "normal.jpg"), DETAIL_PX,
                                                                  DETAIL_SIGMA), False),
            "roughnessDetailMap": write(f"{look}_dr.data", _detail(_load(os.path.join(d, "roughness.jpg"), "L",
                                                                         DETAIL_PX), DETAIL_SIGMA), False),
            "detailScale": [ROAD_M / DETAIL_M, ROAD_M / DETAIL_M],
            "detailBaseColorMapStrength": grain,
            "detailNormalMapStrength": 1.0,
            "detailRoughnessMapStrength": 0.15,
        })
        name = f"bne_fr_asphalt_{look}"
        materials[name] = {"name": name, "mapTo": name, "class": "Material", "version": 1.5,
                           "annotation": "ASPHALT", "groundType": "ASPHALT", "Stages": [stage, {}, {}, {}]}
    with open(os.path.join(out_dir, "fr_asphalt.materials.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(materials, fh, indent=2)
    return sorted(materials)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(__doc__)
    sys.path.insert(0, os.getcwd())
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "mod", "beamng_earth", "levels", "beamng_earth", "art", "shapes", "beamngEarth", "fr_asphalt")
    print("matériaux :", ", ".join(build(argv[0], out)))
    print("écrit dans", out)


if __name__ == "__main__":
    main()
