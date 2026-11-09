"""France: the canopy-height raster from IGN's LiDAR HD (MNH, Etalab Licence Ouverte 2.0) in place of the world model.

maprasters' canopy is Meta / WRI's estimate from satellite images; where IGN has flown LiDAR HD its MNH (height above
ground, 0.5 m) is measured. The WMS gives it on the canopy grid of the cell (EPSG:3857, the same uint8 metres), so
vegetation.py is unchanged. A pixel with no lidar keeps the base raster's value. Off: BNE_IGN_MNH=off.
"""
import logging
import os
import time

import numpy as np

log = logging.getLogger("earthd.ign_lidar")

ON = os.environ.get("BNE_IGN_MNH", "on").lower() != "off"
WMS_URL = "https://data.geopf.fr/wms-r/wms"
LAYER = "IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
LANE = "ign_wfs"
FRANCE = (-5.5, 41.0, 9.8, 51.3)
MERC_HALF = 20037508.342789244
NODATA = 255
BLOCK = 512                 # pixels a request: the cell is read in blocks
BLOCK_TRIES = 3


def in_france(cell):
    lon0, lat0, lon1, lat1 = cell.bounds()
    return lon1 >= FRANCE[0] and lon0 <= FRANCE[2] and lat1 >= FRANCE[1] and lat0 <= FRANCE[3]


def to_canopy(values):
    """Metres (float32, <-100 or >200 = no data) to the canopy raster's uint8; NODATA where there is none."""
    ok = np.isfinite(values) & (values > -100) & (values < 200)
    out = np.full(values.shape, NODATA, np.uint8)
    out[ok] = np.clip(np.rint(values[ok]), 0, 254).astype(np.uint8)
    return out


def _url(x0, y0, x1, y1, w, h):
    return (f"{WMS_URL}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={LAYER}&STYLES=&CRS=EPSG:3857"
            f"&BBOX={x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}&WIDTH={w}&HEIGHT={h}&FORMAT=image/x-bil;bits=32")


def _block(fetcher, url):
    """The block's body, or None when the service keeps refusing it (a 400 that a retry answers, seen 2026-11-09)."""
    for attempt in range(BLOCK_TRIES):
        try:
            return fetcher.get(url, kind="demand", lane=LANE).body
        except Exception as e:      # (the block keeps the base raster: one refusal must not lose the whole cell)
            if attempt == BLOCK_TRIES - 1:
                log.warning("ign_lidar block: %s: %s", type(e).__name__, e)
                return None
            time.sleep(0.5 * (attempt + 1))


def measured(fetcher, cell, npx):
    """The cell's MNH as canopy uint8 on an npx² grid, or None where the service has nothing."""
    size = 2 * MERC_HALF / (1 << cell.z)
    px = size / npx
    mx0, my0 = cell.x * size - MERC_HALF, MERC_HALF - cell.y * size
    out = np.full((npx, npx), NODATA, np.uint8)
    for r0 in range(0, npx, BLOCK):
        for c0 in range(0, npx, BLOCK):
            r1, c1 = min(npx, r0 + BLOCK), min(npx, c0 + BLOCK)
            url = _url(mx0 + c0 * px, my0 - r1 * px, mx0 + c1 * px, my0 - r0 * px, c1 - c0, r1 - r0)
            body = _block(fetcher, url)
            if body is None or len(body) != (r1 - r0) * (c1 - c0) * 4:   # (an XML error instead of the grid: this block has none)
                continue
            out[r0:r1, c0:c1] = to_canopy(np.frombuffer(body, "<f4").reshape(r1 - r0, c1 - c0))
    return None if (out == NODATA).all() else out


def canopy(fetcher, cell, base, make):
    """base (a Raster or None) with IGN's measured heights put in; make(data) builds the Raster when base is None.
    Returns (raster, names of the sources used). A failure leaves base as it is."""
    if not ON or not in_france(cell):
        return base, []
    npx = base.data.shape[0] if base is not None else 2048
    try:
        got = measured(fetcher, cell, npx)
    except Exception as e:      # (no network, a changed service: the world model stands)
        log.warning("ign_lidar %s: %s: %s", cell, type(e).__name__, e)
        return base, []
    if got is None:
        return base, []
    if base is None:
        return make(got), ["ign-lidar-hd-mnh"]
    data = base.data.copy()
    ok = got != NODATA
    data[ok] = got[ok]
    base.data = data
    return base, ["ign-lidar-hd-mnh"]
