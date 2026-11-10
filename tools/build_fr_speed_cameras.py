"""Fixed speed cameras of France, from an OpenStreetMap export (ODbL), for server/frspeedcams.py.

Input: an Overpass JSON export of France's nodes tagged highway=speed_camera or enforcement=maxspeed, made once by hand
(the server never asks Overpass at runtime, SOURCES.md). Red-light and mobile cameras are left out. The output keeps
only the position, the enforced direction of travel when the tag gives one, and the support when it is tagged.

    python tools/build_fr_speed_cameras.py export.json [server/data/fr_speed_cameras.json.gz]

Rows: [lon * 1e7, lat * 1e7, heading in degrees clockwise from north or -1, support], support 0 unknown (a post), 1 a
"tourelle" column. Sorted, gzip without a timestamp: the same export gives the same bytes (frspeedcams.DATA_SHA256).
"""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "server" / "data" / "fr_speed_cameras.json.gz"
COMPASS = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180,
           "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
           "NORTH": 0, "EAST": 90, "SOUTH": 180, "WEST": 270}
NOT_SPEED = {"traffic_signals", "traffic-signals", "mobile"}


def heading(value):
    """Degrees clockwise from north, or -1 (no tag, both ways, several values, forward/backward along a way)."""
    v = str(value or "").strip().upper()
    if not v or ";" in v:
        return -1
    if v in COMPASS:
        return float(COMPASS[v])
    try:
        return round(float(v) % 360.0, 1)
    except ValueError:
        return -1


def rows_of(doc):
    rows = {}
    for e in doc.get("elements", ()):
        if e.get("type") != "node" or "lat" not in e or "lon" not in e:
            continue
        t = e.get("tags", {})
        if t.get("highway") != "speed_camera" and t.get("enforcement") != "maxspeed":
            continue
        if t.get("speed_camera") in NOT_SPEED or t.get("enforcement") in NOT_SPEED:
            continue
        tower = 1 if "tourelle" in (t.get("type", "") + t.get("camera:mount", "") + t.get("support", "")).lower() else 0
        key = (round(e["lon"] * 1e7), round(e["lat"] * 1e7))
        rows[key] = [key[0], key[1], heading(t.get("direction")), tower]
    return [rows[k] for k in sorted(rows)]


def main(src, out=OUT):
    rows = rows_of(json.loads(Path(src).read_text(encoding="utf-8")))
    doc = {"format": 1, "source": "OpenStreetMap contributors (ODbL), highway=speed_camera, France",
           "rows": rows}
    raw = gzip.compress(json.dumps(doc, separators=(",", ":")).encode(), mtime=0)
    Path(out).write_bytes(raw)
    print(len(rows), "cameras,", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:])
