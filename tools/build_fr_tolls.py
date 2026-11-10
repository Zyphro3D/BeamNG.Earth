"""Toll lanes of France, from an OpenStreetMap export (ODbL), for server/frtolls.py.

Input: an Overpass JSON export of France's barrier=toll_booth nodes and ways, made once by hand (the server never asks
Overpass at runtime, SOURCES.md). Only nodes are kept: in France each one sits on a toll lane's way; the ways are the
plazas' canopies (building=roof), already drawn as buildings.

    python tools/build_fr_tolls.py export.json [server/data/fr_toll_lanes.json.gz]

Rows: [lon * 1e7, lat * 1e7], sorted, gzip without a timestamp (frtolls.DATA_SHA256).
"""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "server" / "data" / "fr_toll_lanes.json.gz"


def main(src, out=OUT):
    doc = json.loads(Path(src).read_text(encoding="utf-8"))
    rows = sorted({(round(e["lon"] * 1e7), round(e["lat"] * 1e7)) for e in doc.get("elements", ())
                   if e.get("type") == "node" and e.get("tags", {}).get("barrier") == "toll_booth"})
    raw = gzip.compress(json.dumps({"format": 1, "source": "OpenStreetMap contributors (ODbL), barrier=toll_booth "
                                    "nodes, France", "rows": [list(r) for r in rows]},
                                   separators=(",", ":")).encode(), mtime=0)
    Path(out).write_bytes(raw)
    print(len(rows), "toll lanes,", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:])
