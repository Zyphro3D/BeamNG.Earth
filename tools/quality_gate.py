#!/usr/bin/env python3
"""Bounded foundation regression gate. Uses synthetic data, not a gameplay benchmark.

Run after installing server/requirements-ci.lock and building tools/build_native.py. Every server test module runs but
the few in SLOW; the tests on real map data skip themselves (server/tests/realdata.py). The full server suite, SLOW
included, is tools/run_tests.py.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Every server test module runs, but these: too long for the gate's budget (each several minutes alone), or a drive
# that outlasts it; tools/run_tests.py runs them, as it runs everything. A new test module is in the gate by default.
SLOW = {
    "test_otto_bridge",         # ~13 min: real bridge profiles end to end
    "test_e2e",                 # ~6 min: a whole session through the launcher
    "test_credits",             # ~6 min: sessions with credits throttling
    "test_colormatch",          # ~5 min: imagery colour fields over many tiles
    "test_integration_drive",   # drives for 15 min and more
}


def modules():
    """The server's test modules the gate runs: all of server/tests/test_*.py but SLOW."""
    found = sorted(p.stem for p in (ROOT / "server" / "tests").glob("test_*.py"))
    return [m for m in found if m not in SLOW]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per Python test module")
    args = ap.parse_args()
    if args.par < 1 or args.timeout < 1:
        ap.error("--par and --timeout must be positive")
    # Optional cached-world tests must not silently change the CI workload on a developer's machine.
    # (the tests on real map data off whatever the environment says: they download, server/tests/realdata.py)
    for switch in ("BNGE_REAL_TESTS", "BNGE_FARGEO_REAL", "BNGE_ROADGEO_REAL", "BNGE_GAS_REAL", "BNGE_GRAVITY_REAL"):
        os.environ[switch] = "0"
    # Fail before tests instead of accepting skipped native/data-provider coverage.
    sys.path.insert(0, str(ROOT / "server"))
    import fast
    import duckdb
    import fsspec
    import lupa.luajit21
    import lupa.lua53
    import scipy.optimize       # test-only independent reference for the native carriageway QP
    fast.require(fast.MIN_VERSION, "quality gate")
    print(f"Python {sys.version.split()[0]}, earthfast {fast._ef.VERSION}, "
          f"DuckDB {duckdb.__version__}, fsspec {fsspec.__version__}", flush=True)
    commands = [
        [sys.executable, "tools/run_tests.py", *modules(), "--par", str(args.par),
         "--timeout", str(args.timeout), "--logs", "build/quality-tests"],
        [sys.executable, "tools/run_lua_tests.py"],
        ["node", "--test", "mod/tests/test_mapmath.mjs", "server/tests/test_mapmath.mjs"],
        ["cargo", "test", "--locked", "--manifest-path", "native/serverlist/Cargo.toml"],
        [sys.executable, "tools/package.py", "--beammp", "--out", "build/quality-packages"],
    ]
    for command in commands:
        print("+ " + " ".join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
