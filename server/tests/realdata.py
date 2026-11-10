"""Test helper: whether the tests on real map data run.

Real-data tests build actual places (Baltimore's Fells Point, Pittsburgh's gas stations, the full EGM2008 model ...)
and download what the cache lacks, so they depend on the network and on what a machine happened to cache. They run
only when asked: BNGE_REAL_TESTS=1 (tools/run_tests.py --real), or a test group's own switch set to 1
(BNGE_FARGEO_REAL, BNGE_ROADGEO_REAL, BNGE_GAS_REAL, BNGE_GRAVITY_REAL). A group's switch at 0 keeps it off either way.
"""
import os


def wanted(own_switch):
    """Whether a real-data test group runs: its own switch (own_switch, an environment variable's name) when set,
    else BNGE_REAL_TESTS."""
    own = os.environ.get(own_switch)
    if own is not None and own != "":
        return own == "1"
    return os.environ.get("BNGE_REAL_TESTS") == "1"


REASON = "real map data and the network: BNGE_REAL_TESTS=1 (tools/run_tests.py --real)"
