"""Test helper: patch a name of a split module everywhere it is bound.

roads.py, grading.py, buildings.py and earthd.py were split into layers (roadbase, roadspecs ..., gradbase, gradchains ...,
bldbase ...): a setting defined in a base module is imported by name into the modules that read it, so patching it on
one module leaves the others' copies as they were. patch_all patches it on every loaded module of the family that holds
the same object, and puts them all back after. Used like mock.patch.object: a with block, a decorator, or
start() / stop().

    with patch_all(grading, "PAD_ON", False):
        ...
"""
import contextlib
import sys
from unittest import mock

FAMILIES = {"roads": ("road",), "grading": ("grad",), "buildings": ("bld", "buildings"),
            "earthd": ("earth", "srv", "conn")}


class patch_all(contextlib.ContextDecorator):    # noqa: N801 (named like the mock.patch it stands for)
    def __init__(self, module, name, value=mock.DEFAULT, **mock_kwargs):
        # (no value: one MagicMock made with mock_kwargs, e.g. return_value= or side_effect=, put in every module)
        self.module, self.name = module, name
        self.value = mock.MagicMock(**mock_kwargs) if value is mock.DEFAULT else value
        self._stack = None

    def _targets(self):
        original = getattr(self.module, self.name)
        prefixes = FAMILIES.get(self.module.__name__, (self.module.__name__,))
        found = [m for n, m in list(sys.modules.items())
                 if m is not None and n.startswith(prefixes) and getattr(m, self.name, mock.DEFAULT) is original]
        return found if self.module in found else found + [self.module]

    def start(self):
        self._stack = contextlib.ExitStack()
        for m in self._targets():
            self._stack.enter_context(mock.patch.object(m, self.name, self.value))
        return self.value

    def stop(self):
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
