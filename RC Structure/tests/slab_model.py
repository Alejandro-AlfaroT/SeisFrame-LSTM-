"""Tests of the composite slab model on the NoSlab branch.

The NoSlab branch (2026-10-21) runs with Structure_Parameters.SLAB_STRUCTURAL_PARTICIPATION = False: the floor is a
gravity load and a rigid diaphragm only, beams are rectangular, no slab reinforcement is designed. The composite slab
model stays in the code behind the switch, and its tests stay here so the two branches can be merged. ``composite_slab``
runs one test method, or a whole TestCase class including its setUpClass/setUp/tearDownClass, with the switch True.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

_RC_DIR = Path(__file__).resolve().parents[1]
if str(_RC_DIR) not in sys.path:
    sys.path.insert(0, str(_RC_DIR))
import Structure_Parameters as sp                                               # noqa: E402


def _patch():
    return mock.patch.object(sp, "SLAB_STRUCTURAL_PARTICIPATION", True)


def composite_slab(obj):
    """Decorator: run a test method, or a whole TestCase class, under the composite slab model."""
    if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
        own_setup, own_teardown = obj.__dict__.get("setUpClass"), obj.__dict__.get("tearDownClass")
        # one patcher per decorated class, so a decorated subclass of a decorated class nests two patches and stops both
        slot = f"_composite_slab_patch_{obj.__module__}_{obj.__qualname__}".replace(".", "_")

        def setUpClass(cls):
            patcher = _patch()
            setattr(cls, slot, patcher)
            patcher.start()
            try:
                if own_setup is not None:
                    own_setup.__func__(cls)
                else:
                    super(obj, cls).setUpClass()
            except BaseException:
                patcher.stop()
                raise

        def tearDownClass(cls):
            try:
                if own_teardown is not None:
                    own_teardown.__func__(cls)
                else:
                    super(obj, cls).tearDownClass()
            finally:
                getattr(cls, slot).stop()

        obj.setUpClass, obj.tearDownClass = classmethod(setUpClass), classmethod(tearDownClass)
        return obj
    return _patch()(obj)
