"""Execution ports: the only modules in this package permitted to touch an engine.

`adapter.py` stays free of model loaders, subprocesses, and network clients, and its test
scans that source to prove it. The engine has to be reached from somewhere, so it is reached
from here, where a separate test scans for the different failure this layer can have: an
acquisition step hidden inside a run step.

A port may load weights that are already on disk. A port may not download them, and may not
accept a licence on the user's behalf. Acquisition is a step the user takes knowingly, with
the licence texts in front of them; a port that fetched its own weights would turn that into
a side effect of pressing run.
"""

from asset_mania_engine_triposr.ports.triposr import (
    TripoSRPort,
    TripoSRSettings,
    WeightsMissing,
)

__all__ = ["TripoSRPort", "TripoSRSettings", "WeightsMissing"]
