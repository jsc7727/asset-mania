# SPDX-License-Identifier: GPL-3.0-or-later
"""The file Blender is handed with `--python`.

Blender executes a script path, not a module, so this shim puts the worker package on
`sys.path` and calls its entry point. Keeping it separate means `entrypoint` stays
importable and testable as an ordinary module.
"""

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from asset_mania_blender import entrypoint

raise SystemExit(entrypoint.run(sys.argv[1:]))
