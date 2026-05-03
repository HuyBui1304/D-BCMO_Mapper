"""Shared setup for D_BCMO-Mapper notebooks.

Requires kmapper to be installed (pip install kmapper) with the five
D_BCMO-Mapper modules copied into the kmapper package directory.
See the top-level README for installation instructions.

Each notebook's cell 0 should start with:

    import sys, os
    sys.path.insert(0, os.path.abspath(".."))
    import _nb_setup  # noqa: F401
"""
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))

# Fall back to a local kepler-mapper fork if kmapper is not pip-installed.
try:
    import kmapper  # noqa: F401
except ImportError:
    _KMAPPER = os.path.normpath(os.path.join(_HERE, "..", "kepler-mapper"))
    if os.path.isdir(_KMAPPER) and _KMAPPER not in sys.path:
        sys.path.insert(0, _KMAPPER)

warnings.filterwarnings("ignore")
