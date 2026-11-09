"""Standalone semantic find-and-report service for the aerial consequence planner.

Imports nothing from the training repo: the consequence model is vendored in
``nav.consequence_model`` and the DA3 depth head is loaded as a TorchScript
export (``tools/export_da3.py`` is the only training-repo-side step).
"""

__version__ = "0.1.0"
