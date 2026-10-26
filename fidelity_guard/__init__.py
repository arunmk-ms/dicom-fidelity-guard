"""DICOM Fidelity Guard - reference implementation (see ../ENABLEMENT.md)."""

__version__ = "0.1.0"

from .guard import verify  # noqa: E402
from .types import Decision, LayerResult, Operation, Report  # noqa: E402

__all__ = ["verify", "Decision", "LayerResult", "Operation", "Report", "__version__"]
