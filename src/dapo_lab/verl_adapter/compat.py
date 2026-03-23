from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass


SUPPORTED_VERL_COMMIT = "08e030d9b0d6f3c5c2f154ec28bf2ccb37cab375"


@dataclass(slots=True)
class CompatibilityReport:
    importable: bool
    detected_commit: str | None
    required_commit: str
    compatible: bool
    message: str


def check_verl_compatibility(*, required_commit: str = SUPPORTED_VERL_COMMIT, strict: bool = False) -> CompatibilityReport:
    spec = importlib.util.find_spec("verl")
    if spec is None:
        message = (
            "verl is not installed. Install a compatible checkout before launching training. "
            f"Expected commit: {required_commit}."
        )
        if strict:
            raise RuntimeError(message)
        return CompatibilityReport(
            importable=False,
            detected_commit=None,
            required_commit=required_commit,
            compatible=False,
            message=message,
        )

    import verl  # type: ignore

    detected_commit = getattr(verl, "__commit__", None) or os.environ.get("VERL_COMMIT")
    compatible = detected_commit in {None, required_commit}
    message = (
        f"Detected verl commit {detected_commit or 'unknown'}; expected {required_commit}."
        if not compatible
        else f"verl import succeeded; commit {detected_commit or 'unknown'} accepted."
    )
    if strict and not compatible:
        raise RuntimeError(message)
    return CompatibilityReport(
        importable=True,
        detected_commit=detected_commit,
        required_commit=required_commit,
        compatible=compatible,
        message=message,
    )
