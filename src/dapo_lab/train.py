from __future__ import annotations

import sys
from pathlib import Path

from .runtime import ResearchRuntime


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = Path(argv[0]) if argv else Path("config/experiment.yaml")
    runtime = ResearchRuntime.from_path(config_path)
    runtime.launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
