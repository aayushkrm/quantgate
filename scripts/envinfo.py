#!/usr/bin/env python3
"""Print the environment record as JSON for the repo this script sits in.

The implementation lives in quantgate.envinfo, so that installed copies of the package can
collect the same record. This wrapper keeps the script entry point for manual use.
"""

import json
from pathlib import Path

from quantgate.envinfo import collect, cpu_flags

__all__ = ["collect", "cpu_flags"]

if __name__ == "__main__":
    print(json.dumps(collect(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
