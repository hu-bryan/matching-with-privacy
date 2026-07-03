#!/usr/bin/env python3
"""Thin wrapper over the experiment CLI."""

import sys

from private_matching.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or (args[0] not in ("run", "adversarial", "plot") and args[0].startswith("-")):
        args = ["run", *args]
    raise SystemExit(main(args))
