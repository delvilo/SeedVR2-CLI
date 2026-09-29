#!/usr/bin/env python3
"""Standalone CLI entry point. Importing it has no CUDA or argument side effects."""

from src.cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
