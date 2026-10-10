"""Locate repository fixtures independently of the test file layout."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
