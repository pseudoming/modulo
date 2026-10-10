"""Stable command-line entry point."""

from .commands import main
from .parser import parser

__all__ = ["main", "parser"]
