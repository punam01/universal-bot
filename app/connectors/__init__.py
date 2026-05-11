"""Connector registry — auto-discovers sibling modules at import time."""
import importlib
import pkgutil

from core.registry import Registry

from .base import Connector, Document

connectors: Registry[Connector] = Registry("connectors")

for _, _name, _ in pkgutil.iter_modules(__path__):
    if not _name.startswith("_") and _name != "base":
        importlib.import_module(f"{__name__}.{_name}")

__all__ = ["connectors", "Connector", "Document"]
