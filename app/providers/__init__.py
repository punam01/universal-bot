"""LLM provider registry — auto-discovers sibling modules at import time."""
import importlib
import pkgutil

from core.registry import Registry

from .base import LLMProvider

providers: Registry[LLMProvider] = Registry("providers")

# Import every sibling module so each @providers.register(...) decorator fires.
for _, _name, _ in pkgutil.iter_modules(__path__):
    if not _name.startswith("_") and _name != "base":
        importlib.import_module(f"{__name__}.{_name}")
