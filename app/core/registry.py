"""Generic plugin registry.

A Registry holds named subclasses of a base type. Plugins register themselves
via the @register decorator (executed at import time). Available plugins
can be filtered to those whose runtime preconditions are met (e.g. required
env var present), via the optional `is_available()` classmethod.
"""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if key in self._items:
                raise ValueError(
                    f"{self.name}: '{key}' already registered as "
                    f"{self._items[key].__name__}"
                )
            self._items[key] = cls
            return cls

        return deco

    def get(self, key: str) -> type[T]:
        if key not in self._items:
            raise KeyError(
                f"{self.name}: unknown '{key}'. Registered: {sorted(self._items)}"
            )
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items)

    def available(self) -> list[str]:
        """Keys whose class.is_available() is True (or undefined)."""
        out = []
        for key, cls in self._items.items():
            check = getattr(cls, "is_available", None)
            if check is None or check():
                out.append(key)
        return sorted(out)
