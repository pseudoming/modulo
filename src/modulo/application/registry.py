"""Discover declared subclasses inside built-in or explicitly trusted packages."""

from importlib import import_module
from inspect import getmembers, isabstract, isclass
from pkgutil import iter_modules
from types import ModuleType
from typing import Any

from modulo.core.domain.interfaces import ProtocolAdapter, SourceAdapter, TokenizerBackend

ADAPTER_TYPES = {
    "sources": SourceAdapter,
    "protocols": ProtocolAdapter,
    "tokenizers": TokenizerBackend,
}


def _import(name: str) -> ModuleType:
    try:
        return import_module(name)
    except ImportError as exc:
        raise ValueError(f"cannot import plugin module: {name} ({type(exc).__name__})") from exc


def _modules(root: ModuleType):
    """Walk only this selected package, in stable order; skip private helpers."""
    yield root
    if hasattr(root, "__path__"):
        for info in sorted(iter_modules(root.__path__, root.__name__ + "."), key=lambda m: m.name):
            if not info.name.rsplit(".", 1)[-1].startswith("_"):
                yield from _modules(_import(info.name))


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {kind: {} for kind in ADAPTER_TYPES}

    def register(self, kind: str, name: str, plugin: Any) -> None:
        if kind not in self._items or not name:
            raise ValueError("unknown plugin kind or empty name")
        if name in self._items[kind]:
            raise ValueError(f"plugin already registered: {kind}/{name}")
        self._items[kind][name] = plugin

    def get(self, kind: str, name: str) -> Any:
        try:
            return self._items[kind][name]
        except KeyError as exc:
            raise ValueError(f"unknown plugin: {kind}/{name}") from exc

    def describe(self) -> dict[str, list[str]]:
        return {kind: sorted(plugins) for kind, plugins in self._items.items()}

    def discover(self, module_name: str) -> None:
        """Register locally declared concrete adapter classes, without global subclass state."""
        declarations = {}
        seen = set()
        for module in _modules(_import(module_name)):
            for _, candidate in getmembers(module, isclass):
                if (
                    candidate in seen
                    or candidate.__module__ != module.__name__
                    or isabstract(candidate)
                    or "name" not in candidate.__dict__
                ):
                    continue
                seen.add(candidate)
                kinds = [
                    kind for kind, base in ADAPTER_TYPES.items() if issubclass(candidate, base)
                ]
                if not kinds:
                    continue
                if len(kinds) != 1:
                    raise ValueError(f"adapter must inherit one category: {candidate.__qualname__}")
                name, version = candidate.name, getattr(candidate, "version", None)
                if any(
                    not isinstance(value, str) or not value.strip() for value in (name, version)
                ):
                    raise ValueError(f"adapter requires name and version: {candidate.__qualname__}")
                kind = kinds[0]
                if name in self._items[kind] or (kind, name) in declarations:
                    raise ValueError(f"plugin already registered: {kind}/{name}")
                declarations[kind, name] = candidate
        if not declarations:
            raise ValueError(f"no declared adapters in plugin module: {module_name}")
        # Validate the whole selection before modifying this registry. Sources and
        # tokenizers stay as factories; discovery opens no files, sockets or assets.
        prepared = {}
        for (kind, name), candidate in declarations.items():
            try:
                prepared[kind, name] = candidate() if kind == "protocols" else candidate
            except TypeError as exc:
                raise ValueError(f"protocol requires a no-argument constructor: {name}") from exc
        for (kind, name), plugin in prepared.items():
            self._items[kind][name] = plugin

    def load(self, module_name: str) -> None:
        # This is a trusted Python extension point, not a security sandbox.
        module = _import(module_name)
        hook = getattr(module, "register", None)
        if hook is None:
            self.discover(module_name)
        else:
            if not callable(hook):
                raise ValueError(f"plugin register hook is not callable: {module_name}")
            # A legacy hook owns its registration; do not also discover its classes.
            staged = Registry()
            staged._items = {kind: items.copy() for kind, items in self._items.items()}
            hook(staged)
            self._items = staged._items


def builtins() -> Registry:
    registry = Registry()
    for kind in ADAPTER_TYPES:
        registry.discover(f"modulo.adapters.{kind}")
    return registry
