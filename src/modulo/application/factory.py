"""Create the configured adapter registry and local reference calculator."""

from pathlib import Path

from modulo.core.domain.bindings import Binding, Bindings
from modulo.core.domain.interfaces import AssetError, TokenizerBackend
from modulo.core.processing.recalculation import Recalculator

from .registry import Registry, builtins


def build_registry(plugin_modules: list[str]) -> Registry:
    registry = builtins()
    for name in plugin_modules:
        registry.load(name)
    return registry


def make_calculator(registry: Registry, root: Path, bindings: Bindings, lock: dict) -> Recalculator:
    for binding in bindings.models:
        registry.get("tokenizers", binding.name)
        if binding.name not in lock.get("assets", {}):
            raise ValueError(f"unknown tokenizer asset: {binding.name}")

    def load(binding: Binding) -> TokenizerBackend:
        factory = registry.get("tokenizers", binding.name)
        backend = factory(root, lock)
        if binding.reference_key != binding.name:
            configure = getattr(backend, "configure_reference", None)
            if configure is None:
                raise AssetError("custom_reference_not_supported_by_backend")
            configure(binding)
        return backend

    return Recalculator(bindings, load, lock)
