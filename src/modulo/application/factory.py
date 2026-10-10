"""Create the configured adapter registry and local reference calculator."""

from pathlib import Path

from modulo.core.domain.bindings import Binding, Bindings
from modulo.core.domain.interfaces import TokenizerBackend
from modulo.core.processing.recalculation import Recalculator

from .registry import Registry, builtins
from .tokenizer_assets import get_backend_name


def build_registry(plugin_modules: list[str]) -> Registry:
    registry = builtins()
    for name in plugin_modules:
        registry.load(name)
    return registry


def make_calculator(registry: Registry, root: Path, bindings: Bindings, lock: dict) -> Recalculator:
    for binding in bindings.models:
        registry.get("tokenizers", get_backend_name(binding.asset, lock))

    def load(binding: Binding) -> TokenizerBackend:
        factory = registry.get("tokenizers", get_backend_name(binding.asset, lock))
        return factory(root, binding.asset, lock)

    return Recalculator(bindings, load, lock)
