"""Shared offline Hugging Face renderer and encoder for pinned text references."""

import logging
from pathlib import Path
from typing import Any

from modulo.core.domain.bindings import Binding
from modulo.core.domain.data_models import Call
from modulo.core.domain.interfaces import AssetError, TokenizerBackend
from modulo.shared.serialization import bytes_digest, digest, loads, ordered_json

from .assets import verified_asset
from .rendering import RenderProfile, template_json


class _TokenizersOnlyNoticeFilter(logging.Filter):
    """Drop only HF's expected no-inference-framework notice during our import."""

    def filter(self, record: logging.LogRecord) -> bool:
        expected = (
            "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
            "Models won't be available and only tokenizers, configuration "
            "and file/data utilities can be used."
        )
        return not (record.name == "transformers" and record.getMessage() == expected)


def _auto_tokenizer():
    # This module never runs model inference. Preserve every other diagnostic and
    # leave the embedding application's logging level/environment untouched.
    logger = logging.getLogger("transformers")
    notice_filter = _TokenizersOnlyNoticeFilter()
    logger.addFilter(notice_filter)
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer
    finally:
        logger.removeFilter(notice_filter)


class HuggingFaceTextTokenizer(RenderProfile, TokenizerBackend):
    version = "hf.reasoning-chat.v3"
    family = ""
    allowed_repo = ""
    allowed_repos: tuple[str, ...] = ()

    def __init__(self, root: Path, asset: str, lock: dict[str, Any]) -> None:
        directory, entry = verified_asset(
            root, asset, lock, self.family, self.allowed_repos or (self.allowed_repo,)
        )
        config = loads((directory / "tokenizer_config.json").read_bytes())
        template_file = entry.get("template_file")
        if template_file is not None and template_file not in entry["files"]:
            raise AssetError("template_not_locked")
        template = (
            (directory / template_file).read_text(encoding="utf-8")
            if template_file is not None
            else config.get("chat_template")
        )
        if not isinstance(template, str):
            raise AssetError("template_missing")
        if "strftime_now" in template:
            raise AssetError("dynamic_template_not_supported")
        # No downloads, model weights, custom remote code, or guessed model aliases.
        auto = _auto_tokenizer()
        try:
            if entry.get("loader") == "fast_json":
                from transformers import PreTrainedTokenizerFast

                # New upstream class names can target Transformers 5. The pinned
                # tokenizer.json fully specifies the reviewed native encoder; do
                # not import upstream Python or resolve an AutoTokenizer class.
                options = {
                    k: v
                    for k, v in config.items()
                    if k not in {"tokenizer_class", "auto_map", "tokenizer_file", "chat_template"}
                }
                from tokenizers import AddedToken

                def token(value):
                    if isinstance(value, dict) and "content" in value:
                        return AddedToken(**{k: v for k, v in value.items() if k != "__type"})
                    return value

                if "added_tokens_decoder" in options:
                    options["added_tokens_decoder"] = {
                        int(k): token(v) for k, v in options["added_tokens_decoder"].items()
                    }
                for key in (
                    "bos_token",
                    "eos_token",
                    "pad_token",
                    "unk_token",
                    "sep_token",
                    "cls_token",
                    "mask_token",
                ):
                    if key in options:
                        options[key] = token(options[key])
                if "additional_special_tokens" in options:
                    options["additional_special_tokens"] = [
                        token(v) for v in options["additional_special_tokens"]
                    ]
                # Transformers 5 renamed the additional-token list. In 4.x
                # extra_special_tokens instead means a named attribute mapping.
                if isinstance(options.get("extra_special_tokens"), list):
                    extra = options.pop("extra_special_tokens")
                    options["additional_special_tokens"] = [
                        *options.get("additional_special_tokens", []),
                        *[token(v) for v in extra],
                    ]
                self.tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(directory / "tokenizer.json"), **options
                )
            elif entry.get("loader") is None:
                self.tokenizer = auto.from_pretrained(
                    str(directory), local_files_only=True, trust_remote_code=False, use_fast=True
                )
            else:
                raise AssetError("asset_loader_unsupported")
        except AssetError:
            raise
        except Exception as exc:
            raise AssetError(f"tokenizer_load_error:{type(exc).__name__}") from exc
        self.tokenizer.chat_template = template
        from transformers.utils.chat_template_utils import _compile_jinja_template

        environment = _compile_jinja_template(template).environment.overlay()
        environment.filters = environment.filters.copy()
        environment.filters.update(tojson=ordered_json, fromjson=template_json)
        self.template = environment.from_string(template)
        self.template_kwargs = entry.get("template_kwargs", {})
        self.entry, self.asset_hash = entry, digest(entry)
        self.template_hash = bytes_digest(template.encode())

    def count(self, call: Call, binding: Binding) -> dict[str, Any]:
        options, sources = self.resolve_options(call)
        rendered = self.template.render(
            messages=self.chat_messages(call),
            tools=template_json(call.tools_json) if call.tools_json is not None else None,
            add_generation_prompt=binding.add_generation_prompt,
            **{**self.tokenizer.special_tokens_map, **options},
        )
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        return {
            "input_tokens": len(ids),
            "input_ids_sha256": digest(ids),
            "rendered_input_sha256": bytes_digest(rendered.encode()),
            **self.count_output(call),
            **self.option_evidence(options, sources),
            "asset_revision": self.entry["revision"],
            "asset_hash": self.asset_hash,
            "template_hash": self.template_hash,
        }

    def count_output(self, call: Call) -> dict[str, Any]:
        return self.output_counts(
            call, lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        )
