"""Shared offline Hugging Face rendering, image expansion and text encoding."""

import logging
from pathlib import Path
from typing import Any

from modulo.core.domain.bindings import Binding
from modulo.core.domain.data_models import Call
from modulo.core.domain.interfaces import AssetError, TokenizerBackend, UnsupportedInput
from modulo.shared.serialization import bytes_digest, digest, loads, ordered_json

from .assets import verified_asset
from .images import expand_images, prepare_images
from .output import count_output
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


def _fast_tokenizer():
    # This module never runs model inference. Preserve every other diagnostic and
    # leave the embedding application's logging level/environment untouched.
    logger = logging.getLogger("transformers")
    notice_filter = _TokenizersOnlyNoticeFilter()
    logger.addFilter(notice_filter)
    try:
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast
    finally:
        logger.removeFilter(notice_filter)


class HuggingFaceTextTokenizer(RenderProfile, TokenizerBackend):
    version = "hf.reasoning-chat.v3"
    family = ""
    repo = ""
    loader = "fast_json"
    output_separator = ""
    output_end = None
    empty_thinking_prefix = ""

    def __init__(self, root: Path, lock: dict[str, Any]) -> None:
        directory, entry = verified_asset(
            root, self.name, lock, self.repo, ("tokenizer.json", "tokenizer_config.json")
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
        if self.loader != "fast_json":
            raise AssetError("asset_loader_unsupported")
        fast = _fast_tokenizer()
        try:
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
            self.tokenizer = fast(tokenizer_file=str(directory / "tokenizer.json"), **options)
        except AssetError:
            raise
        except Exception as exc:
            raise AssetError(f"tokenizer_load_error:{type(exc).__name__}") from exc
        self.tokenizer.chat_template = template
        self.template = self.compile_template(template)
        self.template_source = template
        self.template_kwargs = entry.get("template_kwargs", {})
        self.entry, self.asset_hash = entry, digest(entry)
        self.template_hash = bytes_digest(template.encode())

    def compile_template(self, template):
        from transformers.utils.chat_template_utils import _compile_jinja_template

        environment = _compile_jinja_template(template).environment.overlay()
        environment.filters = environment.filters.copy()
        environment.filters.update(tojson=ordered_json, fromjson=template_json)
        return environment.from_string(template)

    def count(self, call: Call, binding: Binding) -> dict[str, Any]:
        options, sources = self.resolve_options(call)
        plans, images = prepare_images(self, call)
        template = self.custom_template or self.template
        rendered = template.render(
            messages=self.chat_messages(call),
            tools=template_json(call.tools_json) if call.tools_json is not None else None,
            **{
                **self.tokenizer.special_tokens_map,
                "add_generation_prompt": binding.add_generation_prompt,
                **options,
            },
        )
        if plans:
            rendered = expand_images(
                rendered, self.image_placeholder, [plan["replacement"] for plan in plans]
            )
            for plan, image in zip(plans, images, strict=True):
                image["input_tokens"] = len(
                    self.tokenizer.encode(plan["replacement"], add_special_tokens=False)
                )
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        return {
            "input_images": images,
            "image_input_tokens": sum(image["input_tokens"] for image in images)
            if images
            else None,
            "input_tokens": len(ids),
            "input_ids_sha256": digest(ids),
            "rendered_input_sha256": bytes_digest(rendered.encode()),
            **self.option_evidence(options, sources),
            "asset_revision": self.entry["revision"],
            "asset_hash": self.asset_hash,
            "template_hash": self.template_hash,
        }

    def count_output(self, call: Call) -> dict[str, Any]:
        return count_output(
            self, call, lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        )

    def assistant_output(self, response, options):
        # The empty user turn is only a template scaffold, never an observed input.
        # It isolates the assistant format independently of missing request logs.
        messages = [{"role": "user", "content": ""}]
        kwargs = {**self.tokenizer.special_tokens_map, **options}
        template = self.custom_template or self.template
        prefix = template.render(
            messages=messages,
            tools=None,
            **{**kwargs, "add_generation_prompt": options.get("add_generation_prompt", True)},
        )
        responses = response if isinstance(response, list) else [response]
        completed = template.render(
            messages=[*messages, *responses],
            tools=None,
            **{**kwargs, "add_generation_prompt": False},
        )
        # Some history templates omit an empty channel's opening tag even when
        # generation has already prefixed it. Only the declared empty case applies.
        if (
            self.empty_thinking_prefix
            and responses[0].get("reasoning_content") == ""
            and prefix.endswith(self.empty_thinking_prefix)
            and not completed.startswith(prefix)
        ):
            prefix = prefix[: -len(self.empty_thinking_prefix)]
        if not completed.startswith(prefix):
            raise UnsupportedInput("output_generation_prefix_mismatch")
        return completed[len(prefix) :]

    def render_output(self, response, options):
        rendered = self.assistant_output(response, options)
        ending = (
            str(self.tokenizer.eos_token)
            if self.custom_template
            else self.output_end or str(self.tokenizer.eos_token)
        )
        suffix = ending + ("" if self.custom_template else self.output_separator)
        if not rendered.endswith(suffix):
            raise UnsupportedInput("output_end_marker_mismatch")
        return rendered[: -len(suffix)], ending

    def encode_output(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def literal_output(text):
        return text

    @staticmethod
    def output_text(text):
        return text
