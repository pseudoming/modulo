"""Shared Moonshot asset loading and counting, without importing upstream Python."""

import base64
import math
from pathlib import Path
from typing import Any

from modulo.core.domain.bindings import Binding
from modulo.core.domain.data_models import Call
from modulo.core.domain.interfaces import TokenizerBackend, UnsupportedInput
from modulo.shared.serialization import bytes_digest, digest, loads

from ..shared.assets import verified_asset
from ..shared.images import prepare_images
from ..shared.output import count_output
from ..shared.rendering import RenderProfile, template_json


class MoonshotTokenizer(RenderProfile, TokenizerBackend):
    family = "moonshot"
    version = "kimi.reasoning-reference"
    parse_tool_arguments = False
    named_messages = True
    encode_special = True

    def __init__(self, root: Path, lock: dict[str, Any]) -> None:
        directory, entry = verified_asset(
            root, self.name, lock, self.repo, ("tiktoken.model", "tokenizer_config.json")
        )
        import tiktoken

        config = loads((directory / "tokenizer_config.json").read_bytes())
        ranks = {}
        for line in (directory / "tiktoken.model").read_bytes().splitlines():
            token, rank = line.split()
            ranks[base64.b64decode(token, validate=True)] = int(rank)
        mapping = config["added_tokens_decoder"]
        special = {
            mapping.get(str(i), {}).get("content", f"<|reserved_token_{i}|>"): i
            for i in range(len(ranks), len(ranks) + 256)
        }
        self.tokenizer = tiktoken.Encoding(
            name="modulo-" + self.name,
            pat_str=_PATTERN,
            mergeable_ranks=ranks,
            special_tokens=special,
        )
        self.entry, self.asset_hash = entry, digest(entry)
        self.template_kwargs = entry["template_kwargs"]

    def _encode(self, text: str, special: bool | None = None) -> list[int]:
        if special is None:
            special = self.encode_special
        result = []
        for piece in _pieces(text):
            result.extend(
                self.tokenizer.encode(piece, allowed_special="all")
                if special
                else self.tokenizer.encode(piece, disallowed_special=())
            )
        return result

    def count(self, call: Call, binding: Binding) -> dict[str, Any]:
        options, sources = self.resolve_options(call)
        plans, images = prepare_images(self, call)
        messages = self.chat_messages(call)
        if images and any(
            "<|kimi_image_placeholder|>" in (message.content or "") for message in call.messages
        ):
            raise UnsupportedInput("image_placeholder_count_mismatch")
        for plan, image in zip(plans, images, strict=True):
            messages[image["message_index"]]["content"][image["part_index"]]["prompt"] = plan[
                "replacement"
            ]
            image["input_tokens"] = (
                len(self._encode(plan["replacement"], True)) + plan["visual_tokens"] - 1
            )
        if self.custom_template:
            parts = [(self._custom_render(messages, options, True, call.tools_json), True)]
            replacements = [plan["replacement"] for plan in plans]
            if any(
                parts[0][0].count(text) != replacements.count(text) for text in set(replacements)
            ):
                raise UnsupportedInput("image_placeholder_count_mismatch")
        else:
            parts = self.render_input(call, options, messages)
        ids = [token for text, special in parts for token in self._encode(text, special)]
        rendered = "".join(text for text, _ in parts)
        return {
            # K3's model replaces one media_pad embedding with N visual positions;
            # its tokenizer itself still returns one ID for that placeholder.
            "input_tokens": len(ids) + sum(plan["visual_tokens"] - 1 for plan in plans),
            "input_ids_scope": "tokenizer_ids_before_image_expansion"
            if images
            else "reference_input_ids",
            "input_images": images,
            "image_input_tokens": sum(image["input_tokens"] for image in images)
            if images
            else None,
            "input_ids_sha256": digest(ids),
            "rendered_input_sha256": bytes_digest(rendered.encode()),
            **self.option_evidence(options, sources),
            "asset_revision": self.entry["revision"],
            "asset_hash": self.asset_hash,
            "template_hash": self.template_hash,
        }

    def image_plan(self, width, height, parameters, grid):
        if grid is not None:
            raise UnsupportedInput("unsupported_image_grid_thw")
        patch = parameters["patch_size"]
        merge = parameters["merge_kernel_size"]
        side = parameters["patch_limit_on_one_side"]
        scale = min(
            1.0,
            math.sqrt(
                parameters["in_patch_limit"]
                / (max(1.0, width // patch) * max(1.0, height // patch))
            ),
            side * patch / width,
            side * patch / height,
        )
        new_width, new_height = (
            min(max(1, int(size * scale)), side * patch) for size in (width, height)
        )
        factor = merge * patch
        columns, rows = (math.ceil(size / factor) for size in (new_width, new_height))
        tokens = parameters["fixed_output_tokens"]
        if tokens is None:
            if max(rows, columns) * merge > side:
                raise UnsupportedInput("image_grid_exceeds_processor_limit")
            tokens = rows * columns
        return {
            "visual_tokens": tokens,
            "replacement": self.image_prompt(width, height),
            "layout": {"width": new_width, "height": new_height, "rows": rows, "columns": columns},
        }

    def thinking_enabled(self, options):
        return options.get("thinking", True) and super().thinking_enabled(options)

    def image_prompt(self, width, height):
        return "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"

    def count_output(self, call: Call) -> dict[str, Any]:
        return count_output(self, call, self._encode)

    def render_output(self, response, options):
        responses = response if isinstance(response, list) else [response]
        scaffold = [{"role": "user", "content": ""}]
        render = self._custom_render if self.custom_template else self.render_messages
        prefix = render(scaffold, options, True)
        # The current output retains its reasoning, even if the template drops
        # reasoning from historical turns. Explicit custom templates own this rule.
        full_options = options if self.custom_template else {**options, "preserve_thinking": True}
        completed = render([*scaffold, *responses], full_options, False)
        if not completed.startswith(prefix) or not completed.endswith(self.end_token):
            raise UnsupportedInput("custom_output_boundary_mismatch")
        return completed[len(prefix) : -len(self.end_token)], self.end_token

    @staticmethod
    def literal_output(text):
        return text

    def _custom_render(self, messages, options, generation, tools_json=None):
        return self.custom_template.render(
            messages=messages,
            tools=template_json(tools_json) if tools_json else None,
            **{
                **options,
                "eos_token": self.end_token,
                "bos_token": "",
                "add_generation_prompt": generation and options.get("add_generation_prompt", True),
            },
        )

    def encode_output(self, parts):
        if isinstance(parts, str):
            return self._encode(parts, True)
        return [token for text, special in parts for token in self._encode(text, special)]

    @staticmethod
    def output_text(parts):
        return parts if isinstance(parts, str) else "".join(text for text, _ in parts)


_PATTERN = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


def _pieces(text: str):
    # Match upstream's bounded encoder calls; never truncate the request.
    for offset in range(0, len(text), 400_000):
        chunk = text[offset : offset + 400_000]
        start, run = 0, 0
        previous_space = None
        for index, char in enumerate(chunk):
            space = char.isspace()
            run = run + 1 if space == previous_space else 1
            previous_space = space
            if run > 25_000:
                yield chunk[start:index]
                start, run = index, 1
        yield chunk[start:]
