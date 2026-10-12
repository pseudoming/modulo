"""Kimi K2.x templates and TypeScript tool declarations from the pinned reference."""

from pathlib import Path
from typing import Any

from modulo.core.domain.interfaces import AssetError, UnsupportedInput
from modulo.shared.serialization import bytes_digest, ordered_json

from ..shared.rendering import template_json
from .moonshot import MoonshotTokenizer


class KimiK2XTokenizer(MoonshotTokenizer):
    loader = "tiktoken_kimi_k2"
    end_token = "<|im_end|>"
    option_aliases = {"enable_thinking": "thinking"}
    thinking_parameter = "thinking"
    thinking_values = {"enabled": True, "disabled": False}
    option_rules = {"thinking": bool, "preserve_thinking": bool}

    def __init__(self, root: Path, lock: dict[str, Any]) -> None:
        super().__init__(root, lock)
        template_file = self.entry.get("template_file")
        if template_file not in self.entry["files"]:
            raise AssetError("template_not_locked")
        self.template_source = (root / self.name / template_file).read_text(encoding="utf-8")
        self.template_hash = bytes_digest(self.template_source.encode())
        from transformers.utils.chat_template_utils import _compile_jinja_template

        env = _compile_jinja_template(self.template_source).environment.overlay()
        env.filters = {**env.filters, "tojson": ordered_json, "fromjson": template_json}
        self.template = env.from_string(self.template_source)

    def render_input(self, call, options, messages):
        return [(self.render_messages(messages, options, True, call.tools_json), True)]

    def render_messages(self, messages, options, generation, tools_json=None):
        if any(not tool.get("id") for m in messages for tool in m.get("tool_calls", [])):
            raise UnsupportedInput("missing_tool_call_id")
        tools = sorted_tools(template_json(tools_json)) if tools_json else None
        declaration = None
        if tools:
            try:
                declaration = tool_declaration(tools)
            except (ValueError, KeyError, TypeError):
                # The pinned tokenizer falls back to sorted JSON for schemas its
                # TypeScript converter rejects. It does not drop the tools.
                declaration = None
        return self.template.render(
            messages=messages,
            tools=tools,
            tools_ts_str=declaration,
            **{
                **options,
                "add_generation_prompt": generation and options.get("add_generation_prompt", True),
            },
        )


class KimiK26Tokenizer(KimiK2XTokenizer):
    name = "kimi-k2.6"
    repo = "moonshotai/Kimi-K2.6"


class KimiK27CodeTokenizer(KimiK2XTokenizer):
    name = "kimi-k2.7-code"
    repo = "moonshotai/Kimi-K2.7-Code"
    option_rules = {"thinking": (True,), "preserve_thinking": (True,)}


def sorted_tools(value):
    if isinstance(value, dict):
        return {key: sorted_tools(value[key]) for key in sorted(value)}
    return [sorted_tools(v) for v in value] if isinstance(value, list) else value


def _description(text, indent=""):
    return "\n".join(indent + "// " + line if line else "" for line in (text or "").split("\n"))


def _docs(schema, indent):
    if not isinstance(schema, dict):
        return ""
    lines = []
    if schema.get("description"):
        lines.append(_description(schema["description"], indent))
    kind = schema.get("type")
    keys = ()
    if not any(key in schema for key in ("$ref", "enum", "anyOf")):
        if kind == "string":
            keys = ("maxLength", "minLength", "pattern")
        elif kind in ("integer", "number"):
            keys = ("maximum", "minimum")
        elif kind == "array":
            keys = ("maxItems", "minItems")
    limits = ", ".join(f"{key}: {schema[key]}" for key in keys if key in schema)
    if limits:
        lines.append(indent + "// " + limits)
    return "".join(line + "\n" for line in lines)


class _Schema:
    def __init__(self):
        self.definitions = {}
        self.self_ref = False

    def prepare(self, schema):
        if not isinstance(schema, dict):
            return
        if "$ref" in schema:
            self.render(schema)
        elif "anyOf" in schema:
            for item in schema["anyOf"]:
                self.prepare(item)
        elif "enum" not in schema:
            if schema.get("type") == "object":
                self.definitions.update(schema.get("$defs", {}))
                self.prepare(schema.get("additionalProperties"))
                for item in schema.get("properties", {}).values():
                    self.prepare(item)
            elif schema.get("type") == "array":
                self.prepare(schema.get("items"))

    def field(self, name, schema, optional, indent):
        docs = _docs(schema, indent)
        default = schema.get("default") if isinstance(schema, dict) else None
        if default is not None:
            value = (
                repr(default) if isinstance(default, (int, float, bool)) else ordered_json(default)
            )
            docs += f"{indent}// Default: {value}\n"
        return docs + f"{indent}{name}{'?' if optional else ''}: " + self.render(schema, indent)

    def render(self, schema, indent=""):
        if isinstance(schema, bool):
            return "any" if schema else "null"
        if "$ref" in schema:
            ref = schema["$ref"]
            if ref == "#":
                self.self_ref = True
                return "parameters"
            if not ref.startswith("#/$defs/") or ref.rsplit("/", 1)[-1] not in self.definitions:
                raise ValueError("unsupported_schema_reference")
            return ref.rsplit("/", 1)[-1]
        if "anyOf" in schema:
            return " | ".join(self.render(part, indent) for part in schema["anyOf"])
        if "enum" in schema:
            kind = schema.get("type")
            if isinstance(kind, list):
                if len(kind) > 2 or (len(kind) == 2 and "null" not in kind):
                    raise ValueError("unsupported_enum_type")
                kind = next((v for v in kind if v != "null"), "null")
            checks = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
            for item in schema["enum"]:
                if item is not None and kind in checks and not isinstance(item, checks[kind]):
                    raise ValueError("invalid_enum_value")
            return " | ".join(
                '"' + v + '"' if isinstance(v, str) else str(v) for v in schema["enum"]
            )
        kind = schema.get("type")
        if isinstance(kind, list):
            mapping = {
                "string": "string",
                "number": "number",
                "integer": "number",
                "boolean": "boolean",
                "null": "null",
                "object": "{}",
                "array": "Array<any>",
            }
            return " | ".join(mapping[v] for v in kind)
        if kind == "object":
            self.definitions.update(schema.get("$defs", {}))
            required = schema.get("required", [])
            properties = schema.get("properties", {})
            fields = [
                self.field(key, properties[key], key not in required, indent + "  ")
                for key in sorted(properties, key=lambda k: (k not in required, k))
            ]
            extra = schema.get("additionalProperties")
            if extra is not None:
                if isinstance(extra, bool):
                    value = "any" if extra else "never"
                elif isinstance(extra, dict):
                    value = self.render(extra, indent + "  ")
                else:
                    raise ValueError("invalid_additional_properties")
                fields.append(indent + "  [k: string]: " + value)
            return "{\n" + ",\n".join(fields) + "\n" + indent + "}" if fields else "{}"
        if kind == "array":
            item = schema.get("items") or {}
            doc = _docs(item, indent + "  ")
            if doc:
                return (
                    "Array<\n"
                    + doc
                    + indent
                    + "  "
                    + self.render(item, indent + "  ")
                    + "\n"
                    + indent
                    + ">"
                )
            return "Array<" + self.render(item, indent) + ">"
        if kind:
            return "number" if kind == "integer" else kind
        if not schema:
            return "any"
        raise ValueError("invalid_json_schema")


def tool_declaration(tools):
    functions = []
    for tool in tools:
        fn = tool["function"]
        schema = _Schema()
        parameters = fn.get("parameters") or {}
        root = {
            k: v
            for k, v in parameters.items()
            if k in {"properties", "required", "additionalProperties", "$defs"}
        }
        root["type"] = "object"
        schema.prepare(root)
        parsed = schema.render(root)
        interfaces = []
        if schema.self_ref:
            fields = [
                schema.field(k, v, k not in parameters.get("required", []), "  ")
                for k, v in parameters.get("properties", {}).items()
            ]
            interfaces.append(
                "interface parameters {"
                + ("\n" + ",\n".join(fields) + "\n" if fields else "")
                + "}"
            )
        for name, value in list(schema.definitions.items()):
            schema.prepare(value)
            description = _description(value.get("description", ""))
            interfaces.append(
                (description + "\n" if description else "")
                + f"interface {name} "
                + schema.render(value)
            )
        # Upstream discovers self-references while parsing the root, before defs.
        root = (
            "parameters"
            if interfaces and interfaces[0].startswith("interface parameters ")
            else parsed
        )
        description = _description(fn.get("description", ""))
        functions.append(
            "\n".join(
                filter(
                    None,
                    [
                        *interfaces,
                        description,
                        f"type {fn.get('name', 'function')} = (_: {root}) => any;",
                    ],
                )
            )
        )
    return (
        "# Tools\n\n## functions\nnamespace functions {\n" + "\n".join(functions) + "\n}\n"
        if functions
        else ""
    )
