"""Freeze image vectors from reviewed upstream code, without importing modulo.

--sources contains <asset>/<upstream basename>, each SHA-256 checked against
static/image-processing-sources.json. Extract only the dimension/layout functions:
no model weights, torch, URL loading or upstream package imports. Shared BPE kernels
are not independent segmentation algorithms. Run explicitly, never from the test suite.
"""

import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from build_family_reference import kimi_reference, reference_assets
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[3]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def upstream_nodes(source, names, namespace):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    # Defer annotations containing image/torch types. Only the reviewed pure
    # methods below are called; no imports or module initialization are run.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, "<reviewed image reference>", "exec"), namespace)
    return namespace


def reference_layout(family, files, tokenizer, asset):
    namespace = {"math": math, "ceil": math.ceil, "np": np, "product": itertools.product}
    if family == "deepseek":
        upstream_nodes(
            files["image_processor.py"],
            [
                "num_image_tokens",
                "llm_grid",
                "solve_resize_ratio",
                "safe_resize",
                "plan_image_grid",
            ],
            namespace,
        )

        def plan(w, h, p):
            rows, columns, _, _ = namespace["plan_image_grid"](w, h, SimpleNamespace(**p))
            count = namespace["num_image_tokens"](rows, columns)
            return rows * columns, "<｜deepseek_image｜>" * count

        return plan, "<｜deepseek_image｜>"
    if family == "moonshot":
        upstream_nodes(files["media_utils.py"], ["navit_resize_image"], namespace)

        def plan(w, h, p):
            result = namespace["navit_resize_image"](w, h, **p)
            # Only the K3 processor includes original dimensions in its prompt.
            prompt = "<|media_begin|>image" + (f" {w}x{h}" if asset == "kimi-k3" else "")
            prompt += "<|media_content|><|media_pad|><|media_end|>"
            return result["num_tokens"], prompt

        return plan, None
    if family == "glm":
        upstream_nodes(files["image_processing_glm5_next.py"], ["smart_resize"], namespace)

        def plan(w, h, p):
            h, w = namespace["smart_resize"](
                num_frames=p["temporal_patch_size"],
                height=h,
                width=w,
                temporal_factor=p["temporal_patch_size"],
                factor=p["patch_size"] * p["merge_size"] * p["patch_expand_factor"],
                min_pixels=p["min_image_tokens"],
                max_pixels=p["max_image_tokens"],
            )
            count = (h // p["patch_size"]) * (w // p["patch_size"]) // p["merge_size"] ** 2
            return count, "<|begin_of_image|>" + "<|image|>" * count + "<|end_of_image|>"

        return plan, "<|begin_of_image|><|image|><|end_of_image|>"
    if family in {"minimax", "mimo", "qwen"}:
        names = ["smart_resize"]
        if family == "minimax":
            names += ["round_by_factor", "ceil_by_factor", "floor_by_factor"]
            namespace["MAX_RATIO"] = 200
        source = files[
            "image_processor.py" if family == "minimax" else "image_processing_qwen2_vl.py"
        ]
        upstream_nodes(source, names, namespace)
        token = "]<]image[>[" if family == "minimax" else "<|image_pad|>"
        start = "]<]start of image[>[" if family == "minimax" else "<|vision_start|>"
        end = "]<]end of image[>[" if family == "minimax" else "<|vision_end|>"

        def plan(w, h, p):
            factor = p["patch_size"] * p["merge_size"]
            if p["do_resize"]:
                opts = {"min_pixels": p["min_pixels"]} if "min_pixels" in p else {}
                h, w = namespace["smart_resize"](
                    h, w, factor=factor, max_pixels=p["max_pixels"], **opts
                )
            count = (h // p["patch_size"]) * (w // p["patch_size"]) // p["merge_size"] ** 2
            return count, start + token * count + end

        return plan, token if family == "minimax" else start + token + end
    namespace["MAX_IMAGE_SIZE"] = 3024
    namespace["ProcessorMixin"] = object
    upstream_nodes(files["processing_step3.py"], ["ImagePatcher", "Step3VLProcessor"], namespace)
    processor = object.__new__(namespace["Step3VLProcessor"])
    processor.patcher = namespace["ImagePatcher"]()
    processor.tokenizer = SimpleNamespace(
        convert_tokens_to_ids=tokenizer.token_to_id, get_vocab=tokenizer.get_vocab
    )
    processor.image_token = "<im_patch>"
    processor.num_image_feature_size, processor.num_patch_feature_size = 169, 81
    processor.image_feature_placeholder = "<im_patch>" * 169
    processor.patch_feature_placeholder = "<im_patch>" * 81

    def plan(w, h, p):
        patches, newlines = processor.patcher.get_num_patches(w, h)
        columns = patches // (newlines + 1)
        newline_mask = [((i + 1) % columns == 0 and i + 1 < patches) for i in range(patches)]
        text, ids = processor._get_image_repl_features(1, patches, newline_mask)
        assert len(ids) == processor.get_num_image_tokens(w, h)
        assert tokenizer.encode(text, add_special_tokens=False).ids == ids
        return patches * 81 + 169, text

    return plan, "<im_patch>"


def fail(message):
    raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--assets", nargs="+", help="Only replace these assets' frozen vectors")
    args = parser.parse_args()
    lock = reference_assets()
    sources = json.loads((ROOT / "static/image-processing-sources.json").read_text())
    selected = args.assets or list(lock["assets"])
    assert set(selected) <= lock["assets"].keys()
    vectors, provenance = [], {}
    for asset, entry in lock["assets"].items():
        if "image_processing" not in entry or asset not in selected:
            continue
        family, profile = entry["family"], entry["image_processing"]
        directory = ROOT / "assets/tokenizers" / asset
        for name, expected in entry["files"].items():
            assert sha((directory / name).read_bytes()) == expected, name
        files = {}
        for source in sources[asset]:
            name = source["url"].rsplit("/", 1)[-1]
            data = (args.sources / asset / name).read_bytes()
            assert sha(data) == source["sha256"], source["url"]
            files[name] = data.decode()
        provenance[asset] = sources[asset]
        if family == "deepseek":
            config = json.loads(files["config.json"])
            assert profile["parameters"] == {k: config[k] for k in profile["parameters"]}
        elif family in {"moonshot", "mimo", "qwen", "glm"}:
            config = json.loads(
                files["processor_config.json" if family == "glm" else "preprocessor_config.json"]
            )
            config = config.get("media_proc_cfg", config.get("image_processor", config))
            if family == "qwen":
                config.update(
                    min_pixels=config["size"]["shortest_edge"],
                    max_pixels=config["size"]["longest_edge"],
                )
            for key, value in profile["parameters"].items():
                assert value == config.get(key, True if key == "do_resize" else None)
        elif family == "minimax":
            cls = next(
                node
                for node in ast.parse(files["image_processor.py"]).body
                if getattr(node, "name", None) == "MiniMaxM3VLImageProcessor"
            )
            constants = {
                t.id: ast.literal_eval(node.value)
                for node in cls.body
                if isinstance(node, ast.Assign)
                for t in node.targets
                if isinstance(t, ast.Name) and t.id in profile["parameters"]
            }
            assert constants == profile["parameters"]
        if family == "moonshot":
            renderer, encoder = kimi_reference(directory)
            tokenizer = None

            def encode(text, encoder=encoder):
                return encoder.encode(text, allowed_special="all")
        else:
            tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))

            def encode(text, tokenizer=tokenizer):
                return tokenizer.encode(text, add_special_tokens=False).ids

            config = json.loads((directory / "tokenizer_config.json").read_text())
            template = (
                (directory / entry["template_file"]).read_text()
                if entry.get("template_file")
                else config["chat_template"]
            )
            env = ImmutableSandboxedEnvironment(
                trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
            )
            env.globals["raise_exception"] = fail
            env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
            env.filters["fromjson"] = json.loads
            renderer = env.from_string(template)
            special = {
                k: v["content"] if isinstance(v, dict) else v
                for k, v in config.items()
                if k.endswith("_token") and isinstance(v, (str, dict))
            }
        plan, placeholder = reference_layout(family, files, tokenizer, asset)
        sizes = [
            (1, 1),
            (28, 28),
            (31, 200),
            (640, 480),
            (728, 728),
            (729, 729),
            (1000, 50),
            (50, 1000),
            (4096, 3072),
            (3025, 1513),
            (1200, 1000),
        ]
        cases = [(f"{w}x{h}", [(w, h)], {}) for w, h in sizes]
        cases.append(("two_images", [(640, 480), (50, 1000)], {}))
        changes = {
            "deepseek": {"vision_max_n_token": 128, "vision_min_pixels": 16384},
            "moonshot": {"in_patch_limit": 1024, "patch_limit_on_one_side": 64},
            "minimax": {"max_pixels": 200704},
            "mimo": {"max_pixels": 200704},
            "qwen": {"max_pixels": 200704},
            "glm": {"max_image_tokens": 1024},
            "stepfun": {},
        }[family]
        if changes:
            cases.append(("explicit_options", [(4096, 3072)], changes))
        for name, dimensions, overrides in cases:
            parameters = {**profile["parameters"], **overrides}
            spans = [plan(w, h, parameters) for w, h in dimensions]
            content = [{"type": "text", "text": "图片之前\n"}]
            for w, h in dimensions:
                content.extend(
                    [
                        {"type": "image", "width": w, "height": h},
                        {"type": "text", "text": " 图片之后 e\u0301🙂 "},
                    ]
                )
            messages = [{"role": "user", "content": content}]
            if family == "moonshot":
                segments = renderer.build_chat_segments(
                    messages,
                    image_prompts=[text for _, text in spans],
                    add_generation_prompt=True,
                    **entry["template_kwargs"],
                )
                rendered = "".join(s.text for s in segments)
                ids = []
                for s in segments:
                    ids.extend(
                        encoder.encode(s.text, allowed_special="all")
                        if s.allow_special
                        else encoder.encode(s.text, disallowed_special=())
                    )
                count = len(ids) + sum(n - 1 for n, _ in spans)
                image_counts = [len(encode(text)) + n - 1 for n, text in spans]
            else:
                rendered = renderer.render(
                    messages=messages,
                    tools=None,
                    add_generation_prompt=True,
                    **special,
                    **entry["template_kwargs"],
                )
                for _ in spans:
                    # Temporary sentinel avoids finding a placeholder inside the
                    # just-expanded span when the next image is replaced.
                    rendered = rendered.replace(placeholder, "\x00", 1)
                for _, text in spans:
                    rendered = rendered.replace("\x00", text, 1)
                ids = encode(rendered)
                count, image_counts = len(ids), [len(encode(text)) for _, text in spans]
            vectors.append(
                {
                    "asset": asset,
                    "backend": family,
                    "name": name,
                    "messages": messages,
                    "overrides": overrides,
                    "visual_tokens": [n for n, _ in spans],
                    "image_input_tokens": image_counts,
                    "input_tokens": count,
                    "input_ids_sha256": sha(json.dumps(ids, separators=(",", ":")).encode()),
                    "rendered_input_sha256": sha(rendered.encode()),
                }
            )
    path = ROOT / "tests/fixtures/tokenizers/image_reference_vectors.json"
    if args.assets:
        previous = json.loads(path.read_text())
        vectors = [v for v in previous["vectors"] if v["asset"] not in selected] + vectors
        provenance = {**previous["sources"], **provenance}
    path.write_text(
        json.dumps({"sources": provenance, "vectors": vectors}, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Wrote {len(vectors)} image reference vectors to {path}")


if __name__ == "__main__":
    main()
