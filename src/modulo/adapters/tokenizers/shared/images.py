"""Image layout evidence and expansion shared by the fixed visual references."""

import math

from modulo.core.domain.data_models import ImageMeasurement
from modulo.core.domain.interfaces import UnsupportedInput

IMAGE_OPTIONS = {"mm_processor_kwargs", "images_kwargs"}


def smart_grid(width, height, *, patch_size, merge_size, min_pixels, max_pixels, minimax=False):
    """Pinned Qwen2-VL and MiniMax differ at the minimum-side boundaries."""
    if max(width, height) / min(width, height) > 200:
        raise UnsupportedInput("image_aspect_ratio_exceeds_processor_limit")
    factor = patch_size * merge_size
    h, w = (round(side / factor) * factor for side in (height, width))
    if minimax:
        h, w = max(factor, h), max(factor, w)
    if h * w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        h, w = (math.floor(side / scale / factor) * factor for side in (height, width))
        if not minimax:
            h, w = max(factor, h), max(factor, w)
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h, w = (math.ceil(side * scale / factor) * factor for side in (height, width))
    if min(h, w) <= 0:
        raise UnsupportedInput("invalid_image_processor_grid")
    return [1, h // patch_size, w // patch_size]


def grid_tokens(grid, merge_size):
    if (
        not isinstance(grid, list)
        or len(grid) != 3
        or any(type(side) is not int or side <= 0 for side in grid)
        or grid[0] != 1
        or any(side % merge_size for side in grid[1:])
    ):
        raise UnsupportedInput("invalid_image_grid_thw")
    return grid[1] * grid[2] // merge_size**2


def grid_plan(width, height, parameters, observed, *, minimax=False):
    computed = None
    if width is not None and height is not None:
        if parameters["do_resize"]:
            computed = smart_grid(
                width,
                height,
                patch_size=parameters["patch_size"],
                merge_size=parameters["merge_size"],
                min_pixels=3136 if minimax else parameters["min_pixels"],
                max_pixels=parameters["max_pixels"],
                minimax=minimax,
            )
        else:
            factor = parameters["patch_size"] * parameters["merge_size"]
            if width % factor or height % factor:
                raise UnsupportedInput("image_size_not_divisible_by_processor_grid")
            computed = [1, height // parameters["patch_size"], width // parameters["patch_size"]]
    grid = computed if observed is None else observed
    tokens = grid_tokens(grid, parameters["merge_size"])
    if observed is not None and computed is not None and observed != computed:
        raise UnsupportedInput("conflicting_image_grid_thw")
    return grid, tokens


def image_slots(token, count):
    if type(count) is not int or not 0 < count <= 1_000_000:
        raise UnsupportedInput("image_reference_position_limit")
    return token * count


def image_parameters(backend, call, image):
    profile = backend.entry.get("image_processing")
    if not profile:
        raise UnsupportedInput("asset_has_no_image_reference")
    parameters = dict(profile["parameters"])
    sources = {key: ["asset_profile"] for key in parameters}
    explicit = {}
    roots = []
    for name in sorted(IMAGE_OPTIONS):
        if name in call.render_options:
            root = call.render_options[name]
            if not isinstance(root, dict):
                raise UnsupportedInput("invalid_image_processor_options")
            paths = call.render_option_sources.get(name, ["call." + name])
            roots.append((root, {key: [f"{path}.{key}" for path in paths] for key in root}))
    roots.append(
        (
            image.options,
            {key: [image.option_sources.get(key, image.source)] for key in image.options},
        )
    )
    for root, paths in roots:
        for key, value in root.items():
            if root is image.options and key in {"detail", "image_grid_thw"}:
                continue
            if key not in parameters:
                raise UnsupportedInput("unsupported_image_option:" + key)
            if key in explicit and (
                type(explicit[key]) is not type(value) or explicit[key] != value
            ):
                raise UnsupportedInput("conflicting_image_option:" + key)
            if key == "do_resize":
                valid = type(value) is bool
            elif key in {"vision_max_wh_ratio", "fixed_output_tokens"} and value is None:
                valid = True
            else:
                valid = type(value) is int and value > 0
            if not valid:
                raise UnsupportedInput("invalid_image_option:" + key)
            explicit[key] = value
            parameters[key] = value
            sources[key] = paths[key]
    if parameters.get("min_pixels", 1) > parameters.get("max_pixels", math.inf):
        raise UnsupportedInput("invalid_image_pixel_limits")
    if parameters.get("min_image_tokens", 1) > parameters.get("max_image_tokens", math.inf):
        raise UnsupportedInput("invalid_image_token_limits")
    if parameters.get("vision_max_n_token", 4) < 4:
        raise UnsupportedInput("invalid_image_option:vision_max_n_token")
    return parameters, sources


def prepare_images(backend, call):
    plans, measurements, reasons = [], [], []
    for message_index, message in enumerate(call.messages or []):
        for part_index, part in enumerate(message.content_parts):
            if part.type != "image":
                continue
            item = ImageMeasurement(
                message_index=message_index,
                part_index=part_index,
                source=part.source,
                width=part.width,
                height=part.height,
                dimensions_source=part.dimensions_source,
                sha256=part.sha256,
                download=part.download,
            ).model_dump()
            try:
                if part.gaps:
                    raise UnsupportedInput(",".join(part.gaps))
                parameters, sources = image_parameters(backend, call, part)
                item.update(parameters=parameters, parameter_sources=sources)
                if part.options.get("detail") not in (None, "auto"):
                    raise UnsupportedInput("unsupported_image_detail")
                width, height = part.width, part.height
                if backend.entry["image_processing"].get("exif_transpose") and part.orientation in {
                    5,
                    6,
                    7,
                    8,
                }:
                    width, height = height, width
                    item["layout"]["exif_transpose"] = True
                grid = part.options.get("image_grid_thw")
                if grid is not None:
                    item["layout"]["image_grid_source"] = part.option_sources.get(
                        "image_grid_thw", part.source
                    )
                if (width is None or height is None) and not (
                    grid is not None and backend.image_grid_supported
                ):
                    raise UnsupportedInput("missing_image_dimensions")
                plan = backend.image_plan(width, height, parameters, grid)
                if not 0 < plan["visual_tokens"] <= 1_000_000:
                    raise UnsupportedInput("image_reference_position_limit")
                item["layout"].update(plan.get("layout", {}))
                item.update(execution="completed", visual_tokens=plan["visual_tokens"])
                plans.append(plan)
            except UnsupportedInput as exc:
                item["reason"] = str(exc)
                reasons.append(f"image[{message_index}:{part_index}]:{exc}")
            measurements.append(item)
    if reasons:
        raise UnsupportedInput(";".join(reasons), evidence={"input_images": measurements})
    if not measurements and any(call.render_options.get(key) for key in IMAGE_OPTIONS):
        raise UnsupportedInput("image_options_without_images")
    return plans, measurements


def expand_images(rendered, placeholder, replacements):
    parts = rendered.split(placeholder)
    if len(parts) != len(replacements) + 1:
        raise UnsupportedInput("image_placeholder_count_mismatch")
    return parts[0] + "".join(
        image + text for image, text in zip(replacements, parts[1:], strict=True)
    )
