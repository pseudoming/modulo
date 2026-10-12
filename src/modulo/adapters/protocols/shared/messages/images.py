"""Read image evidence already inside a log; never dereference URLs or paths."""

import base64
import binascii
import io
import warnings

from PIL import Image, UnidentifiedImageError

from modulo.shared.serialization import bytes_digest


def image_header(data):
    """Inspect supplied bytes only; shared with the optional URL downloader."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                return {
                    "width": image.width,
                    "height": image.height,
                    "sha256": bytes_digest(data),
                    "orientation": image.getexif().get(274, 1),
                    "gaps": ["animated_image_not_supported"]
                    if getattr(image, "n_frames", 1) != 1
                    else [],
                }
        except (
            ValueError,
            OSError,
            SyntaxError,
            UnidentifiedImageError,
            Image.DecompressionBombWarning,
            Image.DecompressionBombError,
        ) as exc:
            raise ValueError("invalid_image_data") from exc


def image_part(block, path):
    part = {"type": "image", "source": path, "options": {}, "gaps": []}
    roots = [(path, block)]
    for key in ("image_url", "image", "source"):
        if isinstance(block.get(key), dict):
            roots.append((f"{path}.{key}", block[key]))
    values, evidence, embedded = {}, {}, []
    for root_path, root in roots:
        for key in ("width", "height", "detail", "min_pixels", "max_pixels", "image_grid_thw"):
            if root.get(key) is not None:
                value = root[key]
                if key in values and (type(values[key]) is not type(value) or values[key] != value):
                    part["gaps"].append("conflicting_image_" + key)
                values[key] = value
                evidence[key] = f"{root_path}.{key}"
        for key in ("url", "image_url", "image", "data", "base64"):
            value = root.get(key)
            if not isinstance(value, str):
                continue
            if value.startswith("data:") or key in {"data", "base64"}:
                embedded.append((value, f"{root_path}.{key}"))
            elif value:
                if part.get("url") and part["url"] != value:
                    part["gaps"].append("conflicting_image_urls")
                part.update(url=value, url_source=f"{root_path}.{key}")
        for key in ("resized_width", "resized_height", "crop", "crops", "patches"):
            if key in root:
                part["gaps"].append("unsupported_image_preprocessing:" + key)
    for key in ("width", "height"):
        if key in values:
            if type(values[key]) is not int or values[key] <= 0:
                part["gaps"].append("invalid_image_" + key)
            else:
                part[key] = values[key]
    if part.get("width") and part.get("height"):
        part["dimensions_source"] = ",".join(evidence[k] for k in ("width", "height"))
    elif "width" in values or "height" in values:
        part["gaps"].append("incomplete_image_dimensions")
    for value, source in embedded:
        try:
            if "[TRUNCATED]" in value:
                part["gaps"].append("truncated_image_data")
                continue
            if value.startswith("data:"):
                header, separator, value = value.partition(",")
                if not separator or not header.endswith(";base64"):
                    raise ValueError("invalid data URL")
            # Bound the extra decoded copy; the original remains in raw.jsonl.
            if len(value) > 48 * 1024 * 1024:
                part["gaps"].append("embedded_image_too_large")
                continue
            data = base64.b64decode(value, validate=True)
            sha256 = bytes_digest(data)
            if part.get("sha256") and part["sha256"] != sha256:
                part["gaps"].append("conflicting_embedded_images")
                continue
            header = image_header(data)
            part["gaps"].extend(header.pop("gaps"))
            if any(part.get(k) not in (None, header[k]) for k in ("width", "height")):
                part["gaps"].append("conflicting_image_dimensions")
            part.update(header)
            part["dimensions_source"] = source + ":image_header"
        except (ValueError, binascii.Error):
            part["gaps"].append("invalid_embedded_image")
    part["options"] = {k: v for k, v in values.items() if k not in {"width", "height"}}
    part["option_sources"] = {k: evidence[k] for k in part["options"]}
    part["gaps"] = sorted(set(part["gaps"]))
    return part
