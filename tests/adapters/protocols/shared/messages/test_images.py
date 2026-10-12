import base64
import hashlib
import io

import pytest
from PIL import Image

from modulo.adapters.protocols.shared.messages.chat import message
from modulo.adapters.protocols.shared.messages.images import image_part
from modulo.shared.serialization import loads


def embedded_image(format="PNG", *, size=(64, 48), orientation=None):
    image = Image.new("RGB", size, color="white")
    buffer = io.BytesIO()
    options = {}
    if orientation is not None:
        exif = image.getexif()
        exif[274] = orientation
        options["exif"] = exif.tobytes()
    image.save(buffer, format, **options)
    data = buffer.getvalue()
    return base64.b64encode(data).decode(), hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
@pytest.mark.parametrize("style", ["openai", "anthropic", "langchain"])
def test_embedded_formats_preserve_positions_and_dimension_evidence(format, style):
    data, sha = embedded_image(format)
    block = {
        "openai": {"type": "image_url", "image_url": {"url": f"data:image/{format};base64,{data}"}},
        "anthropic": {"type": "image", "source": {"type": "base64", "data": data}},
        "langchain": {"type": "image", "source_type": "base64", "base64": data},
    }[style]
    content = [{"type": "text", "text": " before "}, block, {"type": "text", "text": " after "}]
    parsed = message({"role": "user", "content": content}, path="request.messages.0.")
    assert parsed.content == " before  after "
    assert [p.type for p in parsed.content_parts] == ["text", "image", "text"]
    image = parsed.content_parts[1]
    assert (image.width, image.height, image.sha256) == (64, 48, sha)
    assert image.dimensions_source.startswith("request.messages.0.content.1.")
    assert image.dimensions_source.endswith(":image_header")
    assert not image.gaps and not parsed.gaps
    assert loads(parsed.structured_content_json)["content"] == content


def test_never_reads_remote_or_local_image_paths(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("Image path must not be opened")

    monkeypatch.setattr(Image, "open", fail)
    for path in ["https://private.invalid/image", "/etc/passwd", "file:///etc/passwd"]:
        part = image_part({"image_url": {"url": path, "width": 64, "height": 48}}, "content.0")
        assert part["width"] == 64 and part["height"] == 48
        assert not part["gaps"] and "sha256" not in part


@pytest.mark.parametrize(
    "block,source",
    [
        (
            {"type": "image_url", "image_url": {"url": "https://example.test/image"}},
            "image_url.url",
        ),
        ({"type": "input_image", "image_url": "https://example.test/image"}, "image_url"),
        (
            {"type": "image", "source": {"type": "url", "url": "https://example.test/image"}},
            "source.url",
        ),
        ({"type": "image", "source_type": "url", "url": "https://example.test/image"}, "url"),
    ],
)
def test_image_url_source_is_preserved_without_fetching(block, source):
    part = image_part(block, "messages.0.content.1")
    assert part["url"] == "https://example.test/image"
    assert part["url_source"] == "messages.0.content.1." + source
    assert "width" not in part and not part["gaps"]


def test_conflicting_urls_are_not_silently_chosen():
    part = image_part(
        {
            "url": "https://example.test/one",
            "image_url": {"url": "https://example.test/two"},
        },
        "content.0",
    )
    assert "conflicting_image_urls" in part["gaps"]


@pytest.mark.parametrize(
    "block,reason",
    [
        ({"data": "not-an-image"}, "invalid_embedded_image"),
        ({"data": "ab...[TRUNCATED]...cd"}, "truncated_image_data"),
        ({"width": 0, "height": 48}, "invalid_image_width"),
        ({"width": True, "height": 48}, "invalid_image_width"),
        ({"width": 64}, "incomplete_image_dimensions"),
        ({"width": 64, "image_url": {"width": 48}}, "conflicting_image_width"),
        (
            {"width": 64, "height": 48, "crop": [0, 0, 32, 32]},
            "unsupported_image_preprocessing:crop",
        ),
    ],
)
def test_incomplete_and_conflicting_evidence_has_specific_diagnostics(block, reason):
    assert reason in image_part(block, "content.0")["gaps"]


def test_embedded_dimensions_conflicts_and_exif_are_retained():
    data, sha = embedded_image("JPEG", orientation=6)
    part = image_part({"data": data, "width": 48, "height": 64}, "image")
    assert part["gaps"] == ["conflicting_image_dimensions"]
    assert part["orientation"] == 6 and part["sha256"] == sha
    assert (part["width"], part["height"]) == (64, 48)


def test_dify_separate_attachments_preserve_text_and_explain_lost_information():
    files = [{"type": "image", "data": "data:image...[TRUNCATED]...AA", "detail": "high"}]
    parsed = message({"role": "user", "text": "已保存的文字", "files": files}, content_key="text")
    assert parsed.content == "已保存的文字" and parsed.content_parts == []
    assert parsed.gaps == ["attachment_content_order_missing", "truncated_image_data"]
    assert loads(parsed.structured_content_json)["files"] == files
