import base64
import copy
import io
import json

import pytest
from PIL import Image

from modulo.adapters.protocols.shared.messages.chat import message
from modulo.application.registry import builtins
from modulo.application.tokenizer_assets import load_tokenizer_assets
from modulo.core.domain.bindings import Binding
from modulo.core.domain.data_models import Call
from modulo.core.domain.interfaces import UnsupportedInput
from tests.support.paths import ROOT

VECTORS = json.loads((ROOT / "tests/fixtures/tokenizers/image_reference_vectors.json").read_text())[
    "vectors"
]
ASSETS = list(dict.fromkeys(v["asset"] for v in VECTORS))


@pytest.fixture(scope="module")
def backends():
    return {}


def backend_for(asset, assets, backends):
    if asset not in backends:
        lock = load_tokenizer_assets()
        backends[asset] = builtins().get("tokenizers", asset)(assets, lock)
    return backends[asset]


def call_for(vector):
    return Call(
        call_id="image-reference",
        protocol="synthetic",
        messages=[message(m, path=f"messages.{i}.") for i, m in enumerate(vector["messages"])],
        render_options={"mm_processor_kwargs": vector["overrides"]} if vector["overrides"] else {},
        render_option_sources={"mm_processor_kwargs": ["request.extra_body.mm_processor_kwargs"]},
    )


@pytest.mark.assets
@pytest.mark.parametrize("vector", VECTORS, ids=[v["asset"] + ":" + v["name"] for v in VECTORS])
def test_official_image_layout_and_full_input(official_assets, backends, vector):
    asset = vector["asset"]
    backend = backend_for(asset, official_assets, backends)
    call = call_for(vector)
    before = call.model_dump()
    measured = backend.count(call, Binding(aliases=[asset], name=asset))
    for field in ("input_tokens", "input_ids_sha256", "rendered_input_sha256"):
        assert measured[field] == vector[field]
    assert [i["visual_tokens"] for i in measured["input_images"]] == vector["visual_tokens"]
    assert [i["input_tokens"] for i in measured["input_images"]] == vector["image_input_tokens"]
    assert measured["image_input_tokens"] == sum(vector["image_input_tokens"])
    assert call.model_dump() == before
    assert measured.get("input_ids_scope", "reference_input_ids") == (
        "tokenizer_ids_before_image_expansion"
        if vector["backend"] == "moonshot"
        else "reference_input_ids"
    )
    for item in measured["input_images"]:
        assert item["execution"] == "completed" and item["reason"] is None
        for key in vector["overrides"]:
            assert item["parameter_sources"][key] == [
                "request.extra_body.mm_processor_kwargs." + key
            ]


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
def test_unknown_image_does_not_become_text_only_success(official_assets, backends, asset):
    vector = copy.deepcopy(
        next(v for v in VECTORS if v["asset"] == asset and v["name"] == "two_images")
    )
    vector["messages"][0]["content"][3] = {
        "type": "image_url",
        "image_url": {"url": "https://never.invalid/image"},
    }
    backend = backend_for(asset, official_assets, backends)
    with pytest.raises(UnsupportedInput, match="missing_image_dimensions") as caught:
        backend.count(call_for(vector), Binding(aliases=[asset], name=asset))
    images = caught.value.evidence["input_images"]
    assert images[0]["visual_tokens"] == vector["visual_tokens"][0]
    assert images[0]["execution"] == "completed"
    assert images[1]["execution"] == "not_run" and images[1]["visual_tokens"] is None


@pytest.mark.assets
@pytest.mark.parametrize(
    "asset", ["minimax-m3", "mimo-v2.6", "qwen3.5-122b", "qwen3.5-397b", "glm-5.3-flash"]
)
def test_logged_grid_can_supply_dimensions_but_must_agree_when_both_exist(
    official_assets, backends, asset
):
    backend = backend_for(asset, official_assets, backends)
    binding = Binding(aliases=[asset], name=asset)
    vector = copy.deepcopy(
        next(v for v in VECTORS if v["asset"] == asset and v["name"] == "640x480")
    )
    original = backend.count(call_for(vector), binding)
    grid = original["input_images"][0]["layout"]["image_grid_thw"]
    block = vector["messages"][0]["content"][1]
    block.update(image_grid_thw=grid)
    del block["width"], block["height"]
    assert backend.count(call_for(vector), binding)["input_tokens"] == original["input_tokens"]
    block.update(width=1, height=1)
    with pytest.raises(UnsupportedInput, match="conflicting_image_grid_thw"):
        backend.count(call_for(vector), binding)


@pytest.mark.assets
@pytest.mark.parametrize(
    "options,reason",
    [
        ({"max_pixels": False}, "invalid_image_option:max_pixels"),
        ({"max_pixels": 0}, "invalid_image_option:max_pixels"),
        ({"min_pixels": 3000, "max_pixels": 1000}, "invalid_image_pixel_limits"),
        ({"crop": True}, "unsupported_image_option:crop"),
        ({"image_grid_thw": [1, 2, 2]}, "unsupported_image_option:image_grid_thw"),
    ],
)
def test_image_options_are_never_silently_ignored(official_assets, backends, options, reason):
    vector = copy.deepcopy(next(v for v in VECTORS if v["asset"] == "mimo-v2.6"))
    vector["overrides"] = options
    with pytest.raises(UnsupportedInput, match=reason):
        backend_for("mimo-v2.6", official_assets, backends).count(
            call_for(vector), Binding(aliases=["image"], name="mimo-v2.6")
        )


@pytest.mark.assets
def test_text_asset_does_not_claim_an_image_reference(official_assets, backends):
    backend = backend_for("qwen-3.8-2.4t", official_assets, backends)
    with pytest.raises(UnsupportedInput, match="asset_has_no_image_reference"):
        backend.count(call_for(VECTORS[0]), Binding(aliases=["image"], name="qwen-3.8-2.4t"))


@pytest.mark.assets
def test_glm_image_budget_must_be_valid(official_assets, backends):
    vector = copy.deepcopy(next(v for v in VECTORS if v["asset"] == "glm-5.3-flash"))
    vector["overrides"] = {"min_image_tokens": 200, "max_image_tokens": 100}
    with pytest.raises(UnsupportedInput, match="invalid_image_token_limits"):
        backend_for("glm-5.3-flash", official_assets, backends).count(
            call_for(vector), Binding(aliases=["glm"], name="glm-5.3-flash")
        )


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
def test_exif_orientation_follows_the_selected_fixed_processor(official_assets, backends, asset):
    vector = copy.deepcopy(
        next(v for v in VECTORS if v["asset"] == asset and v["name"] == "1000x50")
    )
    image = Image.new("RGB", (1000, 50))
    exif = image.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif.tobytes())
    vector["messages"][0]["content"][1] = {
        "type": "image",
        "data": base64.b64encode(buffer.getvalue()).decode(),
    }
    measured = backend_for(asset, official_assets, backends).count(
        call_for(vector), Binding(aliases=[asset], name=asset)
    )
    transpose = load_tokenizer_assets()["assets"][asset]["image_processing"]["exif_transpose"]
    expected = next(
        v
        for v in VECTORS
        if v["asset"] == asset and v["name"] == ("50x1000" if transpose else "1000x50")
    )
    assert measured["input_tokens"] == expected["input_tokens"]
    assert measured["input_images"][0]["layout"].get("exif_transpose", False) == transpose


@pytest.mark.assets
def test_conflicting_image_parameters_and_literal_placeholder_are_rejected(
    official_assets, backends
):
    vector = copy.deepcopy(next(v for v in VECTORS if v["asset"] == "mimo-v2.6"))
    vector["overrides"] = {"max_pixels": 102400}
    vector["messages"][0]["content"][1]["max_pixels"] = 409600
    with pytest.raises(UnsupportedInput, match="conflicting_image_option:max_pixels"):
        backend_for("mimo-v2.6", official_assets, backends).count(
            call_for(vector), Binding(aliases=["mimo"], name="mimo-v2.6")
        )
    vector = copy.deepcopy(next(v for v in VECTORS if v["asset"] == "kimi-k3"))
    vector["messages"][0]["content"][0]["text"] = "<|kimi_image_placeholder|>"
    with pytest.raises(UnsupportedInput, match="image_placeholder_count_mismatch"):
        backend_for("kimi-k3", official_assets, backends).count(
            call_for(vector), Binding(aliases=["kimi"], name="kimi-k3")
        )
