import copy
import json

import pytest

from modulo.application.registry import builtins
from modulo.application.tokenizer_assets import asset_lock
from modulo.core.domain.bindings import Binding
from modulo.core.domain.data_models import Call, Message
from modulo.core.domain.interfaces import AssetError
from modulo.shared.serialization import bytes_digest, digest
from tests.support.paths import ROOT

VECTORS = json.loads(
    (ROOT / "tests/fixtures/tokenizers/family_reference_vectors.json").read_text()
)["vectors"]


ASSETS = list(dict.fromkeys(v["asset"] for v in VECTORS))


@pytest.fixture(scope="module")
def backends():
    return {}


@pytest.mark.assets
@pytest.mark.parametrize("vector", VECTORS, ids=[v["asset"] + ":" + v["name"] for v in VECTORS])
def test_frozen_family_reference(official_assets, backends, vector):
    asset, family = vector["asset"], vector["backend"]
    if asset not in backends:
        backends[asset] = builtins().get("tokenizers", family)(official_assets, asset, asset_lock())
    backend = backends[asset]
    call = Call(
        call_id="frozen-reference",
        protocol="synthetic",
        messages=[Message(**m) for m in vector["messages"]],
        output_text=vector["output"],
    )
    measured = backend.count(call, Binding(modelnames=[asset], asset=asset))
    assert measured["input_tokens"] == len(vector["input_ids"])
    assert measured["input_ids_sha256"] == digest(vector["input_ids"])
    assert measured["rendered_input_sha256"] == bytes_digest(vector["rendered"].encode())
    assert measured["visible_output_tokens"] == vector["visible_output_tokens"]
    assert measured["asset_revision"] == vector["asset_revision"]
    if family != "moonshot":
        assert (
            backend.tokenizer.apply_chat_template(
                vector["messages"],
                tokenize=True,
                add_generation_prompt=True,
                **asset_lock()["assets"][asset]["template_kwargs"],
            )
            == vector["input_ids"]
        )


@pytest.mark.parametrize("asset", ASSETS)
def test_new_assets_do_not_fall_back_to_downloads(tmp_path, asset):
    entry = asset_lock()["assets"][asset]
    with pytest.raises(AssetError, match="asset_missing"):
        builtins().get("tokenizers", entry["family"])(tmp_path, asset, asset_lock())


def test_unreviewed_repo_is_rejected_before_loading(tmp_path):
    lock = copy.deepcopy(asset_lock())
    lock["assets"]["qwen3.8"]["repo"] = "Qwen/unknown-next-model"
    with pytest.raises(AssetError, match="asset_family_mismatch"):
        builtins().get("tokenizers", "qwen")(tmp_path, "qwen3.8", lock)


@pytest.mark.assets
def test_profile_is_part_of_asset_identity_and_kimi_cannot_ignore_an_override(official_assets):
    lock = copy.deepcopy(asset_lock())
    lock["assets"]["kimi-k3"]["template_kwargs"]["thinking"] = False
    with pytest.raises(AssetError, match="asset_profile_unsupported"):
        builtins().get("tokenizers", "moonshot")(official_assets, "kimi-k3", lock)
