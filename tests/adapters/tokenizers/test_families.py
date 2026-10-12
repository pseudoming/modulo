import copy
import json

import pytest

from modulo.application.registry import builtins
from modulo.application.tokenizer_assets import load_tokenizer_assets
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
        backends[asset] = builtins().get("tokenizers", asset)(
            official_assets, load_tokenizer_assets()
        )
    backend = backends[asset]
    call = Call(
        call_id="frozen-reference",
        protocol="synthetic",
        messages=[Message(**m) for m in vector["messages"]],
        output_text=vector["output"],
    )
    measured = {
        **backend.count(call, Binding(aliases=[asset], name=asset)),
        **backend.count_output(call),
    }
    assert measured["input_tokens"] == len(vector["input_ids"])
    assert measured["input_ids_sha256"] == digest(vector["input_ids"])
    assert measured["rendered_input_sha256"] == bytes_digest(vector["rendered"].encode())
    assert measured["visible_output_tokens"] == vector["visible_output_tokens"]
    assert measured["asset_revision"] == vector["asset_revision"]
    if family != "moonshot" and not load_tokenizer_assets()["assets"][asset].get(
        "template_file", ""
    ).endswith(".py"):
        assert (
            backend.tokenizer.apply_chat_template(
                vector["messages"],
                tokenize=True,
                add_generation_prompt=True,
                **load_tokenizer_assets()["assets"][asset]["template_kwargs"],
            )
            == vector["input_ids"]
        )


@pytest.mark.parametrize("asset", ASSETS)
def test_new_assets_do_not_fall_back_to_downloads(tmp_path, asset):
    with pytest.raises(AssetError, match="asset_missing"):
        builtins().get("tokenizers", asset)(tmp_path, load_tokenizer_assets())


def test_unreviewed_repo_is_rejected_before_loading(tmp_path):
    lock = copy.deepcopy(load_tokenizer_assets())
    lock["assets"]["qwen-3.8-2.4t"]["repo"] = "Qwen/unknown-next-model"
    with pytest.raises(AssetError, match="asset_repository_mismatch"):
        builtins().get("tokenizers", "qwen-3.8-2.4t")(tmp_path, lock)


@pytest.mark.assets
def test_profile_is_part_of_asset_identity_and_kimi_cannot_ignore_an_override(official_assets):
    lock = copy.deepcopy(load_tokenizer_assets())
    lock["assets"]["kimi-k3"]["template_kwargs"]["thinking"] = False
    with pytest.raises(AssetError, match="asset_profile_unsupported"):
        builtins().get("tokenizers", "kimi-k3")(official_assets, lock)
