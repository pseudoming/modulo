import shutil
from copy import deepcopy

import pytest

from modulo.adapters.tokenizers.qwen import Qwen38Tokenizer
from modulo.application.registry import builtins
from modulo.application.tokenizer_assets import load_tokenizer_assets
from modulo.core.domain.bindings import Binding
from modulo.core.domain.data_models import Call, Message
from modulo.core.domain.interfaces import AssetError
from modulo.shared.serialization import digest, loads
from tests.support.paths import ROOT


@pytest.mark.assets
def test_asset_tamper_is_not_used(official_assets, tmp_path):
    shutil.copytree(official_assets / "qwen-3.8-2.4t", tmp_path / "qwen-3.8-2.4t")
    (tmp_path / "qwen-3.8-2.4t/tokenizer_config.json").write_text("{}")
    with pytest.raises(AssetError, match="asset_hash_mismatch"):
        Qwen38Tokenizer(tmp_path, load_tokenizer_assets())


def test_missing_assets_are_explicit(tmp_path):
    with pytest.raises(AssetError, match="asset_missing"):
        Qwen38Tokenizer(tmp_path, load_tokenizer_assets())


@pytest.mark.assets
def test_unread_sidecars_do_not_change_counting(official_assets, tmp_path):
    shutil.copytree(official_assets / "qwen-3.8-2.4t", tmp_path / "qwen-3.8-2.4t")
    directory = tmp_path / "qwen-3.8-2.4t"
    call = Call(
        call_id="sidecars", protocol="synthetic", messages=[Message(role="user", content="Hi")]
    )
    binding = Binding(name="qwen-3.8-2.4t", aliases=["qwen"])
    before = Qwen38Tokenizer(tmp_path, load_tokenizer_assets()).count(call, binding)
    (directory / "special_tokens_map.json").write_text('{"eos_token": "changed"}')
    (directory / "LICENSE").write_text("changed")
    assert Qwen38Tokenizer(tmp_path, load_tokenizer_assets()).count(call, binding) == before
    (directory / "LICENSE").unlink()
    assert Qwen38Tokenizer(tmp_path, load_tokenizer_assets()).count(call, binding) == before


@pytest.mark.assets
@pytest.mark.parametrize("asset", load_tokenizer_assets()["assets"])
def test_runtime_only_files_reproduce_frozen_token_ids(official_assets, tmp_path, asset):
    lock = load_tokenizer_assets()
    for name in lock["assets"][asset]["files"]:
        target = tmp_path / asset / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(official_assets / asset / name, target)
    vector = next(
        v
        for v in loads(
            (ROOT / "tests/fixtures/tokenizers/family_reference_vectors.json").read_bytes()
        )["vectors"]
        if v["asset"] == asset
    )
    backend = builtins().get("tokenizers", asset)(tmp_path, lock)
    call = Call(
        call_id="runtime-only",
        protocol="synthetic",
        messages=[Message(**m) for m in vector["messages"]],
    )
    measured = backend.count(call, Binding(name=asset, aliases=[asset]))
    assert measured["input_ids_sha256"] == digest(vector["input_ids"])
    assert measured["input_tokens"] == len(vector["input_ids"])


@pytest.mark.parametrize("filename", ["tokenizer.json", "tokenizer_config.json"])
def test_required_file_must_be_locked_before_it_is_read(tmp_path, filename):
    lock = deepcopy(load_tokenizer_assets())
    entry = lock["assets"]["qwen-3.8-2.4t"]
    del entry["files"][filename]
    with pytest.raises(AssetError, match="required_asset_not_locked"):
        Qwen38Tokenizer(tmp_path, lock)


@pytest.mark.assets
@pytest.mark.parametrize("asset", ["qwen-3.8-2.4t", "kimi-k2.6"])
def test_unlocked_template_is_not_loaded(official_assets, asset):
    lock = deepcopy(load_tokenizer_assets())
    entry = lock["assets"][asset]
    del entry["files"][entry["template_file"]]
    with pytest.raises(AssetError, match="template_not_locked"):
        builtins().get("tokenizers", asset)(official_assets, lock)
