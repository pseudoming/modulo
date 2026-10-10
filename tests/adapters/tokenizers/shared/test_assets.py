import shutil

import pytest

from modulo.adapters.tokenizers import QwenTokenizer
from modulo.application.tokenizer_assets import asset_lock
from modulo.core.domain.interfaces import AssetError


@pytest.mark.assets
def test_asset_tamper_is_not_used(official_assets, tmp_path):
    shutil.copytree(official_assets / "qwen3.8", tmp_path / "qwen3.8")
    (tmp_path / "qwen3.8/tokenizer_config.json").write_text("{}")
    with pytest.raises(AssetError, match="asset_hash_mismatch"):
        QwenTokenizer(tmp_path, "qwen3.8", asset_lock())


def test_missing_assets_are_explicit(tmp_path):
    with pytest.raises(AssetError, match="asset_missing"):
        QwenTokenizer(tmp_path, "qwen3.8", asset_lock())


@pytest.mark.assets
def test_unlocked_extra_tokenizer_config_rejected(official_assets, tmp_path):
    shutil.copytree(official_assets / "qwen3.8", tmp_path / "qwen3.8")
    (tmp_path / "qwen3.8/special_tokens_map.json").write_text("{}")
    with pytest.raises(AssetError, match="asset_directory_contents_mismatch"):
        QwenTokenizer(tmp_path, "qwen3.8", asset_lock())
