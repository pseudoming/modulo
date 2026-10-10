import pytest

from modulo.application.tokenizer_assets import asset_lock, prepare_assets


@pytest.mark.assets
def test_only_selected_assets_are_verified(official_assets):
    verified = prepare_assets(official_assets, ["kimi-k3", "kimi-k3"])
    assert set(verified) == {
        f"kimi-k3/{name}" for name in asset_lock()["assets"]["kimi-k3"]["files"]
    }


def test_unknown_prepare_selection_fails_before_creating_directory(tmp_path):
    root = tmp_path / "assets"
    with pytest.raises(ValueError, match="unknown tokenizer assets"):
        prepare_assets(root, ["unknown"])
    assert not root.exists()
