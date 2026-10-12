import io
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from modulo.adapters.tokenizers.shared.assets import verified_asset
from modulo.application.tokenizer_assets import load_tokenizer_assets, prepare_assets
from modulo.core.domain.interfaces import AssetError
from modulo.shared.serialization import bytes_digest


@pytest.mark.assets
def test_only_selected_assets_are_verified(official_assets):
    verified = prepare_assets(official_assets, ["kimi-k3", "kimi-k3"])
    assert set(verified) == {
        f"kimi-k3/{name}" for name in load_tokenizer_assets()["assets"]["kimi-k3"]["files"]
    }


def test_unknown_prepare_selection_fails_before_creating_directory(tmp_path):
    root = tmp_path / "assets"
    with pytest.raises(ValueError, match="unknown tokenizer assets"):
        prepare_assets(root, ["unknown"])
    assert not root.exists()


def test_nested_reference_is_prepared_and_checked_as_part_of_the_asset(tmp_path, monkeypatch):
    data = b"# fixed reference, not executed\n"
    lock = {
        "assets": {
            "nested": {
                "repo": "test/repo",
                "revision": "fixed",
                "files": {"encoding/reference.py": bytes_digest(data)},
            }
        }
    }
    monkeypatch.setattr("modulo.application.tokenizer_assets.load_tokenizer_assets", lambda: lock)
    monkeypatch.setattr(
        "modulo.application.tokenizer_assets.urlopen", lambda *a, **kw: io.BytesIO(data)
    )
    assert prepare_assets(tmp_path) == ["nested/encoding/reference.py"]
    verified_asset(tmp_path, "nested", lock, "test/repo")
    (tmp_path / "nested/encoding/reference.py").write_bytes(b"changed")
    with pytest.raises(AssetError, match="asset_hash_mismatch"):
        verified_asset(tmp_path, "nested", lock, "test/repo")
    (tmp_path / "nested/encoding/reference.py").write_bytes(data)
    (tmp_path / "nested/encoding/unlisted.py").write_text("")
    verified_asset(tmp_path, "nested", lock, "test/repo")


def test_catalog_is_independent_of_the_working_directory(tmp_path, monkeypatch):
    expected = load_tokenizer_assets()
    (tmp_path / "static").mkdir()
    (tmp_path / "static/tokenizer-assets.json").write_text('{"assets": {}}')
    monkeypatch.chdir(tmp_path)
    assert load_tokenizer_assets() == expected


@pytest.fixture
def download_catalog():
    return {
        "assets": {
            "sample": {
                "repo": "author/official",
                "revision": "hf-commit",
                "files": {"encoding/template file.py": bytes_digest(b"fixed contents")},
                "modelscope": {"repo": "other-author/official", "revision": "ms-commit"},
            }
        }
    }


@pytest.mark.parametrize("source", ["global", "cn"])
def test_selected_source_uses_its_own_repository_and_revision(
    tmp_path, monkeypatch, download_catalog, source
):
    requests = []

    def download(request, **kwargs):
        requests.append(urlsplit(request.full_url))
        return io.BytesIO(b"fixed contents")

    monkeypatch.setattr("modulo.application.tokenizer_assets.urlopen", download)
    expected = ["sample/encoding/template file.py"]
    assert prepare_assets(tmp_path, source=source, catalog=download_catalog) == expected
    request = requests[0]
    if source == "cn":
        assert request.netloc == "modelscope.cn"
        assert request.path == "/api/v1/models/other-author/official/repo"
        assert parse_qs(request.query) == {
            "Revision": ["ms-commit"],
            "FilePath": ["encoding/template file.py"],
        }
    else:
        assert request.netloc == "huggingface.co"
        assert request.path.endswith("author/official/hf-commit/encoding/template%20file.py")
    requests.clear()
    other_source = "cn" if source == "global" else "global"
    assert prepare_assets(tmp_path, source=other_source, catalog=download_catalog) == expected
    assert requests == []  # Either region reuses the same verified bytes.


def test_cn_digest_mismatch_does_not_publish_or_fall_back(tmp_path, monkeypatch, download_catalog):
    requests = []

    def download(request, **kwargs):
        requests.append(urlsplit(request.full_url).netloc)
        return io.BytesIO(b"different revision")

    monkeypatch.setattr("modulo.application.tokenizer_assets.urlopen", download)
    with pytest.raises(ValueError, match="asset digest mismatch"):
        prepare_assets(tmp_path, source="cn", catalog=download_catalog)
    assert requests == ["modelscope.cn"]
    assert not list(tmp_path.rglob("*.py"))
    assert not list(tmp_path.rglob("*.download"))


def test_cn_http_error_does_not_contact_global(tmp_path, monkeypatch, download_catalog):
    requests = []

    def download(request, **kwargs):
        requests.append(urlsplit(request.full_url).netloc)
        raise HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr("modulo.application.tokenizer_assets.urlopen", download)
    monkeypatch.setattr("modulo.application.tokenizer_assets.time.sleep", lambda _: None)
    with pytest.raises(HTTPError):
        prepare_assets(tmp_path, source="cn", catalog=download_catalog)
    assert requests == ["modelscope.cn"] * 3


def test_missing_cn_source_does_not_guess_repository(tmp_path, download_catalog):
    del download_catalog["assets"]["sample"]["modelscope"]
    with pytest.raises(ValueError, match="no locked ModelScope source"):
        prepare_assets(tmp_path, source="cn", catalog=download_catalog)


def test_unknown_region_fails_before_creating_directories(tmp_path, download_catalog):
    root = tmp_path / "missing"
    with pytest.raises(ValueError, match="asset source must be"):
        prepare_assets(root, source="unknown", catalog=download_catalog)
    assert not root.exists()
