import hashlib

import pytest

from modulo.adapters.sources import FileSource
from modulo.core.domain.data_models import SourceError


def test_file_preserves_bytes_and_locations(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"x":1}\r\n\n{"x":2}\n')
    records = list(FileSource(path).read())
    assert [record.locator for record in records] == ["line:1", "line:3"]
    assert records[0].data == b'{"x":1}\r\n'


def test_file_detects_change_and_size_limit(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"x":1}\n')
    source = FileSource(path)
    path.write_text('{"x":2}\n')
    with pytest.raises(SourceError, match="changed"):
        list(source.read())
    with pytest.raises(SourceError, match="byte limit"):
        list(FileSource(path, max_record_bytes=3).read())


def test_directory_freezes_matching_files_and_keeps_file_and_line_references(tmp_path):
    second = tmp_path / "dify-20261002.jsonl"
    first = tmp_path / "dify-20261001.jsonl"
    second.write_bytes(b'{"x":2}\n')
    first.write_bytes(b'{"x":1}\r\n\n{"x":3}\n')
    (tmp_path / "dify-20260930.jsonl").write_bytes(b"excluded\n")
    (tmp_path / "dify-20261003.jsonl").mkdir()
    source = FileSource(tmp_path, pattern="dify-202610*.jsonl")
    description = source.describe()
    assert description["files"] == [
        {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in [first, second]
    ]
    # This batch uses the recorded selection; a later file belongs to a later import.
    (tmp_path / "dify-20261004.jsonl").write_bytes(b"new file\n")
    records = list(source.read())
    assert [record.locator for record in records] == [
        "file:dify-20261001.jsonl:line:1",
        "file:dify-20261001.jsonl:line:3",
        "file:dify-20261002.jsonl:line:1",
    ]
    assert [record.data for record in records] == [b'{"x":1}\r\n', b'{"x":3}\n', b'{"x":2}\n']
    assert source.describe() == description


@pytest.mark.parametrize("change", ["modify", "delete"])
def test_selected_file_failure_keeps_the_prefix_and_never_finishes_successfully(tmp_path, change):
    first = tmp_path / "01.log"
    second = tmp_path / "02.log"
    first.write_bytes(b"first\n")
    second.write_bytes(b"second\n")
    source = FileSource(tmp_path, pattern="*.log")
    if change == "modify":
        second.write_bytes(b"changed\n")
    else:
        second.unlink()
    records = source.read()
    assert next(records).data == b"first\n"
    with pytest.raises((SourceError, FileNotFoundError)):
        list(records)


@pytest.mark.parametrize("pattern", [None, "", "../*.log", "/tmp/*.log", "missing-*.log"])
def test_invalid_or_empty_directory_selection_is_rejected(tmp_path, pattern):
    with pytest.raises(ValueError):
        FileSource(tmp_path, pattern=pattern)


def test_nested_pattern_keeps_paths_distinct_and_applies_each_record_limit(tmp_path):
    for name in ["first", "second"]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "access.log").write_bytes(b"ok\n")
    source = FileSource(tmp_path, pattern="**/access.log", max_record_bytes=3)
    assert [record.locator for record in source.read()] == [
        "file:first/access.log:line:1",
        "file:second/access.log:line:1",
    ]
    with pytest.raises(SourceError, match="byte limit"):
        list(FileSource(tmp_path, pattern="**/access.log", max_record_bytes=2).read())
