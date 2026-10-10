import pytest

from modulo.adapters.protocols import DifyProtocol
from modulo.adapters.sources import FileSource
from modulo.application.importer import run_batch
from modulo.shared.serialization import dumps


def test_projection_failure_never_publishes_a_partially_prepared_bundle(
    tmp_path, unit_calculator, dify_record, monkeypatch
):
    def fail_index(directory):
        assert (directory / "manifest.json").is_file()
        raise RuntimeError("injected_index_failure")

    output = tmp_path / "runs"
    source = tmp_path / "logs.jsonl"
    source.write_text(dumps(dify_record) + "\n")
    monkeypatch.setattr("modulo.application.importer.rebuild_index", fail_index)
    with pytest.raises(RuntimeError, match="injected_index_failure"):
        run_batch(
            FileSource(source),
            DifyProtocol(),
            unit_calculator,
            namespace="save-failure",
            output_root=output,
        )
    assert list(output.iterdir()) == []
