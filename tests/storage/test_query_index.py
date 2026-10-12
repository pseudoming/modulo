import duckdb
import pytest

from modulo.core.domain.bindings import Bindings
from modulo.core.domain.data_models import BatchResult
from modulo.storage.query_index import details, rebuild_index
from modulo.storage.result_bundle import BundleWriter, verify_bundle


@pytest.mark.parametrize("platform", [None, "", '\\N,\r\n"审计🙂'])
def test_projection_preserves_null_empty_and_quoted_metadata(tmp_path, platform):
    row = {
        "call_key": "one",
        "call": {
            "call_id": "one",
            "model": '模型\n"name",',
            "platform": platform,
            "status": "success",
        },
        "measurement": {"verdict": "UNVERIFIABLE", "input_verdict": "UNVERIFIABLE"},
    }
    root = tmp_path / "auditor's output"
    with BundleWriter(root) as writer:
        directory, _ = writer.publish(
            {},
            BatchResult([row], [], {"run_status": "COMPLETE"}),
            execution={},
            bindings=Bindings(),
            asset_lock={},
            rebuild_index=rebuild_index,
        )
    assert details(directory) == [row]
    with duckdb.connect(str(directory / "query.duckdb"), read_only=True) as db:
        assert db.execute("SELECT model, platform FROM calls").fetchone() == (
            row["call"]["model"],
            platform,
        )
    assert not list(directory.glob(".query*"))


@pytest.mark.parametrize("format", ["input-only", "input-output", "three-verdicts"])
def test_projection_keeps_large_integers_and_does_not_rewrite_old_results(tmp_path, format):
    large = 10**100 + 1
    measurement = {
        "verdict": "CONSISTENT",
        "reported_input_tokens": large,
        "input_tokens": large,
        "input_delta": 0,
    }
    has_output = format != "input-only"
    if has_output:
        measurement.update(
            reported_output_tokens=large,
            output_tokens=large + 3,
            output_delta=-3,
            output_verdict="DEVIATION",
            output_candidates=[{"index": 0, "execution": "completed", "tokens": large + 3}],
        )
    if format == "three-verdicts":
        measurement.update(verdict="DEVIATION", input_verdict="CONSISTENT")
    row = {
        "call_key": "one",
        "call": {"call_id": "one", "model": "synthetic", "status": "success"},
        "measurement": measurement,
    }
    with BundleWriter(tmp_path) as writer:
        directory, _ = writer.publish(
            {},
            BatchResult([row], [], {"run_status": "COMPLETE"}),
            execution={},
            bindings=Bindings(),
            asset_lock={},
            rebuild_index=rebuild_index,
        )
    before = verify_bundle(directory)
    original = (directory / "results.jsonl").read_bytes()
    assert details(directory) == [row]
    assert verify_bundle(directory) == before
    assert (directory / "results.jsonl").read_bytes() == original
    with duckdb.connect(str(directory / "query.duckdb"), read_only=True) as db:
        assert db.execute(
            "SELECT reported_input_tokens, reported_output_tokens, output_tokens, "
            "output_delta, output_verdict FROM calls"
        ).fetchone() == (
            str(large),
            str(large) if has_output else None,
            str(large + 3) if has_output else None,
            "-3" if has_output else None,
            "DEVIATION" if has_output else None,
        )
        assert db.execute("SELECT output_tokens FROM output_candidates").fetchall() == (
            [(str(large + 3),)] if has_output else []
        )
        assert db.execute(
            "SELECT verdict, input_verdict, output_verdict FROM calls"
        ).fetchone() == (
            "DEVIATION" if format == "three-verdicts" else None,
            "CONSISTENT",
            "DEVIATION" if has_output else None,
        )
