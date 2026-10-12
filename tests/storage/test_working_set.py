from decimal import Decimal

import pytest

from modulo.core.domain.data_models import RawRecord
from modulo.shared.serialization import dumps, iter_dumps, loads
from modulo.storage.working_set import WorkingSet


def test_disk_collections_preserve_values_order_and_streaming_json(tmp_path):
    value = {"z": Decimal("1.2300"), "a": [False, 0, None, 2**90]}
    with WorkingSet(tmp_path / "state", memory_mib=8) as working:
        mapping = working.mapping("values")
        mapping["b"], mapping["a"] = value, value
        assert list(mapping["b"]) == ["z", "a"]
        assert mapping["b"] == value
        sequence = working.sequence("rows")
        sequence.extend([value, {"second": True}])
        assert sequence[-1] == {"second": True}
        assert loads("".join(iter_dumps(mapping))) == loads(dumps(dict(mapping)))
        assert "".join(iter_dumps(sequence)) == dumps(list(sequence))
        groups = working.groups("events")
        groups.append(("request", None), value)
        groups.append(("request", None), {"second": True})
        key, group = next(groups.items())
        assert key == ("request", None)
        assert list(group) == list(sequence)
        assert group.byte_size > 0


def test_raw_and_cursor_rollback_together_and_derived_data_is_rebuilt(tmp_path):
    state = tmp_path / "state"
    with pytest.raises(KeyboardInterrupt):
        with WorkingSet(state) as working:
            working.autocommit = False
            working.remember(RawRecord("line:1", b"one\n", {"file": {"offset": 4}}))
            working.commit()
            working.remember(RawRecord("line:2", b"two\n", {"file": {"offset": 8}}))
            raise KeyboardInterrupt
    with WorkingSet(state) as working:
        assert [row.data for row in working.raw_records()] == [b"one\n"]
        assert working.mapping("saved:cursors")["file"] == {"offset": 4}
        working.mapping("derived")["key"] = "discard on reopening"
        working.mapping("saved:measurements")["key"] = "keep"
        assert working.remember(RawRecord("line:1", b"one\n")) == 0
    with WorkingSet(state) as working:
        assert not working.mapping("derived")
        assert working.mapping("saved:measurements")["key"] == "keep"


def test_state_refuses_parallel_writer_and_changed_rules(tmp_path):
    with WorkingSet(tmp_path) as first:
        first.bind({"namespace": "one"})
        with pytest.raises(ValueError, match="already in use"):
            with WorkingSet(tmp_path):
                pass
    with WorkingSet(tmp_path) as second:
        second.bind({"namespace": "one"})
        with pytest.raises(ValueError, match="different rules"):
            second.bind({"namespace": "two"})
