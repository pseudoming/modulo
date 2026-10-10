from decimal import Decimal

import pytest

from modulo.shared.serialization import loads, ordered_json


def test_duplicate_keys_and_nonfinite_json_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        loads('{"id":"first","id":"second"}')
    with pytest.raises(ValueError, match="non-finite"):
        loads('{"usage":NaN}')


def test_escaped_lone_surrogates_rejected_at_boundary():
    with pytest.raises(UnicodeError):
        loads(r'{"text":"\ud800"}')


def test_ordered_prompt_json_keeps_numbers_numeric_and_key_order():
    value = {"z": Decimal("1.2300"), "a": {"x": 2**80, "v": "\n中"}}
    encoded = ordered_json(value)
    assert encoded.startswith('{"z": 1.2300, "a":')
    assert loads(encoded) == value
