import pytest

from modulo.adapters.protocols import DifyProtocol, LangChainProtocol
from modulo.core.domain.data_models import DataError


def test_unknown_shape_rejected_and_non_llm_explicitly_ignored():
    with pytest.raises(DataError):
        DifyProtocol().parse({"arbitrary": "object"})
    assert DifyProtocol().parse({"node_type": "start"}).ignored == "non_llm_node"
    with pytest.raises(DataError):
        LangChainProtocol().parse({"arbitrary": "object"})
