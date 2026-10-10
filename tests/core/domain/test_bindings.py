import pytest

from modulo.core.domain.bindings import Bindings


@pytest.mark.parametrize("names", [[], [""], [" \t"], "model-a"])
def test_aliases_require_a_nonempty_list_of_nonblank_names(names):
    with pytest.raises(ValueError):
        Bindings.model_validate({"models": [{"modelnames": names, "asset": "test-only"}]})


@pytest.mark.parametrize("platform", [None, "platform-a"])
@pytest.mark.parametrize("separate_entries", [False, True])
def test_duplicate_aliases_are_rejected_in_each_binding_scope(platform, separate_entries):
    entries = [{"modelnames": ["model-a", "alias"], "asset": "first"}]
    if separate_entries:
        entries.append({"modelnames": ["alias"], "asset": "second"})
    else:
        entries[0]["modelnames"].append("alias")
    if platform is not None:
        entries = [{**entry, "platform": platform} for entry in entries]
    config = {"models": entries}
    with pytest.raises(ValueError, match="duplicate modelname 'alias'"):
        Bindings.model_validate(config)


@pytest.mark.parametrize(
    "condition",
    [
        {"platform": " "},
        {"platform-tag-source": "metadata:platform"},
        {"platform": "a", "platform-tag-source": "none"},
        {"platform": "a", "platform-tag-source": "field:"},
        {"platform": "a", "platform-tag-source": "endpoint"},
        {"platform": "a", "endpoint-map": {"url": "a"}},
        {"platform": "a", "platform-tag-source": "metadata:platform", "endpoint-map": {"url": "a"}},
    ],
)
def test_incomplete_or_meaningless_platform_conditions_are_rejected(condition):
    with pytest.raises(ValueError):
        Bindings.model_validate(
            {"models": [{"modelnames": ["model-a"], "asset": "first", **condition}]}
        )
