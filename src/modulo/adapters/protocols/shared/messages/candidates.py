"""Keep completion candidates and request-level usage at distinct scopes."""

from modulo.adapters.protocols.shared.messages.chat import (
    CONTAINERS,
    decode_response,
    output_fields,
)
from modulo.shared.serialization import loads


def group_outputs(candidates, gaps=()):
    gaps = list(gaps)
    indexes = [candidate["index"] for candidate in candidates]
    if len(set(indexes)) != len(indexes):
        gaps.append("duplicate_output_choice_index")
    if sorted(set(indexes)) != list(range(len(set(indexes)))):
        gaps.append("output_choice_index_gap")
    if len(candidates) == 1 and indexes == [0]:
        result = {k: v for k, v in candidates[0].items() if k != "index"}
        gaps.extend(result.get("output_gaps", []))
    else:
        result = {"output_candidates": sorted(candidates, key=lambda candidate: candidate["index"])}
        if any(candidate.get("output_gaps") for candidate in candidates):
            gaps.append("incomplete_output_candidates")
    if not candidates:
        gaps.append("missing_output")
    if gaps:
        result["output_gaps"] = sorted(set(gaps))
    return result


def response_outputs(value, *, path="response.choices.", normalize=None):
    if isinstance(value, str):
        try:
            value = loads(value)
        except ValueError:
            return {"output_gaps": ["invalid_response_json"]}
    if not isinstance(value, dict) or not isinstance(value.get("choices"), list):
        return {"output_gaps": ["missing_output"]}
    choices = value["choices"]
    candidates, gaps = [], []
    for position, choice in enumerate(choices):
        if not isinstance(choice, dict):
            gaps.append("invalid_output_choice")
            continue
        index = choice.get("index", 0 if len(choices) == 1 else None)
        if type(index) is not int or index < 0:
            gaps.append("invalid_output_choice_index")
            continue
        body = decode_response({"choices": [choice]})
        if normalize is not None and body is not None:
            body = normalize(body)
        fields = output_fields(body, path=f"{path}{position}.message.")
        for key in ("stop_reason", "stop_sequence"):
            if (
                key in choice
                and fields.get("output_stop_reason_source")
                == f"{path}{position}.message.response_metadata.{key}"
            ):
                fields["output_stop_reason_source"] = f"{path}{position}.{key}"
        candidates.append({"index": index, **fields})
    return group_outputs(candidates, gaps)


def expected_candidates(output, *roots):
    """Read the logged n, never pass a sampling count to the chat template."""
    output = dict(output)
    gaps, values, sources = list(output.get("output_gaps", [])), [], []
    pending = list(roots)
    while pending:
        path, root = pending.pop(0)
        if not isinstance(root, dict):
            continue
        for key, value in root.items():
            if key in CONTAINERS:
                pending.append((f"{path}.{key}", value))
            elif key == "n":
                if type(value) is not int or value < 1:
                    gaps.append("invalid_requested_output_count")
                else:
                    values.append(value)
                    sources.append(f"{path}.n")
    if len(set(values)) > 1:
        gaps.append("conflicting_requested_output_count")
    elif values:
        output["expected_output_candidates"] = values[0]
        output["expected_output_candidates_sources"] = sources
        actual = len(output["output_candidates"]) if "output_candidates" in output else 1
        if actual != values[0]:
            gaps.append("output_candidate_count_mismatch")
    if gaps:
        output["output_gaps"] = sorted(set(gaps))
    return output
