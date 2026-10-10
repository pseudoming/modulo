"""Normalize collected gateway HTTP bodies into the existing chat contract."""

from modulo.core.domain.data_models import Call, DataError, ProtocolContext, Usage
from modulo.shared.serialization import digest

from .chat import decode_response, message, output_fields, request_options
from .fields import text_or_none, usage_from
from .gateway_logs import body_object
from .sse import assemble_sse


def response_fields(response, format="openai"):
    if format not in ("openai", "ali", "ollama"):
        raise DataError("unsupported_gateway_response_format")
    if response is None:
        return {"output_gaps": ["missing_response_body"]}, Usage(), None
    response = body_object(response)
    if response.get("error") or (format == "ali" and response.get("code")):
        return {"output_gaps": ["gateway_error_response"]}, Usage(), response.get("model")
    if format == "ali":
        output = response.get("output") or {}
        if not isinstance(output, dict):
            raise DataError("invalid_ali_response_output")
        body = decode_response(output)
        if body is None and isinstance(output.get("text"), str):
            body = {"content": output["text"]}
        usage = usage_from(response.get("usage"), "input_tokens", "output_tokens")
    elif format == "ollama":
        body = response.get("message")
        if body is not None and not isinstance(body, dict):
            raise DataError("invalid_ollama_response_message")
        usage = usage_from(
            {
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
            },
            "prompt_eval_count",
            "eval_count",
        )
    else:
        body = decode_response(response)
        usage = usage_from(response.get("usage"), "prompt_tokens", "completion_tokens")
    return output_fields(body, path=f"response_body.{format}."), usage, response.get("model")


def build_call(
    gateway,
    request_id,
    attempt_id,
    request,
    output,
    usage,
    response_model=None,
    *,
    gaps=(),
    status="unknown",
):
    request = body_object(request) if request is not None else {}
    values = request.get("messages")
    if values is None and isinstance(request.get("input"), dict):
        values = request["input"].get("messages")  # One API's Alibaba outbound body.
    input_gaps = list(gaps)
    if status == "failed":
        input_gaps.append("failed_call")
    messages = None
    if not isinstance(values, list) or not values:
        input_gaps.append("missing_messages")
    else:
        try:
            messages = [
                message(body_object(v), path=f"request_body.messages.{i}.")
                for i, v in enumerate(values)
            ]
        except DataError as exc:
            input_gaps.append(str(exc))
    options, sources, tools, option_gaps = request_options(
        ("request_body", request), ("request_body.parameters", request.get("parameters"))
    )
    input_gaps.extend(option_gaps)
    return Call(
        call_id="body:" + digest([request_id, attempt_id])[:32],
        protocol=gateway,
        model=text_or_none(request.get("model", response_model), "model") or None,
        messages=messages,
        tools_json=tools,
        render_options=options,
        render_option_sources=sources,
        usage=usage,
        gaps=sorted(set(input_gaps)),
        status=status,
        **output,
    )


def exchange(record, gateway):
    if record.get("schema_version") != "gateway.exchange.v1":
        raise DataError("unsupported_gateway_exchange_schema")
    if record.get("gateway", gateway) != gateway:
        raise DataError("gateway_exchange_protocol_mismatch")
    request_id, attempt_id = record.get("request_id"), record.get("attempt_id")
    if not isinstance(request_id, str) or not request_id:
        raise DataError("gateway_request_id_missing")
    if attempt_id is not None and (not isinstance(attempt_id, str) or not attempt_id):
        raise DataError("invalid_gateway_attempt_id")
    request = record.get("request_body")
    if record.get("request_stage", "unspecified") not in ("ingress", "upstream", "unspecified"):
        raise DataError("invalid_gateway_request_stage")
    response_format = record.get("response_format", "openai")
    if response_format not in ("openai", "ali", "ollama"):
        raise DataError("unsupported_gateway_response_format")
    if record.get("response_body") is not None and record.get("response_sse") is not None:
        raise DataError("ambiguous_gateway_response")
    stream = None
    if record.get("response_sse") is not None:
        if response_format != "openai":
            raise DataError("unsupported_gateway_stream_format")
        if not isinstance(record["response_sse"], str):
            raise DataError("response_sse_must_be_text")
        stream = assemble_sse(record["response_sse"].splitlines())
        output, usage, model = stream.output, stream.usage, stream.model
    else:
        output, usage, model = response_fields(record.get("response_body"), response_format)
    complete = record.get("response_complete", True)
    if type(complete) is not bool:
        raise DataError("invalid_response_complete")
    if not complete:
        output["output_gaps"] = [*output.get("output_gaps", []), "response_capture_incomplete"]
    code = record.get("status_code")
    if code is not None and (type(code) is not int or not 100 <= code <= 599):
        raise DataError("invalid_gateway_status_code")
    status = (
        "failed"
        if code is not None and code >= 400
        else (
            "success"
            if code is not None and 200 <= code < 300 and complete and not output.get("output_gaps")
            else "unknown"
        )
    )
    call = build_call(gateway, request_id, attempt_id, request, output, usage, model, status=status)
    context = ProtocolContext(record.get("metadata"), "metadata")
    assembly = {
        "schema_version": "gateway-assembly.v1",
        "kind": "exchange",
        "request_id": request_id,
        "attempt_id": attempt_id,
        "link_basis": "single_captured_exchange",
        "request_stage": record.get("request_stage", "unspecified"),
        "response_kind": "sse" if stream else record.get("response_format", "openai"),
        "response_complete": complete and not output.get("output_gaps"),
    }
    return call, context, assembly
