"""Generate synthetic calls through the real LangChain callback lifecycle.

This is test tooling, not a production log schema or a model service. Token
expectations come from the frozen vectors, never from modulo's tokenizer.
"""

import json
from copy import deepcopy
from uuid import NAMESPACE_URL, uuid5

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler


class SyntheticChatModel(BaseChatModel):
    answer: dict
    input_tokens: int
    output_tokens: int

    @property
    def _llm_type(self):
        return "modulo-local-synthetic"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(**self.answer))],
            llm_output={"token_usage": {"prompt_tokens": self.input_tokens}},
        )

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        reasoning = self.answer["additional_kwargs"]["reasoning_content"]
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", additional_kwargs={"reasoning_content": reasoning})
        )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=self.answer["content"],
                chunk_position="last",
                usage_metadata={
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "total_tokens": self.input_tokens + self.output_tokens,
                },
            )
        )


def serialized_run(vector, model, *, stream=False):
    """Use upstream RunCollector serialization; preserve raw tool argument text."""
    messages = []
    for original in vector["messages"]:
        body = deepcopy(original)
        role = body.pop("role")
        additional = {k: body.pop(k) for k in ("reasoning_content", "tool_calls") if k in body}
        cls = {
            "user": HumanMessage,
            "system": SystemMessage,
            "assistant": AIMessage,
            "tool": ToolMessage,
        }.get(role, ChatMessage)
        if cls is ChatMessage:
            body["role"] = role
        messages.append(cls(**body, additional_kwargs=additional))
    response = {
        "content": vector["output_text"],
        "additional_kwargs": {"reasoning_content": vector["output_reasoning"]},
    }
    collector = RunCollectorCallbackHandler(_schema_format="original+chat")
    parameters = deepcopy(vector["logged_options"])
    if vector["tools"]:
        parameters["tools"] = vector["tools"]
    chat = SyntheticChatModel(
        answer=response,
        input_tokens=len(vector["input_ids"]),
        output_tokens=vector["visible_output_tokens"] + vector["reasoning_output_tokens"],
    )
    config = {
        "callbacks": [collector],
        "run_id": uuid5(NAMESPACE_URL, f"modulo/{vector['asset']}/{vector['name']}/{stream}"),
        "metadata": {"ls_model_name": model},
    }
    if stream:
        list(chat.stream(messages, config=config, **parameters))
    else:
        chat.invoke(messages, config=config, **parameters)
    return json.loads(collector.traced_runs[0].model_dump_json())
