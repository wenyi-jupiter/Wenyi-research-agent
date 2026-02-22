"""Minimax LLM provider implementation."""

import json
from typing import Any, AsyncIterator, Sequence

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from agent_engine.config import get_settings
from agent_engine.llm.base import BaseLLMProvider, LLMResponse, StreamChunk


class MinimaxChatModel(BaseChatModel):
    """LangChain-compatible Minimax chat model wrapper."""

    def __init__(self, api_key: str, group_id: str, model: str = "abab6.5-chat", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.group_id = group_id
        self.model = model
        self.base_url = "https://api.minimax.chat/v1"

    @property
    def _llm_type(self) -> str:
        return "minimax"

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        """Synchronous generation (not implemented for async)."""
        raise NotImplementedError("Use async methods")

    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs):
        """Async generation."""
        # This is a simplified wrapper - full implementation would handle tool calls
        messages = kwargs.get("messages", [])
        if not messages:
            return

        # Convert LangChain messages to Minimax format
        minimax_messages = self._convert_messages(messages)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/text/chatcompletion_v2",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": minimax_messages,
                    "group_id": self.group_id,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract response
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                return self._create_result(content)

    def _convert_messages(self, messages: Sequence[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to Minimax format."""
        minimax_messages = []
        for msg in messages:
            role = "USER"
            if isinstance(msg, SystemMessage):
                role = "SYSTEM"
            elif isinstance(msg, AIMessage):
                role = "ASSISTANT"

            minimax_messages.append({
                "role": role,
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            })
        return minimax_messages

    def _create_result(self, content: str):
        """Create a LangChain generation result."""
        from langchain_core.outputs import ChatGeneration, ChatResult

        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])


class MinimaxProvider(BaseLLMProvider):
    """Minimax LLM provider."""

    def __init__(self, model: str | None = None, **kwargs: Any):
        """Initialize Minimax provider.

        Args:
            model: Model name (default from settings).
            **kwargs: Additional arguments.
        """
        settings = get_settings()
        model = model or settings.default_minimax_model
        super().__init__(model, **kwargs)

        self._api_key = kwargs.pop("api_key", None) or settings.minimax_api_key
        self._group_id = kwargs.pop("group_id", None) or settings.minimax_group_id

        if not self._api_key:
            raise ValueError("Minimax API key is required")
        if not self._group_id:
            raise ValueError("Minimax Group ID is required")

        self.base_url = "https://api.minimax.chat/v1"

    @property
    def provider_name(self) -> str:
        return "minimax"

    def get_langchain_model(self, tools: Sequence[BaseTool] | None = None) -> BaseChatModel:
        """Get a LangChain-compatible Minimax model instance."""
        return MinimaxChatModel(
            api_key=self._api_key,
            group_id=self._group_id,
            model=self.model,
            **self._kwargs,
        )

    async def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Invoke Minimax with messages."""
        # Convert messages to Minimax format
        minimax_messages = []
        for msg in messages:
            role = "USER"
            if isinstance(msg, SystemMessage):
                role = "SYSTEM"
            elif isinstance(msg, AIMessage):
                role = "ASSISTANT"
            elif isinstance(msg, HumanMessage):
                role = "USER"
            elif isinstance(msg, ToolMessage):
                role = "TOOL"

            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            message_payload = {
                "role": role,
                "content": content,
            }
            if isinstance(msg, ToolMessage):
                tool_call_id = str(getattr(msg, "tool_call_id", "") or "")
                if tool_call_id:
                    message_payload["tool_call_id"] = tool_call_id
            minimax_messages.append(message_payload)

        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": minimax_messages,
            "group_id": self._group_id,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/text/chatcompletion_v2",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # Extract response
        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(
                content="",
                model=self.model,
                finish_reason="error",
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

        # Extract tool calls if present
        tool_calls = []
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "args": json.loads(tc.get("function", {}).get("arguments", "{}")),
                })

        # Extract usage
        usage_data = data.get("usage", {})
        usage = {
            "input_tokens": usage_data.get("prompt_tokens", 0),
            "output_tokens": usage_data.get("completion_tokens", 0),
        }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=self.model,
            finish_reason=finish_reason,
        )

    async def stream(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream responses from Minimax."""
        # Convert messages
        minimax_messages = []
        for msg in messages:
            role = "USER"
            if isinstance(msg, SystemMessage):
                role = "SYSTEM"
            elif isinstance(msg, AIMessage):
                role = "ASSISTANT"
            elif isinstance(msg, HumanMessage):
                role = "USER"
            elif isinstance(msg, ToolMessage):
                role = "TOOL"

            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            message_payload = {
                "role": role,
                "content": content,
            }
            if isinstance(msg, ToolMessage):
                tool_call_id = str(getattr(msg, "tool_call_id", "") or "")
                if tool_call_id:
                    message_payload["tool_call_id"] = tool_call_id
            minimax_messages.append(message_payload)

        payload = {
            "model": self.model,
            "messages": minimax_messages,
            "group_id": self._group_id,
            "stream": True,
        }

        if tools:
            payload["tools"] = self._convert_tools(tools)

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/text/chatcompletion_v2",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip() or line.startswith("data: "):
                        continue

                    if line.startswith("data: "):
                        line = line[6:]

                    if line.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(line)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            finish_reason = choices[0].get("finish_reason")

                            if content:
                                yield StreamChunk(content=content, finish_reason=finish_reason)
                    except json.JSONDecodeError:
                        continue

    def _convert_tools(self, tools: Sequence[BaseTool]) -> list[dict]:
        """Convert LangChain tools to Minimax format."""
        minimax_tools = []
        for tool in tools:
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                },
            }

            # Convert parameters schema
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema = tool.args_schema.model_json_schema()
                tool_dict["function"]["parameters"] = {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                }
            else:
                tool_dict["function"]["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }

            minimax_tools.append(tool_dict)

        return minimax_tools
