"""Qwen (通义千问) LLM provider implementation using Aliyun DashScope API."""

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
from langchain_openai import ChatOpenAI

from agent_engine.config import get_settings
from agent_engine.llm.base import BaseLLMProvider, LLMResponse, StreamChunk


class QwenProvider(BaseLLMProvider):
    """Qwen (通义千问) LLM provider using Aliyun DashScope OpenAI-compatible API."""

    def __init__(self, model: str | None = None, **kwargs: Any):
        """Initialize Qwen provider.

        Args:
            model: Model name (default from settings).
            **kwargs: Additional arguments.
        """
        settings = get_settings()
        model = model or settings.default_qwen_model
        super().__init__(model, **kwargs)

        self._api_key = kwargs.pop("api_key", None) or settings.dashscope_api_key
        self._base_url = kwargs.pop("base_url", None) or settings.dashscope_base_url

        # Web search configuration
        self._enable_search = kwargs.pop("enable_search", None)
        if self._enable_search is None:
            self._enable_search = settings.qwen_enable_search
        self._search_strategy = kwargs.pop("search_strategy", None) or settings.qwen_search_strategy
        self._forced_search = kwargs.pop("forced_search", None)
        if self._forced_search is None:
            self._forced_search = settings.qwen_forced_search

        if not self._api_key:
            raise ValueError("DashScope API key is required. Set DASHSCOPE_API_KEY in .env")

    @property
    def provider_name(self) -> str:
        return "qwen"

    def get_langchain_model(
        self,
        tools: Sequence[BaseTool] | None = None,
        enable_search: bool | None = None,
    ) -> BaseChatModel:
        """Get a LangChain-compatible Qwen model instance.
        
        Uses OpenAI-compatible interface provided by DashScope.
        
        Args:
            tools: Optional tools to bind to the model.
            enable_search: Override built-in search. None uses default setting.
                Set to False when you want the LLM to use web_search tool
                instead of built-in search, for proper citation tracking.
        """
        # Build extra kwargs for DashScope-specific features
        extra_kwargs = dict(self._kwargs)
        extra_kwargs.pop("model_kwargs", None)

        # Determine if built-in search should be enabled
        use_search = enable_search if enable_search is not None else self._enable_search

        # Enable DashScope built-in web search via extra_body
        # ChatOpenAI supports extra_body as a direct parameter for non-standard API fields
        extra_body = {}
        if use_search:
            extra_body["enable_search"] = True
            search_options = {"search_strategy": self._search_strategy}
            if self._forced_search:
                search_options["forced_search"] = True
            extra_body["search_options"] = search_options

        model = ChatOpenAI(
            model=self.model,
            api_key=self._api_key,
            base_url=self._base_url,
            extra_body=extra_body if extra_body else None,
            **extra_kwargs,
        )
        
        if tools:
            return model.bind_tools(tools)
        return model

    async def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Invoke Qwen with messages using OpenAI-compatible API."""
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, ToolMessage):
                role = "tool"

            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            message_payload = {
                "role": role,
                "content": content,
            }
            if isinstance(msg, ToolMessage):
                tool_call_id = str(getattr(msg, "tool_call_id", "") or "")
                if tool_call_id:
                    message_payload["tool_call_id"] = tool_call_id
            openai_messages.append(message_payload)

        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": openai_messages,
        }

        # Add web search configuration (DashScope built-in feature)
        if self._enable_search:
            payload["enable_search"] = True
            payload["search_options"] = {
                "search_strategy": self._search_strategy,
            }
            if self._forced_search:
                payload["search_options"]["forced_search"] = True

        # Add tools if provided
        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
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
        content = message.get("content", "") or ""
        finish_reason = choice.get("finish_reason", "stop")

        # Extract tool calls if present
        tool_calls = []
        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "args": json.loads(tc.get("function", {}).get("arguments", "{}")),
                })

        # Extract usage — handle both OpenAI and DashScope key formats:
        #   OpenAI standard: {"prompt_tokens": N, "completion_tokens": N}
        #   DashScope native: {"input_tokens": N, "output_tokens": N}
        usage_data = data.get("usage", {})
        usage = {
            "input_tokens": (
                usage_data.get("prompt_tokens", 0)
                or usage_data.get("input_tokens", 0)
            ),
            "output_tokens": (
                usage_data.get("completion_tokens", 0)
                or usage_data.get("output_tokens", 0)
            ),
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
        """Stream responses from Qwen."""
        # Convert messages
        openai_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, ToolMessage):
                role = "tool"

            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            message_payload = {
                "role": role,
                "content": content,
            }
            if isinstance(msg, ToolMessage):
                tool_call_id = str(getattr(msg, "tool_call_id", "") or "")
                if tool_call_id:
                    message_payload["tool_call_id"] = tool_call_id
            openai_messages.append(message_payload)

        payload = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Add web search configuration (DashScope built-in feature)
        if self._enable_search:
            payload["enable_search"] = True
            payload["search_options"] = {
                "search_strategy": self._search_strategy,
            }
            if self._forced_search:
                payload["search_options"]["forced_search"] = True

        if tools:
            payload["tools"] = self._convert_tools(tools)

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
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
        """Convert LangChain tools to OpenAI format."""
        openai_tools = []
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

            openai_tools.append(tool_dict)

        return openai_tools
