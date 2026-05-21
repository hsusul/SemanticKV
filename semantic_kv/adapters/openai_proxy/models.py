from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OpenAIProxySettings(BaseModel):
    upstream_base_url: str = "http://localhost:8001"
    trace_output_dir: str = "outputs/live_traces"
    redact_text: bool = True
    timeout_seconds: float = 60.0


class ChatCompletionProxyRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False

    def prompt_text(self) -> str:
        return "\n\n".join(f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in self.messages)


class CompletionProxyRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[str]
    stream: bool = False

    def prompt_text(self) -> str:
        if isinstance(self.prompt, list):
            return "\n\n".join(self.prompt)
        return self.prompt

