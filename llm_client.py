"""LLM client for AgentLocD.

This module wraps Alibaba Cloud Bailian / DashScope Qwen models through an
OpenAI-compatible interface. API keys are never hard-coded. Set
DASHSCOPE_API_KEY or BAILIAN_API_KEY in the environment before running.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class LLMConfig:
    model: str = os.getenv("AGENTLOCD_LLM_MODEL", "qwen3.5-max")
    base_url: str = os.getenv(
        "AGENTLOCD_LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    temperature: float = float(os.getenv("AGENTLOCD_LLM_TEMPERATURE", "0"))
    max_tokens: int = int(os.getenv("AGENTLOCD_LLM_MAX_TOKENS", "2048"))


class BailianQwenClient:
    """OpenAI-compatible client for Qwen models served by Alibaba Cloud Bailian.

    The API key must be provided through one of the following environment variables:
    DASHSCOPE_API_KEY, BAILIAN_API_KEY, or ALIBABA_CLOUD_API_KEY.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()
        api_key = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("BAILIAN_API_KEY")
            or os.getenv("ALIBABA_CLOUD_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Missing LLM API key. Set DASHSCOPE_API_KEY or BAILIAN_API_KEY "
                "in your local environment. Do not commit the key to GitHub."
            )
        self.client = OpenAI(api_key=api_key, base_url=self.config.base_url)

    def chat_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the model and parse a JSON object from the response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, default=str),
            },
        ]
        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        return self._parse_json(text)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise
            return json.loads(match.group(0))
