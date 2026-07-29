from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any


class CodeAgent:
    """Minimal client for an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, *, model: str, base_url: str, api_key_env: str = "LLM_API_KEY"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env

    @staticmethod
    def extract_python(content: str) -> str:
        content = content.strip()
        if "```python" in content:
            return content.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in content:
            return content.split("```", 1)[1].split("```", 1)[0].strip()
        return content

    def complete(self, system: str, user: str, *, timeout: float = 240) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"]

    def generate(self, plan: dict[str, Any], *, system_prompt: Path, user_prompt: Path, timeout: float = 240) -> str:
        system = system_prompt.read_text(encoding="utf-8")
        user = user_prompt.read_text(encoding="utf-8").replace(
            "{{PLAN_JSON}}", json.dumps(plan, ensure_ascii=False, indent=2)
        )
        return self.extract_python(self.complete(system, user, timeout=timeout))

    def repair(
        self, plan: dict[str, Any], code: str, issues: list[str], *,
        system_prompt: Path, user_prompt: Path, timeout: float = 240,
    ) -> str:
        system = system_prompt.read_text(encoding="utf-8")
        user = user_prompt.read_text(encoding="utf-8")
        user = user.replace("{{ISSUE_TEXT}}", "\n".join(issues))
        user = user.replace("{{TASK_JSON}}", json.dumps(plan, ensure_ascii=False, indent=2))
        user = user.replace("{{CODE}}", code)
        return self.extract_python(self.complete(system, user, timeout=timeout))
