from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class DifyClient:
    """Thin Dify Workflow HTTP client.

    Python sends structured inputs to Dify and reads structured outputs back.
    Prompts, vision reasoning, reply generation, and learning logic stay in Dify.
    """

    def __init__(
        self,
        api_key: str | None,
        api_base: str = "https://api.dify.ai/v1",
        user: str = "memory-agent",
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.user = user
        self.timeout = timeout

    def run_workflow(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run Dify Workflow once.

        The Agent calls this twice:
        - stage="retrieval_query": Dify produces retrieval_query
        - stage="learning": Dify produces reply, memory updates, reviews, working memory
        """
        if not self.api_key:
            return self._mock_response(inputs)

        url = f"{self.api_base}/workflows/run"
        payload = {
            "inputs": inputs,
            "response_mode": "blocking",
            "user": self.user,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Dify request failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Dify request failed: {exc.reason}") from exc

    def _mock_response(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Local mock output for running the backend without DIFY_API_KEY."""
        stage = inputs.get("stage")
        chat_text = inputs.get("chatText", "")

        if stage == "retrieval_query":
            return {
                "retrieval_query": chat_text,
            }

        return {
            "reply": {
                "should_reply": False,
                "content": "",
                "reason": "Current backend stage only verifies memory workflow.",
            },
            "memory_updates": [
                {
                    "memory_type": "conversation_style",
                    "content": "The user may react weakly to frequent follow-up questions.",
                    "confidence": 0.82,
                    "has_conflict": False,
                }
            ],
            "memory_reviews": [],
            "updated_working_memory": {
                "content": f"Latest chat context: {chat_text}",
                "confidence": 0.74,
            },
        }
