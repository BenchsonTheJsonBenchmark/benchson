from typing import Any, Dict, List
from src.llm.llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key, model="gpt-4", base_url=None, extra_headers=None, model_params=None,
                 drop_empty_assistant_turn=False, timeout=None, max_retries=None, **kwargs):
        super().__init__(**kwargs)  # must call it
        self.install_dependency("openai")  # Ensure the package is installed
        import openai

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers
        if timeout is not None:          # seconds; a hung local server should not block a worker for the 10-minute default
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        self.client = openai.OpenAI(**client_kwargs)
        self.model = model
        self.model_params = model_params or {}
        # Local servers (vLLM) reject/continue a trailing empty assistant turn; off by default so
        # the behaviour for OpenAI itself is unchanged.
        self.drop_empty_assistant_turn = drop_empty_assistant_turn

    def _generate(
        self, messages: List[Dict[str, str]], parameters: Dict[str, Any] = None
    ) -> str:
        params = {**self.model_params, **(parameters or {})}
        if self.drop_empty_assistant_turn:
            messages = [m for m in messages if not (m["role"] == "assistant" and not m.get("content"))]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **params
        )
        # A server can answer with content=None (e.g. nothing generated); hand back an empty string so the
        # builder records an 'empty_response' event instead of crashing on None.
        return response.choices[0].message.content or ""

    def count_prompt_tokens(self, messages):
        """Exact prompt length from the vLLM server's own tokenizer and chat template (POST <root>/tokenize).

        Returns (prompt_tokens, max_model_len reported by the server). Only vLLM-style servers have this endpoint.
        Plain standard library, so it needs no extra package; a loopback server is reached without any proxy.
        """
        import json
        import urllib.parse
        import urllib.request

        if self.drop_empty_assistant_turn:   # the same filtering _generate applies
            messages = [m for m in messages if not (m["role"] == "assistant" and not m.get("content"))]
        root = str(self.client.base_url).rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        body = json.dumps({"model": self.model, "messages": messages, "add_generation_prompt": True}).encode()
        request = urllib.request.Request(root + "/tokenize", data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.client.api_key}"})
        loopback = urllib.parse.urlparse(root).hostname in ("127.0.0.1", "localhost", "::1")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if loopback else urllib.request.build_opener()
        with opener.open(request, timeout=30) as response:
            answer = json.loads(response.read().decode("utf-8"))
        return int(answer["count"]), answer.get("max_model_len")
