"""Refuses, before sending, a request that cannot fit the model's context window.

A vLLM server rejects a chat request when  prompt_tokens + max_tokens > max_model_len  (the rule is the server's, not
ours). Sending such a request only produces an error, so the guard counts the prompt tokens with the SERVER'S OWN
tokenizer and chat template (vLLM `/tokenize`, exact) and raises `ContextOverflow` without making a model call.

Nothing about a request that fits is changed: the same messages and the same sampling parameters (including
max_tokens) are forwarded untouched. A request that does not fit is never shortened or retried.

The exception is recognised by its `is_context_overflow` attribute rather than by class, because this module can be
imported under two package names (`llm.` and `src.llm.`) and two classes would then not match.
"""
import threading

from .llm_provider import LLMProvider
from .recording_provider import ProviderFailing


class ContextOverflow(Exception):
    is_context_overflow = True

    def __init__(self, prompt_tokens, max_tokens, max_model_len, n_messages):
        self.prompt_tokens, self.max_tokens = prompt_tokens, max_tokens
        self.max_model_len, self.n_messages = max_model_len, n_messages
        super().__init__(f"context_overflow: prompt {prompt_tokens} + max_tokens {max_tokens} > "
                         f"max_model_len {max_model_len} (request not sent)")


def is_context_overflow(error):
    return bool(getattr(error, "is_context_overflow", False))


class ContextGuardProvider(LLMProvider):
    """Wraps `inner` (the provider that is called) and uses `counter.count_prompt_tokens(messages)` to count tokens.

    `counter.count_prompt_tokens` returns (prompt_tokens, server_max_model_len_or_None). The guard sits OUTSIDE the
    call recorder, so a refused request is neither a model call nor a failed call.
    """

    def __init__(self, inner, counter, max_model_len, max_tokens, max_consecutive_count_errors=3):
        super().__init__()
        if not isinstance(max_model_len, int) or isinstance(max_model_len, bool) or max_model_len < 1:
            raise ValueError("context guard: max_model_len must be a positive integer")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 0 < max_tokens < max_model_len:
            raise ValueError("context guard: the provider needs an integer max_tokens below max_model_len")
        self.wrapped, self.counter = inner, counter
        self.max_model_len, self.max_tokens = max_model_len, max_tokens
        self.model = getattr(inner, "model", None)
        self.stats = {"enabled": True, "max_model_len": max_model_len, "max_tokens": max_tokens,
                      "checked": 0, "overflow": 0, "largest_prompt_tokens": 0}
        self._count_errors = 0
        self._max_count_errors = max_consecutive_count_errors
        self._lock = threading.Lock()

    @property
    def inner(self):   # keeps the provider name in the metadata sidecars unchanged (OpenAIProvider, not the wrappers)
        return getattr(self.wrapped, "inner", self.wrapped)

    def _generate(self, messages, parameters=None):
        max_tokens = self.max_tokens if not parameters or "max_tokens" not in parameters else parameters["max_tokens"]
        try:
            tokens, server_limit = self.counter.count_prompt_tokens(messages)
        except Exception as e:
            with self._lock:
                self._count_errors += 1
                if self._count_errors >= self._max_count_errors:
                    raise ProviderFailing(f"{self._count_errors} token counts in a row failed - the context guard cannot "
                                          f"protect this run. Last error: {str(e)[:300]}")
            raise
        with self._lock:
            self._count_errors = 0
            self.stats["checked"] += 1
            self.stats["largest_prompt_tokens"] = max(self.stats["largest_prompt_tokens"], tokens)
        if server_limit is not None and server_limit != self.max_model_len:
            raise ProviderFailing(f"server max_model_len is {server_limit} but the guard was configured for "
                                  f"{self.max_model_len}; refusing to run with an unverified window")
        if tokens + max_tokens > self.max_model_len:
            with self._lock:
                self.stats["overflow"] += 1
            raise ContextOverflow(tokens, max_tokens, self.max_model_len, len(messages))
        response, _ = self.wrapped.generate(messages, parameters)
        return response
