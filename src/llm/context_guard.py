"""Refuses, before sending, a request that cannot fit the model's context window.

A vLLM server rejects a chat request when  prompt_tokens + max_tokens > max_model_len  (the rule is the server's, not
ours). Sending such a request only produces an error, so the guard counts the prompt tokens with the SERVER'S OWN
tokenizer and chat template (vLLM `/tokenize`, exact) and raises `ContextOverflow` without making a model call.

Nothing about a request that fits is changed: the same messages and the same sampling parameters (including
max_tokens) are forwarded untouched. A request that does not fit is never shortened or retried.

The exception is recognised by its `is_context_overflow` attribute rather than by class, because this module can be
imported under two package names (`llm.` and `src.llm.`) and two classes would then not match.
"""
import socket
import threading
import time
import urllib.error

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


class TokenCountError(Exception):
    """The /tokenize request itself failed. It is NOT a model-generation failure and never a model call.

    kind == "timeout": the count request timed out (retried a bounded number of times first).
    kind == "error":   any other failure of the count request (never retried, always a technical failure).
    Recognised by `is_token_count_error` for the same two-package-name reason as ContextOverflow.
    """
    is_token_count_error = True

    def __init__(self, kind, attempts, cause):
        self.kind, self.attempts = kind, attempts
        super().__init__(f"token_count_{kind}: /tokenize failed after {attempts} attempt(s): "
                         f"{type(cause).__name__}: {str(cause)[:200]} (request not sent)")
        self.__cause__ = cause


def is_token_count_error(error):
    return bool(getattr(error, "is_token_count_error", False))


def is_count_timeout(error):
    """True only for a timeout of the count request itself (a raw socket timeout or a URLError wrapping one)."""
    if isinstance(error, (TimeoutError, socket.timeout)):
        return True
    return isinstance(error, urllib.error.URLError) and isinstance(getattr(error, "reason", None), (TimeoutError, socket.timeout))


class ContextGuardProvider(LLMProvider):
    """Wraps `inner` (the provider that is called) and uses `counter.count_prompt_tokens(messages)` to count tokens.

    `counter.count_prompt_tokens` returns (prompt_tokens, server_max_model_len_or_None). The guard sits OUTSIDE the
    call recorder, so a refused request is neither a model call nor a failed call.
    """

    COUNT_ATTEMPTS = 3                 # total tries of one /tokenize request when it TIMES OUT (other errors: 1 try)
    COUNT_BACKOFF_SECONDS = (2.0, 5.0)  # waits before the 2nd and 3rd try

    def __init__(self, inner, counter, max_model_len, max_tokens, max_consecutive_count_errors=3,
                 count_attempts=None, sleep=time.sleep):
        super().__init__()
        if not isinstance(max_model_len, int) or isinstance(max_model_len, bool) or max_model_len < 1:
            raise ValueError("context guard: max_model_len must be a positive integer")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 0 < max_tokens < max_model_len:
            raise ValueError("context guard: the provider needs an integer max_tokens below max_model_len")
        self.wrapped, self.counter = inner, counter
        self.max_model_len, self.max_tokens = max_model_len, max_tokens
        self.model = getattr(inner, "model", None)
        self.stats = {"enabled": True, "max_model_len": max_model_len, "max_tokens": max_tokens,
                      "checked": 0, "overflow": 0, "largest_prompt_tokens": 0,
                      "count_retries": 0, "count_timeouts": 0, "count_errors": 0}
        self._count_attempts = max(1, int(count_attempts if count_attempts is not None else self.COUNT_ATTEMPTS))
        self._sleep = sleep
        self._count_errors = 0
        self._max_count_errors = max_consecutive_count_errors
        self._lock = threading.Lock()

    @property
    def inner(self):   # keeps the provider name in the metadata sidecars unchanged (OpenAIProvider, not the wrappers)
        return getattr(self.wrapped, "inner", self.wrapped)

    def _count(self, messages):
        """One logical count: only a timeout is retried (bounded); the model is never called from here."""
        last = None
        for attempt in range(1, self._count_attempts + 1):
            try:
                return self.counter.count_prompt_tokens(messages)
            except Exception as e:
                last = e
                if not is_count_timeout(e):
                    kind, used = "error", attempt
                    break
                if attempt < self._count_attempts:
                    with self._lock:
                        self.stats["count_retries"] += 1
                    wait = self.COUNT_BACKOFF_SECONDS[min(attempt - 1, len(self.COUNT_BACKOFF_SECONDS) - 1)]
                    self._sleep(wait)
                    continue
                kind, used = "timeout", attempt
        with self._lock:
            self.stats["count_timeouts" if kind == "timeout" else "count_errors"] += 1
            self._count_errors += 1
            if self._count_errors >= self._max_count_errors:
                raise ProviderFailing(f"{self._count_errors} token counts in a row failed - the context guard cannot "
                                      f"protect this run. Last error: {type(last).__name__}: {str(last)[:300]}")
        raise TokenCountError(kind, used, last)

    def _generate(self, messages, parameters=None):
        max_tokens = self.max_tokens if not parameters or "max_tokens" not in parameters else parameters["max_tokens"]
        tokens, server_limit = self._count(messages)
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
