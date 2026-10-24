"""Wraps any LLMProvider to log every call and to enforce a call budget.

Used by preview runs. The log keeps exactly what the model returned (raw, before any
cleaning or parsing), and the budget stops a run that would otherwise spend too many
paid calls.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone

from .llm_provider import LLMProvider

_WRITE_LOCK = threading.Lock()  # one lock for all wrappers writing to the same log


class CallBudgetExceeded(BaseException):
    """Stops the whole run once the call budget is used up.

    Deliberately a BaseException (like KeyboardInterrupt): the builder catches
    `Exception` per schema and carries on, which would turn a budget stop into
    hundreds of skipped schemas instead of one clean stop.
    """


class ProviderFailing(BaseException):
    """Raised when several model calls in a row failed.

    That is almost always a setup problem (credentials, project, model id), which
    retrying cannot fix. Stopping at once is clearer and cheaper than letting the
    builder skip schema after schema. A BaseException for the same reason as above.
    """


class CallBudget:
    """A shared counter, so the builder and validator models draw from one budget.

    It also counts consecutive failures across both models.
    """

    def __init__(self, max_calls=None, max_consecutive_errors=3):
        self.max_calls = max_calls
        self.max_consecutive_errors = max_consecutive_errors
        self.used = 0
        self.consecutive_errors = 0
        self._lock = threading.Lock()

    def succeeded(self):
        with self._lock:
            self.consecutive_errors = 0

    def failed(self, error):
        with self._lock:
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                raise ProviderFailing(
                    f"{self.consecutive_errors} model calls in a row failed - stopping instead of "
                    f"retrying. Last error: {str(error)[:500]}")

    def take(self):
        with self._lock:
            if self.max_calls is not None and self.used >= self.max_calls:
                raise CallBudgetExceeded(
                    f"LLM call budget of {self.max_calls} calls was used up")
            self.used += 1
            return self.used


class RecordingProvider(LLMProvider):

    def __init__(self, inner, log_path, label="builder", budget=None):
        super().__init__()
        self.inner = inner
        self.log_path = log_path
        self.label = label
        self.budget = budget or CallBudget(None)
        self.model = getattr(inner, "model", None)
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def _generate(self, messages, parameters=None):
        call_no = self.budget.take()
        started = time.time()
        response, error = None, None
        try:
            response, _ = self.inner.generate(messages, parameters)
            self.budget.succeeded()
            return response
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            self.budget.failed(error)
            raise
        finally:
            record = {
                "call": call_no,
                "label": self.label,
                "model": self.model,
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seconds": round(time.time() - started, 2),
                "messages": messages,
                "response": response,
                "error": error,
            }
            with _WRITE_LOCK:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
