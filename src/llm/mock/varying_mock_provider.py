"""Offline test aid: like MockProvider, but every schema-creation answer differs from the previous one.

MockProvider is deterministic, so a second attempt on the same schema repeats the first reference and is rejected as a
duplicate. The chunk tests need several DIFFERENT references per schema (to exercise the per-schema cap), so this
provider overwrites integer fields with a call counter. Never use it to produce real data.
"""
import itertools
import json

from .mock_provider import MockProvider, _SCHEMA_MARKER


class VaryingMockProvider(MockProvider):
    _counter = itertools.count(1)

    def _generate(self, messages, parameters=None):
        text = super()._generate(messages, parameters)
        if any(m["role"] == "user" and _SCHEMA_MARKER in m["content"] for m in messages) and \
                not any("The JSON is invalid" in (m.get("content") or "") for m in messages):
            try:
                obj = json.loads(text)
            except ValueError:
                return text
            if isinstance(obj, dict):
                for key, value in list(obj.items()):
                    if isinstance(value, int) and not isinstance(value, bool):
                        obj[key] = next(self._counter) + 1000
                return json.dumps(obj, indent=2)
        return text
