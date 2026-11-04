from llm.llm_provider import LLMProvider


class VLLMProvider(LLMProvider):

    def __init__(self, model, model_params=None, tensor_parallel_size=1, llm_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model
        self.model = model
        self.model_params = model_params or {}
        self.tensor_parallel_size = tensor_parallel_size
        self.llm_kwargs = llm_kwargs or {}
        self._llm = None

    def _generate(self, messages, parameters=None):
        if self._llm is None:
            from vllm import LLM
            self._llm = LLM(model=self.model_name, tensor_parallel_size=self.tensor_parallel_size, **self.llm_kwargs)

        from vllm import SamplingParams
        params = {**self.model_params, **(parameters or {})}
        max_tokens = params.pop("max_new_tokens", params.pop("max_tokens", 1024))
        temperature = params.pop("temperature", 0.7)
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, **params)

        # Drop trailing empty assistant turn
        msgs = [m for m in messages if not (m["role"] == "assistant" and not m.get("content"))]

        outputs = self._llm.chat(msgs, sampling_params=sampling_params)
        return outputs[0].outputs[0].text
