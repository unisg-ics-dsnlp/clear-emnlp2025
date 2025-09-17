from typing import Optional, Union

from vllm import SamplingParams

from util.template import PromptTemplate


class DirectWContextCheckPrompt(PromptTemplate):
    def __init__(
            self,
            model: str,
            temperature: float = 0.7,
            *args,
            **kwargs
    ):
        super().__init__(model, temperature=temperature, *args, **kwargs)

    def do_prompting(
            self,
            prompt: Union[str, list[str]],
            *args,
            **kwargs
    ) -> str | list[str] | tuple[list[str], list[bool]]:
        return self.send_prompt_to_model(prompt)

    def send_prompt_to_model(
            self,
            prompt: Union[str, list[str]]
    ) -> tuple[list[str], list[bool]]:
        if self.is_vllm:
            return self._prompt_vllm(prompt)
        else:
            raise NotImplementedError(f'This model is not supported yet.')

    def _prompt_vllm(
            self,
            prompt: Union[str, list[str]]
    ) -> tuple[list[str], list[bool]]:
        if isinstance(prompt, str):
            prompt = [prompt]

        model_config = self.model_instance.llm_engine.get_model_config()
        context_size = model_config.max_model_len
        tokenizer = self.model_instance.get_tokenizer()
        tokenized_prompts = tokenizer(prompt)
        fits_in_context = [len(tp) <= context_size for tp in tokenized_prompts.data['input_ids']]

        stop = self.kwargs.get('stop')
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            min_tokens=1,
            stop=stop,
            temperature=self.temperature
        )
        out = self.model_instance.generate(prompt, sampling_params)
        return [o.outputs[0].text for o in out], fits_in_context
