import abc
import asyncio
import os
from typing import Optional, Union

import anthropic
import requests
import torch
from langchain_openai import ChatOpenAI
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from util.api_stuff import prompt_llm, GenerationModel


def phi_format_func(
        system_prompt,
        user_prompt,
        assistant_prompt,
        additional_assistant_prompts,
        additional_user_prompts
):
    template = f'''<|system|>{system_prompt}<|end|>\n<|user|>{user_prompt}<|end|>\n<|assistant|>{assistant_prompt}'''
    if len(additional_user_prompts) > len(additional_assistant_prompts):
        additional_assistant_prompts.append('')
    # alternate between user and assistant prompts
    for user, assistant in zip(additional_user_prompts, additional_assistant_prompts):
        template += f'''<|end|><|user|>{user}<|end|>\n<|assistant|>{assistant}\n'''
    return template


def gemma_format_func(
        system_prompt,
        user_prompt,
        assistant_prompt,
        additional_assistant_prompts,
        additional_user_prompts
):
    template = f'''<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>system\n{assistant_prompt}'''
    if len(additional_user_prompts) > len(additional_assistant_prompts):
        additional_assistant_prompts.append('')
    # alternate between user and assistant prompts
    for user, assistant in zip(additional_user_prompts, additional_assistant_prompts):
        template += f'''<end_of_turn>\n<start_of_turn>user\n{user}\n<end_of_turn>\n<start_of_turn>system\n{assistant}'''
    return template


def gemma2_format_func(
        system_prompt,
        user_prompt,
        assistant_prompt,
        additional_assistant_prompts,
        additional_user_prompts
):
    template = f'''<start_of_turn>model\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{assistant_prompt}'''
    if len(additional_user_prompts) > len(additional_assistant_prompts):
        additional_assistant_prompts.append('')
    # alternate between user and assistant prompts
    for user, assistant in zip(additional_user_prompts, additional_assistant_prompts):
        template += f'''<end_of_turn>\n<start_of_turn>user\n{user}\n<end_of_turn>\n<start_of_turn>model\n{assistant}'''
    return template


def llama3_format_func(
        system_prompt: str,
        user_prompt: str,
        assistant_prompt: str,
        additional_assistant_prompts: Optional[list[str]] = None,
        additional_user_prompts: Optional[list[str]] = None
) -> str:
    template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|><start_header_id|>assistant<|end_header_id|>{assistant_prompt}'''
    template = template.replace('{system_prompt}', system_prompt).replace('{user_prompt}', user_prompt).replace(
        '{assistant_prompt}', assistant_prompt)
    if len(additional_user_prompts) > len(additional_assistant_prompts):
        additional_assistant_prompts.append('')
    # alternate between user and assistant prompts
    for user, assistant in zip(additional_user_prompts, additional_assistant_prompts):
        template += '''<|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{assistant_prompt}'''
        template = template.replace('{user_prompt}', user).replace('{assistant_prompt}', assistant)
    return template


def mixtral_format_func(
        system_prompt: str,
        user_prompt: str,
        assistant_prompt: str,
        additional_assistant_prompts: Optional[list[str]] = None,
        additional_user_prompts: Optional[list[str]] = None
):
    template = '''<s> [INST] {system} [/INST] {assistant_prompt}</s>'''
    template = template.replace('{system}', system_prompt + user_prompt).replace('{assistant_prompt}', assistant_prompt)
    if len(additional_user_prompts) > len(additional_assistant_prompts):
        additional_assistant_prompts.append('')
    # alternate between user and assistant prompts
    for user, assistant in zip(additional_user_prompts, additional_assistant_prompts):
        template += ''' [INST] {system} [/INST] {assistant_prompt}'''
        template = template.replace('{system}', user).replace('{assistant_prompt}', assistant)
    return template


class PromptTemplate(abc.ABC):
    def __init__(
            self,
            model: str,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            use_vllm: bool = False,
            vllm_model_instance: Optional[LLM] = None,
            *args,
            **kwargs
    ):
        self.is_huggingface = False
        self.is_vllm = False
        if model not in [
            'llama',
            'llama3',
            'gpt-4',
            'gpt-4o-2024-05-13',
            'mixtral',
            'gemma',
            'claude3',
            'gpt-3.5-turbo-0125',
            'davinci-002'
        ]:
            force_vllm_preload = bool(os.environ.get('FORCE_VLLM_PRELOAD', False))
            print('> PromptTemplate: FORCE_VLLM_PRELOAD value: ', force_vllm_preload)
            if vllm_model_instance is not None:
                print('> PromptTemplate: Initializing with pre-loaded vllm model instance')
                self.model_instance = vllm_model_instance
                self.is_vllm = True
            else:
                if force_vllm_preload:
                    raise Exception('FORCE_VLLM_PRELOAD is set, but no vllm_model_instance is provided!')
                if use_vllm:
                    model_instance = LLM(
                        model=model,
                        dtype='half',
                        tensor_parallel_size=torch.cuda.device_count(),
                        # trust_remote_code=True
                    )
                    print(f'> Model supported by vllm: {model}, using {torch.cuda.device_count()} GPUs')
                    self.model_instance = model_instance
                    self.is_vllm = True
                else:
                    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    print('> use_vllm is False, trying to load model with HuggingFace')
                    try:
                        model_instance = AutoModelForSeq2SeqLM.from_pretrained(
                            model,
                            device_map='auto',
                            token=os.environ['HUGGINGFACE_ACCESS_TOKEN'],
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True
                        )
                    except OSError:
                        raise NotImplementedError(f'Unknown HuggingFace model: {model}')
                    except ValueError:
                        print("> Model can't be loaded as AutoModelForSeq2SeqLM, trying AutoModelForCausalLM")
                        try:
                            model_instance = AutoModelForCausalLM.from_pretrained(
                                model,
                                device_map='auto',
                                token=os.environ['HUGGINGFACE_ACCESS_TOKEN'],
                                torch_dtype=torch.bfloat16,
                                trust_remote_code=True
                            )
                        except ValueError:
                            raise NotImplementedError(
                                f"HuggingFace model can't be loaded as AutoModelForSeq2SeqLm OR AutoModelForCausalLM: {model}"
                            )
                    self.is_huggingface = True
                    self.model_instance = model_instance
                    self.tokenizer_instance = AutoTokenizer.from_pretrained(
                        model,
                        token=os.environ['HUGGINGFACE_ACCESS_TOKEN'],
                        trust_remote_code=True
                    )
        if model in [
            'gpt-4',
            'gpt-3.5-turbo-0125',
            'gpt-4o-2024-05-13'
        ]:
            self.openai_client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY']
            )
        else:
            self.openai_client = None
        if model == 'claude3':
            self.claude_client = anthropic.Anthropic(
                api_key=os.environ['CLAUDE_API_KEY'],
            )
        else:
            self.claude_client = None
        if model == 'gemma':
            if not os.environ.get('DISABLE_GEMMA_WARNING', False):
                print(
                    f'> WARNING: Gemma must be running! It is not always on, so make sure it is running before using it.')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.args = args
        self.kwargs = kwargs
        self.format_func = None
        if 'format_func' in kwargs:
            self.format_func = kwargs['format_func']

        print(f'Initialized PromptTemplate with model {model}, use_vllm is {use_vllm}, is_vllm is {self.is_vllm}')

        # print(f'Initialized {self.__class__.__name__} with model {model}')

    def format_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            assistant_prompt: Optional[str] = '',
            additional_assistant_prompts: Optional[list[str]] = None,
            additional_user_prompts: Optional[list[str]] = None,
    ) -> str:
        """
        Format the prompt for the specific model.
        :param system_prompt: The system prompt.
        :param user_prompt: The user prompt.
        :param assistant_prompt: The assistant prompt.
        :param additional_assistant_prompts: Additional assistant prompts. If set, alternates between user and assistant.
        :param additional_user_prompts: Additional user prompts. If set, alternates between user and assistant.
        :return: The formatted prompt, formatted to the specific model.
        """
        if additional_assistant_prompts is None:
            additional_assistant_prompts = []
        if additional_user_prompts is None:
            additional_user_prompts = []
        if self.model == 'llama3':
            return llama3_format_func(
                system_prompt,
                user_prompt,
                assistant_prompt,
                additional_assistant_prompts,
                additional_user_prompts
            )
        elif self.model == 'mixtral':
            return mixtral_format_func(
                system_prompt,
                user_prompt,
                assistant_prompt,
                additional_assistant_prompts,
                additional_user_prompts
            )
        else:
            if self.format_func is None:
                # print('> WARNING: No formatting for this model! Returning unformatted prompt.')
                newline = '\n'
                return f'''{system_prompt}\n\n{user_prompt}\n\n{assistant_prompt}\n\n{newline.join(additional_assistant_prompts)}\n\n{newline.join(additional_user_prompts)}'''.strip()
            else:
                return self.format_func(
                    system_prompt,
                    user_prompt,
                    assistant_prompt,
                    additional_assistant_prompts,
                    additional_user_prompts
                )

    @abc.abstractmethod
    def do_prompting(
            self,
            prompt: str,
            demonstrations: Optional[list[str]] = None,
            *args,
            **kwargs
    ) -> str:
        pass

    def send_prompt_to_model(
            self,
            prompt: Union[str, list[str]]
    ) -> Union[str, list[str]]:
        # TODO: implement batching for others models, llama3 is supported
        if self.model == 'llama':
            return self._prompt_llama(prompt)
        elif self.model == 'llama3':
            if isinstance(prompt, str):
                prompt = [prompt]
            return self._prompt_llama3(prompt)
        elif self.model == 'gpt-4':
            return self._prompt_gpt4(prompt)
        elif self.model == 'gpt-3.5-turbo-0125':
            return self._prompt_gpt35(prompt)
        elif self.model == 'gpt-4o-2024-05-13':
            return self._prompt_gpt4o(prompt)
        elif self.model == 'mixtral':
            if isinstance(prompt, str):
                prompt = [prompt]
            return self._prompt_mixtral(prompt)
        elif self.model == 'gemma':
            return self._prompt_gemma(prompt)
        elif self.model == 'claude3':
            return self._prompt_claude3(prompt)
        elif self.model == 'davinci-002':
            return self._prompt_davinci002(prompt)
        elif self.is_vllm:
            return self._prompt_vllm(prompt)
        elif self.is_huggingface:
            return self._prompt_huggingface(prompt)
        else:
            raise NotImplementedError(f'Unknown model {self.model}')

    def _prompt_llama(
            self,
            prompt: str
    ) -> str:
        llm = ChatOpenAI(
            model_name='meta-llama/Llama-2-70b-chat-hf',
            openai_api_base='>SET LLAMA 2 API ENDPOINT HERE<',
            openai_api_key=os.environ['LLAMA_API_KEY'],
            temperature=self.temperature
        )
        out = llm.invoke(prompt)
        return out.content

    def _prompt_llama3(
            self,
            prompts: list[str]
    ) -> list[str]:
        max_tokens = self.max_tokens if self.max_tokens is not None else 2000
        out = asyncio.run(prompt_llm(
            prompts,
            model=GenerationModel.LLAMA3,
            max_tokens=max_tokens,
            temperature=self.temperature
        ))
        responses = []
        for o in out:
            responses.append(o['response'])
        return responses

    def _prompt_gpt4(
            self,
            prompt: str
    ) -> str:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )
        return completion.choices[0].message.content

    def _prompt_gpt4o(
            self,
            prompt: str
    ) -> str:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )
        return completion.choices[0].message.content

    def _prompt_gpt35(
            self,
            prompt: str
    ) -> str:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )
        return completion.choices[0].message.content

    def _prompt_davinci002(
            self,
            prompt: str
    ) -> str:
        api_key = os.environ['OPENAI_API_KEY']
        url = 'https://api.openai.com/v1/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
        data = {
            'prompt': prompt,
            'max_tokens': self.max_tokens if self.max_tokens is not None else 500,
            'model': 'davinci-002'
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['text']

    def _prompt_mixtral(
            self,
            prompts: list[str]
    ) -> list[str]:
        max_tokens = self.max_tokens if self.max_tokens is not None else 2000
        out = asyncio.run(prompt_llm(
            prompts,
            model=GenerationModel.MIXTRAL,
            max_tokens=max_tokens
        ))
        return [o['response'] for o in out]

    def _prompt_gemma(
            self,
            prompt: str
    ) -> str:
        out = asyncio.run(
            prompt_llm(
                [prompt],
                model=GenerationModel.MIXTRAL,  # this is INTENTIONAL - uses same endpoint as mixtral
                max_tokens=self.max_tokens if self.max_tokens is not None else 2000
            )
        )
        return out[0]['response']

    def _prompt_vllm(
            self,
            prompt: Union[str, list[str]]
    ) -> list[str]:
        if isinstance(prompt, str):
            prompt = [prompt]
        stop = self.kwargs.get('stop')
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            stop=stop,
            temperature=self.temperature
        )
        out = self.model_instance.generate(prompt, sampling_params)
        return [o.outputs[0].text for o in out]

    def _prompt_huggingface(
            self,
            prompt: Union[str, list[str]],
    ) -> str:
        max_new_tokens = self.kwargs.get('max_new_tokens', 2000)
        # batch_size = self.kwargs.get('batch_size', 1)
        if isinstance(prompt, str):
            prompt = [prompt]
        outs = []
        for p in tqdm(prompt, '> prompt_huggingface - generating'):
            input_ids = self.tokenizer_instance(
                p,
                return_tensors='pt',
            ).input_ids.to(self.model_instance.device)
            output = self.model_instance.generate(input_ids, max_new_tokens=max_new_tokens)
            tmp = self.tokenizer_instance.batch_decode(output, skip_special_tokens=True)
            outs.extend(tmp)
        return outs

    def _prompt_claude3(
            self,
            prompt: str
    ) -> str:
        message = self.claude_client.messages.create(
            model='claude-3-opus-20240229',
            max_tokens=self.kwargs.get('max_new_tokens', 1000),
            temperature=self.temperature,
            system='',  # intentionally blank
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        message_text = message.content[0].text
        return message_text
