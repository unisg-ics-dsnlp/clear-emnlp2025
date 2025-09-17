import asyncio
import enum
import os
from json import JSONDecodeError
from typing import List, Optional

import aiohttp
from openai import OpenAI
from transformers import AutoTokenizer


async def POST_request(
        url: str,
        session: aiohttp.ClientSession,
        headers: dict,
        payload: dict,
):
    async with session.post(url, json=payload, headers=headers) as response:
        try:
            resp = await response.json(content_type=None)
        except JSONDecodeError:
            return {'response': 'ERROR - JSONDecodeError'}
        return resp


class GenerationModel(enum.Enum):
    LLAMA = 'meta-llama/Llama-2-70b-chat-hf'
    LLAMA3 = 'meta-llama/Meta-Llama-3-70B-Instruct'
    MIXTRAL = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    GPT4 = 'gpt4'
    CLAUDE3 = 'claude3'


async def _make_request_selfhosted(
        prompts: List[str],
        max_tokens: Optional[int] = None,
        model: GenerationModel = GenerationModel.LLAMA,
        temperature: float = 0.0,
):
    if model == GenerationModel.LLAMA3:
        url = os.environ['LLAMA_OPENAI_ENDPOINT']
        model_name = 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'
    elif model == GenerationModel.MIXTRAL:
        url = os.environ['MIXTRAL_OPENAI_ENDPOINT']
        model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    else:
        raise NotImplementedError(f'Unknown model {model}')
    client = OpenAI(
        base_url=url,
        timeout=60 * 30,
        api_key=os.environ['LLAMA_API_KEY']
    )
    batch = client.completions.create(
        model=model_name,
        prompt=prompts,
        max_tokens=max_tokens if max_tokens is not None else 4096,
        temperature=temperature,
    )
    out = []
    for completion in batch.choices:
        out.append({'response': completion.text})
    return out


async def _make_request_external(
        prompts: List[str],
        url: str,
        max_tokens: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model: GenerationModel = GenerationModel.LLAMA
):
    connector = aiohttp.TCPConnector(limit=500)
    timeout = aiohttp.ClientTimeout(total=60 * 60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for prompt in prompts:
            if tokenizer is not None:
                prompt_length = len(tokenizer(prompt)['input_ids'])
                if prompt_length > max_tokens:
                    continue  # skip for now
            headers = {
                'Authorization': 'Bearer ' + os.environ['LLAMA_API_KEY']
            }
            payload = {
                'prompt': prompt,
            }
            if max_tokens is not None:
                payload['maxTokens'] = max_tokens

            if model == GenerationModel.GPT4:
                headers = {
                    'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'],
                    'Content-Type': 'application/json',
                }
                payload = {
                    'model': 'gpt-4-0613',
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt,
                        },
                    ],
                }
                if max_tokens is not None:
                    payload['max_tokens'] = max_tokens
            elif model == GenerationModel.CLAUDE3:
                headers = {
                    'x-api-key': os.environ['CLAUDE_API_KEY'],
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json',
                }
                payload = {
                    # TODO: this is the SMALLEST model!
                    'model': 'claude-3-haiku-20240307',
                    'max_tokens': 1024,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt,
                        },
                    ],
                }
            task = asyncio.create_task(POST_request(url, session, headers, payload))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
        return responses


def prompt_llm(
        prompts: List[str],
        max_tokens: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model: GenerationModel = GenerationModel.LLAMA,
        temperature: float = 0.0,
):
    if model == GenerationModel.LLAMA:
        return _make_request_selfhosted(prompts, max_tokens, model, temperature)
    elif model == GenerationModel.MIXTRAL:
        return _make_request_selfhosted(prompts, max_tokens, model, temperature)
    elif model == GenerationModel.LLAMA3:
        return _make_request_selfhosted(prompts, max_tokens, model, temperature)
    elif model == GenerationModel.GPT4:
        url = 'https://api.openai.com/v1/chat/completions'
        return _make_request_external(prompts, url, max_tokens, tokenizer, model)
    elif model == GenerationModel.CLAUDE3:
        url = 'https://api.anthropic.com/v1/messages'
        return _make_request_external(prompts, url, max_tokens, tokenizer, model)
    else:
        raise NotImplementedError(f'Unknown model {model}')
