from typing import Optional

from vllm import EngineArgs, ModelRegistry


def is_supported_by_vllm(
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = 'auto',
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = 'auto',
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
):
    """
    Checks if the model is supported by vllm.
    Very hacky, likely not stable, and WILL break if vllm is updated.
    Look at the get_model_architecture in vllm, it's in vllm/model_executor/model_loader/utils.py.
    This method is the same as the one that is used there, except that it returns True/False instead of raising an Exception.
    """
    if model in [
        'microsoft/Phi-3-small-8k-instruct',
        'microsoft/Phi-3-medium-4k-instruct'
    ]:
        return False
    engine_args = EngineArgs(
        model=model,
        tokenizer=tokenizer,
        tokenizer_mode=tokenizer_mode,
        skip_tokenizer_init=skip_tokenizer_init,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        quantization=quantization,
        revision=revision,
        tokenizer_revision=tokenizer_revision,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=swap_space,
        enforce_eager=enforce_eager,
        max_context_len_to_capture=max_context_len_to_capture,
        max_seq_len_to_capture=max_seq_len_to_capture,
        disable_custom_all_reduce=disable_custom_all_reduce,
        **kwargs,
    )
    engine_config = engine_args.create_engine_config()
    model_config = engine_config.model_config
    architectures = getattr(model_config.hf_config, 'architectures', [])
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    if (model_config.quantization is not None
            and model_config.quantization != 'fp8'
            and 'MixtralForCausalLM' in architectures):
        architectures = ['QuantMixtralForCausalLM']

    for arch in architectures:
        # model_cls = ModelRegistry.load_model_cls(arch)
        model_cls = ModelRegistry._try_load_model_cls(arch)
        if model_cls is not None:
            return True
    else:
        return False


if __name__ == '__main__':
    print('google/flan-t5-base', is_supported_by_vllm('google/flan-t5-base'))
    print('microsoft/Phi-3-mini-4k-instruct', is_supported_by_vllm('microsoft/Phi-3-mini-4k-instruct'))
