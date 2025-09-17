import torch
from vllm import LLM

model_instance = LLM(
    model='tiiuae/falcon-40b-instruct',
    dtype='half',
    tensor_parallel_size=torch.cuda.device_count(),
)

prompt = 'How many rs are in Strawberry?'
out = model_instance.generate(prompt)
print('foo')
