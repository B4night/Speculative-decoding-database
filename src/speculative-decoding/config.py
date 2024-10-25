import torch

model_list = {
    'llama3.2-1b': '/ibex/user/feic/pjs/Speculative-decoding/Speculative-decoding-database/models/llama3.2-1b',
    'llama3-8b': '/ibex/user/feic/pjs/Speculative-decoding/Speculative-decoding-database/models/Meta-Llama-3-8B-Instruct',
}

default_max_tokens = 20
default_gamma = 4
top_k = 20
top_p = 0.9
temperature = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
