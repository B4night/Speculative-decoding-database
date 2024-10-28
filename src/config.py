import torch

model_list = {
    'llama3.2-1b': '/ibex/user/feic/pjs/Speculative-decoding/Speculative-decoding-database/models/llama3.2-1b',
    'llama3-8b': '/ibex/user/feic/pjs/Speculative-decoding/Speculative-decoding-database/models/Meta-Llama-3-8B-Instruct',
    'llama3-70b': '/ibex/user/feic/pjs/Speculative-decoding/Speculative-decoding-database/models/Meta-Llama-3-70B-Instruct',
}

# random_seed = 42
# default_max_tokens = 100
# default_gamma = 4
# top_k = 20
# top_p = 0.9
# temperature = 1
is_benchmark_needed = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# random_seed = [42]
# default_max_tokens = [50, 100, 150]
# default_gamma = [4, 5, 6]
# top_k = [20, 30, 40]
# top_p = [0.9]
# temperature = [0.5, 1, 1.5]


random_seed = [42]
default_max_tokens = [100]
default_gamma = [4]
top_k = [20]
top_p = [0.9]
temperature = [0.5, 1.5]