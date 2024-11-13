import torch

model_list = {
    'llama3.2-1b': '/home/feic/pjs/Speculative-decoding-database/models/Llama-3.2-1B',
    'llama3-8b': '/home/feic/pjs/Speculative-decoding-database/models/Meta-Llama-3-8B-Instruct',
    'llama3-70b': '/home/feic/pjs/Speculative-decoding-database/models/Meta-Llama-3-70B-Instruct',
    'llama-68m': '/home/feic/pjs/Speculative-decoding-database/models/llama-68m',
    'llama2-70b': '/home/feic/pjs/Speculative-decoding-database/models/Llama-2-70b-chat-hf',
}

random_seed = 42
default_max_tokens = 128
default_gamma = 4
top_k = 20
top_p = 0.9
temperature = 0.8
threshold = 0.0
is_benchmark_needed = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# random_seed = [42]
# default_max_tokens = [10, 100, 150]
# default_gamma = [3]
# top_k = [20, 30, 40]
# top_p = [0.9]
# temperature = [0.5, 1, 1.5]


# random_seed = [42]
# default_max_tokens = [100]
# default_gamma = [4]
# top_k = [20]
# top_p = [0.9]
# temperature = [0.5, 1.5]