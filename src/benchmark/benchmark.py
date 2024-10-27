import pudb
from sampling.utils import norm_logits, sample
import torch
import time

def benchmark(prefix, target_model, default_max_tokens, temperature, top_k, top_p):
    seq_len = prefix.shape[1]
    target_len = seq_len + default_max_tokens
    
    start_time = time.time()
    
    while prefix.shape[1] < target_len:
        target_output = target_model(prefix).logits
        # for i in range(target_output.shape[1]):
        #     target_output[:, i] = norm_logits(target_output[:, i], temperature, top_k, top_p)
        next_token = sample(norm_logits(target_output[:, -1], temperature, top_k, top_p), 1)
        prefix = torch.cat([prefix, next_token], dim=1)
        
    end_time = time.time()
    print(f"=====================================BENCHMARK=====================================")
    print(f"Time taken: {end_time - start_time}, Tokens generated: {prefix.shape[1] - seq_len}, Tokens per second: {(prefix.shape[1] - seq_len) / (end_time - start_time)}\n\n\n")