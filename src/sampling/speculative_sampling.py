import torch
from sampling.utils import sample, norm_logits, norm_max
import pudb
import time

def speculative_sampling(prefix, approx_model, target_model, default_max_tokens, temperature, top_k, top_p, default_gamma, random_seed):
    seq_len = prefix.shape[1]
    target_len = seq_len + default_max_tokens
    
    start_time = time.time()
    
    while prefix.shape[1] < target_len:
        tmp_prefix = prefix
        prefix_len = tmp_prefix.shape[1]
        for _ in range(default_gamma):
            approx_output = approx_model(tmp_prefix).logits
            next_token = sample(norm_logits(approx_output[:, -1], temperature, top_k, top_p), 1)
            tmp_prefix = torch.cat([tmp_prefix, next_token], dim=1)
        
        for i in range(prefix_len - 1, approx_output.shape[1]):
            approx_output[:, i] = norm_logits(approx_output[:, i], temperature, top_k, top_p)
            
        target_output = target_model(tmp_prefix).logits
        for i in range(prefix_len - 1, target_output.shape[1]):
            target_output[:, i] = norm_logits(target_output[:, i], temperature, top_k, top_p)
            
        is_all_accept = True
        n = prefix_len - 1
        
        # pudb.set_trace()
        
        for i in range(default_gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            
            random_num = torch.rand(1, device=target_model.device)
            token_to_add = tmp_prefix[:, prefix_len + i]
            
            if random_num < torch.min(torch.tensor([1], device=target_model.device), target_output[:, prefix_len + i - 1, token_to_add] / approx_output[:, prefix_len + i - 1, token_to_add]):
                # If target_output's probability is higher than approx_output's probability, accept
                # else, accept with a probabilily
                n += 1
            else:
                # reject, generate next token from target model's output logits.
                is_all_accept = False
                t = sample(norm_max(target_output[:, n] - approx_output[:, n]), 1)
                break
            
        if is_all_accept:
            t = torch.cat([
                    sample(target_output[:, prefix_len - 1 + i].squeeze(), 1) 
                    for i in range(default_gamma)
                ])
            t = t.reshape(1, t.shape[0])
        
        prefix = torch.cat([prefix, t], dim=1)
        
    end_time = time.time()
    print(f"\n=====================================SPECULATIVE SAMPLING=====================================")
    print(f"Time taken: {end_time - start_time}, Tokens generated: {prefix.shape[1] - seq_len}, Tokens per second: {(prefix.shape[1] - seq_len) / (end_time - start_time)}\n")
        
    return prefix
                