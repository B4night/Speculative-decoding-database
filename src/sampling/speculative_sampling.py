import torch
from sampling.utils import sample, norm_logits, norm_max
import pudb
import time
from config import *

torch.manual_seed(random_seed)

def speculative_sampling(prefix, approx_model, target_model, default_max_tokens, temperature, top_k, top_p, default_gamma, random_seed):
    # pudb.set_trace()
    seq_len = prefix.shape[1]
    target_len = seq_len + default_max_tokens
    
    start_time = time.time()
    
    approx_model_execute_time = 0
    target_model_execute_time = 0
    accept_num_each_iter = []
    
    while prefix.shape[1] < target_len:
        # clear cache
        approx_model.pass_key_values = None
        target_model.pass_key_values = None
        
        tmp_prefix = prefix
        prefix_len = tmp_prefix.shape[1]
        for _ in range(default_gamma):
            approx_output = approx_model(tmp_prefix, use_cache=True).logits
            next_token = sample(norm_logits(approx_output[:, -1], temperature, top_k, top_p), 1)
            tmp_prefix = torch.cat([tmp_prefix, next_token], dim=1)
            
            approx_model_execute_time += 1
        
        for i in range(prefix_len - 1, approx_output.shape[1]):
            approx_output[:, i] = norm_logits(approx_output[:, i], temperature, top_k, top_p)
            
        target_output = target_model(tmp_prefix, use_cache=True).logits
        target_model_execute_time += 1
        
        for i in range(prefix_len - 1, target_output.shape[1]):
            target_output[:, i] = norm_logits(target_output[:, i], temperature, top_k, top_p)
            
        is_all_accept = True
        n = prefix_len - 1
        
        # pudb.set_trace()
        
        for i in range(default_gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            
            random_num = torch.rand(1)
            token_to_add = tmp_prefix[:, prefix_len + i]
            
            if random_num < torch.min(torch.tensor([1]), target_output[:, prefix_len + i - 1, token_to_add] / approx_output[:, prefix_len + i - 1, token_to_add]):
                # If target_output's probability is higher than approx_output's probability, accept
                # else, accept with a probabilily
                n += 1
            else:
                # reject, generate next token from target model's output logits.
                is_all_accept = False
                tmp_t = sample(norm_max(target_output[:, n] - approx_output[:, n]), 1)
                if n == prefix_len - 1:
                    t = tmp_t
                else:
                    t = torch.cat([
                        sample(target_output[:, i].squeeze(), 1)
                        for i in range(prefix_len - 1, n)
                    ]).reshape(1, -1)
                    t = torch.cat([t, tmp_t], dim=1)
                break
            
        if is_all_accept:
            t = torch.cat([
                    sample(target_output[:, prefix_len - 1 + i].squeeze(), 1) 
                    for i in range(default_gamma + 1)
                ])
            
        t = t.reshape(1, -1)
        
        old_len = prefix.shape[1]
        prefix = torch.cat([prefix, t], dim=1)
        accept_num_each_iter.append(prefix.shape[1] - old_len - 1)
        
    end_time = time.time()
    print(f"Approx model execute time: {approx_model_execute_time}, Target model execute time: {target_model_execute_time}")
    print(f"Average acceptance rate: {sum(accept_num_each_iter) / len(accept_num_each_iter)}")
    print(f"Time taken: {end_time - start_time}, Tokens generated: {prefix.shape[1] - seq_len}, Tokens per second: {(prefix.shape[1] - seq_len) / (end_time - start_time)}\n\n\n", flush=True)
    
        
    return prefix, sum(accept_num_each_iter) / len(accept_num_each_iter), end_time - start_time
                