import torch
from sampling.utils import sample, norm_logits, norm_max
import pudb

def speculative_sampling(prefix, approx_model, target_model, max_len, gamma, temperature, top_k, top_p, random_seed):
    seq_len = len(prefix)
    target_len = seq_len + max_len
    
    # gamma = 1
    # pudb.set_trace()
    
    while prefix.shape[1] < target_len:
        tmp_prefix = prefix
        prefix_len = tmp_prefix.shape[1]
        for _ in range(gamma):
            approx_output = approx_model(tmp_prefix).logits
            next_token = sample(norm_logits(approx_output[:, -1], temperature, top_k, top_p), 1)
            tmp_prefix = torch.cat([tmp_prefix, next_token], dim=1)
        
        for i in range(approx_output.shape[1]):
            approx_output[:, i] = norm_logits(approx_output[:, i], temperature, top_k, top_p)
            
        target_output = target_model(tmp_prefix).logits
        for i in range(target_output.shape[1]):
            target_output[:, i] = norm_logits(target_output[:, i], temperature, top_k, top_p)
            
        is_all_accept = True
        n = prefix_len - 1
        
        # pudb.set_trace()
        
        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            
            random_num = torch.rand(1, device=target_model.device)
            token_to_add = tmp_prefix[:, prefix_len + i]
            
            if random_num < torch.min(torch.tensor([1], device=target_model.device), target_output[:, prefix_len + i - 1, token_to_add] / approx_output[:, prefix_len + i - 1, token_to_add]):
                # accept
                n += 1
            else:
                # reject
                is_all_accept = False
                t = sample(norm_max(target_output[:, n] - approx_output[:, n]), 1)
                break
            
        if is_all_accept:
            t = sample(target_output[:, -1], 1)
        
        prefix = torch.cat([prefix, t], dim=1)
        
    return prefix
                