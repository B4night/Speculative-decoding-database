# Autoregressive sampling
import torch
from sampling.utils import norm_logits, sample

def autoregressive_sampling(token_ids, model, token_num, temperature, top_k, top_p):
    current_len = token_ids.size(1)
    target_len = current_len + token_num
    
    past_key_values = None
    while current_len < target_len:
        if past_key_values is None:
            output = model(token_ids)
        else:
            last_token_id = token_ids[:, -1].unsqueeze(-1)
            output = model(last_token_id, past_key_values=past_key_values, use_cache=True)
        
        last_pred_logits = norm_logits(output.logits[:, -1], temperature, top_k, top_p)
        past_key_values = output.past_key_values
        next_token_id = sample(last_pred_logits, 1)
        token_ids = torch.cat([token_ids, next_token_id], dim=1)
        current_len += 1
        
    return token_ids
