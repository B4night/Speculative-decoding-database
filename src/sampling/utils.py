import torch
from torch.nn import functional as F

def top_k_filtering(logits, top_k):
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.shape[1]))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    return logits

def top_p_filtering(logits, top_p):
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    return logits

def top_k_top_p_filtering(logits, top_k, top_p):
    logits = top_k_filtering(logits, top_k)
    logits = top_p_filtering(logits, top_p)
    return logits

def norm_logits(logits, temperature, top_k, top_p):
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k, top_p)
    probs = F.softmax(logits, dim=-1)
    return probs
    
def sample(probs, num_samples):
    '''
    This function samples next token ids from the given probability distribution.
    
    Args:
        - probs (torch.Tensor): The probability distribution of the next token.
        - num_samples (int): The number of samples.
    
    Returns:
        - next_token_id (torch.Tensor): The sampled next token ids.
    '''
    next_token_id = torch.multinomial(probs, num_samples)
    return next_token_id

def norm_max(input):
    token_max = torch.where(input > 0, input, torch.tensor(0.0, device=input.device))
    token_max_sum = torch.sum(token_max, dim=1, keepdim=True)
    return token_max / token_max_sum