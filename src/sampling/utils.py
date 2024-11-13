import torch
from config import *
from torch.nn import functional as F

torch.manual_seed(random_seed)

def top_k_filtering(logits, top_k):
    if top_k > 0:
        # 获取 top_k 的值
        filter_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        # 设置低于 top_k 的 logit 为 -inf
        logits[logits < filter_vals[:, [-1]]] = float('-inf')
        
        # 检查是否所有 logits 都被过滤为 -inf
        if torch.all(logits == float('-inf')):
            raise ValueError("所有 logits 都被 top_k_filtering 过滤为 -inf，请检查 top_k 的值")
        
    return logits

def top_p_filtering(logits, top_p):
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 确定需要移除的索引
        sorted_indices_to_remove = cumulative_probs > top_p
        # 保证至少保留一个 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 创建一个与 logits 形状相同的 mask
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        # 检查是否所有 logits 都被过滤为 -inf
        if torch.all(logits == float('-inf')):
            raise ValueError("所有 logits 都被 top_p_filtering 过滤为 -inf，请检查 top_p 的值")
        
    return logits


def top_k_top_p_filtering(logits, top_k, top_p):
    logits = top_k_filtering(logits, top_k)
    logits = top_p_filtering(logits, top_p)
    return logits


def norm_logits(logits, temperature, top_k, top_p):
    assert logits.dim() == 2, "logits 必须是二维张量"
    
    # 防止 temperature 为零或负值
    if temperature <= 0:
        raise ValueError("temperature 必须为正数")
    
    logits = logits / temperature
    
    logits = top_k_top_p_filtering(logits, top_k, top_p)
    
    # 检查是否所有 logits 都是 -inf
    if torch.all(logits == float('-inf')):
        raise ValueError("所有 logits 都被 top_k_top_p_filtering 过滤为 -inf，请检查 top_k 和 top_p 的值")
    
    probs = F.softmax(logits, dim=-1)
    
    # 检查 softmax 结果是否包含 NaN 或 Inf
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        raise ValueError("softmax 结果包含 NaN 或 Inf")
    
    # 检查概率是否有负值
    if (probs < 0).any():
        raise ValueError("softmax 结果包含负值")
    
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
    # 检查 probs 是否包含 NaN 或 Inf
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        print(f'sample error NaN or Inf: probs={probs}')
        # replace NaN or Inf with 0
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        # raise ValueError("probabilities 包含 NaN 或 Inf")
    
    # 检查 probs 是否全为零
    if torch.all(probs == 0):
        print(f'sample error all zero: probs={probs}')
        probs = torch.ones_like(probs) / probs.size(-1)
        # raise ValueError("probabilities 全为零，无法进行采样")
    
    next_token_id = torch.multinomial(probs, num_samples)
    
    # 检查采样结果是否在有效范围内
    if (next_token_id >= probs.size(-1)).any() or (next_token_id < 0).any():
        raise ValueError(f"sampled_indices 包含无效值: {next_token_id}")
    
    return next_token_id


def norm_max(input):
    # 使用与输入相同的设备和数据类型
    device = input.device
    dtype = input.dtype
    
    # 将所有小于等于0的值设为0
    token_max = torch.where(input > 0, input, torch.zeros_like(input))
    
    # 计算 token_max 的和，并防止除以零
    token_max_sum = torch.sum(token_max, dim=1, keepdim=True) + 1e-10  # 添加 epsilon 以防除以零
    
    normalized = token_max / token_max_sum
    
    # 检查 normalized 是否包含 NaN 或 Inf
    if torch.isnan(normalized).any() or torch.isinf(normalized).any():
        print(f'norm_max error NaN or Inf: normalized={normalized}')
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        # raise ValueError("normalized 包含 NaN 或 Inf")
    
    return normalized
