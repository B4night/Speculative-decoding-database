import torch
import time
from sampling.utils import norm_logits

def validate(model, prefix, db_answer, temperature, top_k, top_p):
    tmp_prefix = torch.concat([prefix, db_answer[:, 1:]], dim=1)
    
    start_time = time.time()
    
    output = model(tmp_prefix).logits
    
    token_cnt = output.shape[1]
    prob_sum = 0
    
    # Skip the first token, that's why using 'prefix.shape[1] + 1'
    for i in range(prefix.shape[1] + 1, token_cnt):
        prob = norm_logits(output[:, i - 1], temperature, top_k, top_p)
        # prob represents next token of prefix: tmp_prefix[:, :i]
        prob_sum += prob[0, tmp_prefix[0, i]]
        
    end_time = time.time()
    print(f"\n=========================================VALIDATION DATABASE=========================================")
    print(f"Time taken: {end_time - start_time}, Tokens validated: {db_answer.shape[1]}, Tokens per second: {(db_answer.shape[1]) / (end_time - start_time)}\n")
        
    return prob_sum / (token_cnt - prefix.shape[1])