from sampling.speculative_sampling import speculative_sampling, load_model
import torch
from config import default_gamma, model_list, random_seed

torch.manusal_seed(random_seed)

@torch.no_grad()
def test_output( # The answer will be fetched from the local database
    approx_model,
    approx_tokenizer,
    target_model,
    target_tokenizer,
    default_max_tokens,
    temperature,
    top_k,
    top_p,
    random_seed,
):
    torch.manual_seed(random_seed)
    while True:
        prompt = input("Prompt with answer: ")
        input_ids = approx_tokenizer(prompt, return_tensors="pt").input_ids
        output_token_ids, alpha, inference_time = speculative_sampling(input_ids, 
                                                approx_model, 
                                                target_model, 
                                                default_max_tokens, 
                                                temperature, 
                                                top_k, 
                                                top_p, 
                                                default_gamma, 
                                                random_seed)

        target_output = target_tokenizer.decode(output_token_ids[0], skip_special_tokens=True)
        print(f"Target output: {target_output}\n\n\n\n\n")
        
        print(f"=================================Running benchmark=================================\n\n\n\n\n")
        
        # seperate prompt with '?', get the question before '?'
        question = prompt.split('?')[0] + '?'
        input_ids = approx_tokenizer(question, return_tensors="pt").input_ids
        output_token_ids, alpha, inference_time = speculative_sampling(input_ids, 
                                                approx_model, 
                                                target_model, 
                                                default_max_tokens, 
                                                temperature, 
                                                top_k, 
                                                top_p, 
                                                default_gamma, 
                                                random_seed)
        target_output = target_tokenizer.decode(output_token_ids[0], skip_special_tokens=True)
        print(f"Target output: {target_output}\n\n\n\n\n")
        
        print(f"=================================End this round=================================\n\n\n\n\n")

      
approx_tokenizer, approx_model, target_tokenizer, target_model = load_model(model_list['llama3.2-1b'], model_list['llama3-70b'])
test_output(approx_model, approx_tokenizer, target_model, target_tokenizer, 128, 0.8, 20, 0.9, 42)