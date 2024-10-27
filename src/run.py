import torch
from config import top_k, top_p, temperature, device
from ar_sampling import autoregressive_sampling
from transformers import AutoTokenizer, AutoModelForCausalLM
from sampling.speculative_sampling import speculative_sampling

def load_model(approx_model_path, target_model_path):
    '''
    Load the model and tokenizer from the given model path.
    
    Args:
        approx_model_path (str): The path to the approx model.
        target_model_path (str): The path to the target model.
        
    Returns:
        approx_tokenizer (transformers.AutoTokenizer): The tokenizer for the approx model.
        approx_model (transformers.AutoModelForCausalLM): The approx model.
        target_tokenizer (transformers.AutoTokenizer): The tokenizer for the target model.
        target_model (transformers.AutoModelForCausalLM): The target model.
    '''
    approx_tokenizer = AutoTokenizer.from_pretrained(approx_model_path,
                                                     local_files_only=True,
                                                     trust_remote_code=False)
    approx_model = AutoModelForCausalLM.from_pretrained(approx_model_path,
                                                       torch_dtype=torch.float16,
                                                       device_map='auto',
                                                       local_files_only=True,
                                                       trust_remote_code=False).to(device)
    
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path,
                                                     local_files_only=True,
                                                     trust_remote_code=False)
    target_model = AutoModelForCausalLM.from_pretrained(target_model_path,
                                                       torch_dtype=torch.float16,
                                                       device_map='auto',
                                                       local_files_only=True,
                                                       trust_remote_code=False).to(device)
    
    return approx_tokenizer, approx_model, target_tokenizer, target_model
    
    

def run(input, approx_model_path, target_model_path, verbose, seed, benchmark, profiling, max_tokens, gamma):
    approx_tokenizer, approx_model, target_tokenizer, target_model = load_model(approx_model_path, target_model_path)
    
    # approx_input_ids = approx_tokenizer.encode(input, return_tensors='pt').to(device)
    # output = autoregressive_sampling(approx_input_ids, approx_model, max_tokens, temperature, top_k, top_p)
    # approx_output = approx_tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f'Approx model output: {approx_output}\n\n')
    
    # target_input_ids = target_tokenizer.encode(input, return_tensors='pt').to(device)
    # output = autoregressive_sampling(target_input_ids, target_model, max_tokens, temperature, top_k, top_p)
    # target_output = target_tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f'Target model output: {target_output}\n\n')
    input_ids = approx_tokenizer.encode(input, return_tensors='pt').to(device)
    
    output_token_ids = speculative_sampling(input_ids, approx_model, target_model, max_tokens, gamma, temperature, top_k, top_p, seed)
    
    approx_output = approx_tokenizer.decode(output_token_ids[0], skip_special_tokens=True)
    target_output = target_tokenizer.decode(output_token_ids[0], skip_special_tokens=True)
    
    print(f'Approx model output: {approx_output}\n\n')
    print(f'Target model output: {target_output}\n\n')