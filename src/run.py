import torch
from config import device
from ar_sampling import autoregressive_sampling
from transformers import AutoTokenizer, AutoModelForCausalLM
from sampling.speculative_sampling import speculative_sampling
from benchmark.benchmark import benchmark

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
    
    

def run(input, approx_model, approx_tokenizer, target_model, target_tokenizer, verbose, is_benchmark_needed, profiling, default_max_tokens, temperature, top_k, top_p, default_gamma, random_seed):
    # output arguments
    print(f'==========================================Arguments=========================================\n')
    print(f'Benchmark: {is_benchmark_needed}, max_tokens: {default_max_tokens}, temperature: {temperature}, top_k: {top_k}, top_p: {top_p}, gamma: {default_gamma}, random_seed: {random_seed}\n')
    print(f'============================================================================================\n')
    input_ids = approx_tokenizer.encode(input, return_tensors='pt').to(device)
    
    output_token_ids = speculative_sampling(input_ids, approx_model, target_model, default_max_tokens, temperature, top_k, top_p, default_gamma, random_seed)
    
    # approx_output = approx_tokenizer.decode(output_token_ids[0], skip_special_tokens=True)
    # target_output = target_tokenizer.decode(output_token_ids[0], skip_special_tokens=True)
    
    # print(f'Approx model output: {approx_output}\n\n')
    # print(f'Target model output: {target_output}\n\n')
    
    if is_benchmark_needed:
        benchmark(input_ids, target_model, default_max_tokens=default_max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)