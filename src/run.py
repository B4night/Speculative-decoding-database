import torch
from config import *
from ar_sampling import autoregressive_sampling
from transformers import AutoTokenizer, AutoModelForCausalLM
from sampling.speculative_sampling import speculative_sampling
from benchmark.benchmark import benchmark
from database.db import LC_QuAD_query_test
from database.validate import validate
import time
import pandas as pd
import pudb

torch.manual_seed(random_seed)

def load_model(
    approx_model_path, 
    target_model_path
):
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
                                                       trust_remote_code=False)
    
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path,
                                                     local_files_only=True,
                                                     trust_remote_code=False)
    target_model = AutoModelForCausalLM.from_pretrained(target_model_path,
                                                       torch_dtype=torch.float16,
                                                       device_map='auto',
                                                       local_files_only=True,
                                                       trust_remote_code=False)
    
    return approx_tokenizer, approx_model, target_tokenizer, target_model
    
    
@torch.no_grad()
def run_model_sd(input, approx_model, approx_tokenizer, target_model, target_tokenizer, verbose, is_benchmark_needed, profiling, default_max_tokens, temperature, top_k, top_p, default_gamma, random_seed):
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
        print(f'Running benchmark on target model...')
        benchmark(input_ids, target_model, default_max_tokens=default_max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        
        print(f'Running benchmark on approx model...')
        benchmark(input_ids, approx_model, default_max_tokens=default_max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
     
@torch.no_grad()
def run_db_sd_test( # The answer will be fetched from the remote database
    approx_model,
    approx_tokenizer,
    target_model,
    target_tokenizer,
    default_max_tokens,
    temperature,
    top_k,
    top_p,
    random_seed,
    threshold,
):
    inference_time_list = []
    benchmark_time_list = []
    db_tokens_per_second_list = []
    benchmark_tokens_per_second_list = []

    while True:
        # Fetch the question and database answer
        question, db_answer = LC_QuAD_query_test(input)
        if question == 'No more questions.':
            break

        # Prepare the prompt and determine if speculative decoding is needed
        db_answer_exist = not db_answer
        prompt = question if db_answer_exist else f"{question}\nAnswer based on the reference: {db_answer}"
        print(f'Prompt: {prompt}')
        print(f'Speculative decoding needed: {db_answer_exist}')
        print(f'\ndb_answer: {db_answer}\n\n')

        # Encode the input prompt
        input_ids = approx_tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Measure inference time for the approximate model
        start_time = time.time()
        if db_answer_exist:
            print(f"=====================================SPECULATIVE SAMPLING=====================================")
            output_token_ids = speculative_sampling(
                input_ids,
                approx_model,
                target_model,
                default_max_tokens,
                temperature,
                top_k,
                top_p,
                default_gamma,
                random_seed,
            )
        else:
            print(f"=====================================DB AR SAMPLING=====================================")
            output_token_ids = autoregressive_sampling(
                input_ids,
                approx_model,
                default_max_tokens,
                temperature,
                top_k,
                top_p,
            )
        end_time = time.time()

        # Calculate inference metrics
        inference_time = end_time - start_time
        inference_time_list.append(inference_time)
        generated_tokens = output_token_ids.shape[1] - input_ids.shape[1]
        db_tokens_per_second = generated_tokens / inference_time
        db_tokens_per_second_list.append(db_tokens_per_second)
        print(f'Inference time: {inference_time}')
        print(f'Tokens per second: {db_tokens_per_second}', flush=True)

        # Perform validation if speculative decoding is not needed
        if not db_answer_exist:
            validation_score = validate(
                target_model,
                input_ids,
                output_token_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if validation_score > threshold:
                print(f'Validation Score: {validation_score} - Validation passed.', flush=True)
            else:
                print('Validation failed.', flush=True)

        # Benchmark the target model
        start_time = time.time()
        question_ids = target_tokenizer.encode(question, return_tensors='pt').to(device)
        benchmark_token_ids = autoregressive_sampling(
            question_ids,
            target_model,
            default_max_tokens,
            temperature,
            top_k,
            top_p,
        )
        end_time = time.time()

        # Calculate benchmark metrics
        benchmark_time = end_time - start_time
        benchmark_time_list.append(benchmark_time)
        benchmark_generated_tokens = benchmark_token_ids.shape[1] - question_ids.shape[1]
        benchmark_tokens_per_second = benchmark_generated_tokens / benchmark_time
        benchmark_tokens_per_second_list.append(benchmark_tokens_per_second)
        ratio = db_tokens_per_second / benchmark_tokens_per_second
        print(f'Benchmark tokens per second: {benchmark_tokens_per_second}. Ratio: {ratio}\n\n', flush=True)
        print(f'--------------------------------------------------------------------------------------------------\n\n')

        # Free up memory
        del input_ids, output_token_ids, question_ids, benchmark_token_ids
        torch.cuda.empty_cache()

    # Calculate and print average metrics
    avg_inference_time = sum(inference_time_list) / len(inference_time_list) if inference_time_list else 0
    avg_benchmark_time = sum(benchmark_time_list) / len(benchmark_time_list) if benchmark_time_list else 0
    avg_db_tps = sum(db_tokens_per_second_list) / len(db_tokens_per_second_list) if db_tokens_per_second_list else 0
    avg_benchmark_tps = sum(benchmark_tokens_per_second_list) / len(benchmark_tokens_per_second_list) if benchmark_tokens_per_second_list else 0

    print(f'\n\nAverage inference time: {avg_inference_time}')
    print(f'Average benchmark time: {avg_benchmark_time}')
    print(f'Average tokens per second: {avg_db_tps}')
    print(f'Average benchmark tokens per second: {avg_benchmark_tps}')
    
    
@torch.no_grad()
def test_sd_with_db_answer( # The answer will be fetched from the local database
    approx_model,
    approx_tokenizer,
    target_model,
    target_tokenizer,
    default_max_tokens,
    temperature,
    top_k,
    top_p,
    random_seed,
    threshold,
    dataset_name,
    prompt_clue,
    need_decode=False
):
    '''
    1. get answer from the database
    2. if answer exists:
        case 1: run sd with new prompt. 
        case 2: Else run sd with old prompt
    
    Things to record:
    1. case 1: time taken, acceptance rate(alpha), tokens_per_second
    2. case 2: time taken, acceptance rate(alpha), tokens_per_second
    '''
    # pudb.set_trace()
    torch.manual_seed(random_seed)
    
    round_num = 0
    query_time_list = []
    sd_with_answer_time_list = []
    sd_with_answer_alpha_list = []
    sd_with_answer_tokens_per_second_list = []
    sd_without_answer_time_list = []
    sd_without_answer_alpha_list = []
    sd_without_answer_tokens_per_second_list = []
    
    # dataset_path = '/home/feic/pjs/Speculative-decoding-database/data/qa_reduced.csv'
    # dataset_path = '/home/feic/pjs/Speculative-decoding-database/data/qa_noabs.csv'
    dataset_path = f'/home/feic/pjs/Speculative-decoding-database/data/{dataset_name}.csv'
    
    # db = pd.read_csv('/home/feic/pjs/Speculative-decoding-database/data/qa_sample.csv', sep='§', engine='python')
    db = pd.read_csv(dataset_path, sep='§', engine='python')
    
    print(f'Dataset: {dataset_path}\n\n')
    
    # TEST
    # forward_continue = False
    
    for i in range(len(db)):
        
        if round_num > 1001:
            break
        
        # Fetch the question and database answer
        start_time = time.time()
        question, db_answer = db.iloc[i]['Question'], db.iloc[i]['Answer']
        ens_time = time.time()
        
        query_time = ens_time - start_time
        
        # TEST -------------------
        # if question != "Which show's theme music composer's label is MapleMusic Recordings?" and forward_continue == False:
        #     continue
        # forward_continue = True
        # TEST -------------------
        
        db_answer_exist = not pd.isna(db_answer)
        if db_answer_exist == False:
            continue
        
        query_time_list.append(query_time)
        
        round_num += 1
        
        print(f'===========================================================The {round_num}th question============================================================\n')
        
        # TEST----
        if round_num == 233:
            print("hello!")
        # ----
        
        # prompt = f"{question} Answer based on the reference: {db_answer}" if db_answer_exist else question
        prompt = f"{question} {prompt_clue}: {db_answer}" if db_answer_exist else question
        print(f'Prompt: {prompt}')
        print(f'\ndb_answer: {db_answer}\n\n')
        
        input_ids = approx_tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        
        output_token_ids, alpha, inference_time = speculative_sampling(input_ids, 
                                                       approx_model, 
                                                       target_model, 
                                                       default_max_tokens, 
                                                       temperature, 
                                                       top_k, 
                                                       top_p, 
                                                       default_gamma, 
                                                       random_seed)
        
        if need_decode:
            target_output = target_tokenizer.decode(output_token_ids[0], skip_special_tokens=True)
            print(f'\nTarget model output: {target_output}\n\n')
        
        # Record info
        sd_with_answer_time_list.append(inference_time + query_time)
        sd_with_answer_alpha_list.append(alpha)
        sd_with_answer_tokens_per_second_list.append((output_token_ids.shape[1] - input_ids.shape[1]) / (inference_time + query_time))
        
        # Below is benchmark, sd without db_answer, only with original question
        print (f'----------------------------------Running benchmark-----------------------------------------')
    
        input_ids = approx_tokenizer.encode(question, return_tensors='pt').to(device)
        
        output_token_ids, alpha, benchmark_time = speculative_sampling(input_ids, 
                                                       approx_model, 
                                                       target_model, 
                                                       default_max_tokens, 
                                                       temperature, 
                                                       top_k, 
                                                       top_p, 
                                                       default_gamma, 
                                                       random_seed)
        
        if need_decode:
            target_output = target_tokenizer.decode(output_token_ids[0], skip_special_tokens=True)
            print(f'\nTarget model output: {target_output}\n\n')
    
        sd_without_answer_time_list.append(benchmark_time)
        sd_without_answer_alpha_list.append(alpha)
        sd_without_answer_tokens_per_second_list.append((output_token_ids.shape[1] - input_ids.shape[1]) / benchmark_time)
        
        if round_num % 100 == 0:
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~Output for the {round_num - 100} to {round_num} question:~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            print(f'Average query time: {sum(query_time_list) / len(query_time_list)}')
            print(f'Average time taken for sd with db_answer: {sum(sd_with_answer_time_list) / len(sd_with_answer_time_list)}')
            print(f'Average alpha for sd with db_answer: {sum(sd_with_answer_alpha_list) / len(sd_with_answer_alpha_list)}')
            print(f'Average tokens per second for sd with db_answer: {sum(sd_with_answer_tokens_per_second_list) / len(sd_with_answer_tokens_per_second_list)}')
            
            print(f'Average time taken for sd without db_answer: {sum(sd_without_answer_time_list) / len(sd_without_answer_time_list)}')
            print(f'Average alpha for sd without db_answer: {sum(sd_without_answer_alpha_list) / len(sd_without_answer_alpha_list)}')
            print(f'Average tokens per second for sd without db_answer: {sum(sd_without_answer_tokens_per_second_list) / len(sd_without_answer_tokens_per_second_list)}')
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n')
        
    print(f'sd_with_answer_time_list: {sd_with_answer_time_list}')
    print(f'sd_with_answer_alpha_list: {sd_with_answer_alpha_list}')
    print(f'sd_with_answer_tokens_per_second_list: {sd_with_answer_tokens_per_second_list}')
    
    print(f'sd_without_answer_time_list: {sd_without_answer_time_list}')
    print(f'sd_without_answer_alpha_list: {sd_without_answer_alpha_list}')
    print(f'sd_without_answer_tokens_per_second_list: {sd_without_answer_tokens_per_second_list}\n\n')

    print(f'Average query time: {sum(query_time_list) / len(query_time_list)}')

    print(f'Average time taken for sd with db_answer: {sum(sd_with_answer_time_list) / len(sd_with_answer_time_list)}')
    print(f'Average alpha for sd with db_answer: {sum(sd_with_answer_alpha_list) / len(sd_with_answer_alpha_list)}')
    print(f'Average tokens per second for sd with db_answer: {sum(sd_with_answer_tokens_per_second_list) / len(sd_with_answer_tokens_per_second_list)}')
    
    print(f'Average time taken for sd without db_answer: {sum(sd_without_answer_time_list) / len(sd_without_answer_time_list)}')
    print(f'Average alpha for sd without db_answer: {sum(sd_without_answer_alpha_list) / len(sd_without_answer_alpha_list)}')
    print(f'Average tokens per second for sd without db_answer: {sum(sd_without_answer_tokens_per_second_list) / len(sd_without_answer_tokens_per_second_list)}')