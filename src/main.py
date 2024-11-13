from parse_arg import parse_arguments
from config import *
from run import load_model, test_sd_with_db_answer
from contextlib import redirect_stdout, redirect_stderr

torch.manual_seed(random_seed)

if __name__ == '__main__':     
    # {{question}} This is answer or related material: {{db_answer}}. Answer and explain.
    approx_tokenizer, approx_model, target_tokenizer, target_model = load_model(model_list['llama3.2-1b'], model_list['llama3-70b'])
    
    prompt_clue = "This is answer or related material"
    
    for ds_name in ["qa_noabs", "qa_reduced"]:
        file_path = f'/home/feic/pjs/Speculative-decoding-database/src/output/prompt_this_is_answer_or_related_material_maxtoke_128/sd_with_or_without_db_answer_1_70_{ds_name}_with_decode'
        # if file_path not exist, create it
        with open(file_path, 'w') as f:
            pass
        
        with open(file_path, 'w') as f:
            with redirect_stdout(f):
                print(f'Arguments: gamma={default_gamma}, max_tokens={default_max_tokens}, temperature={temperature}, top_k={top_k}, top_p={top_p}, random_seed={random_seed}')
                print('Model: llama3.2-1b, llama3-70b')
                test_sd_with_db_answer( approx_model, 
                                        approx_tokenizer, 
                                        target_model, 
                                        target_tokenizer, 
                                        default_max_tokens, 
                                        temperature, 
                                        top_k, 
                                        top_p, 
                                        random_seed, 
                                        threshold=threshold,
                                        dataset_name=ds_name,
                                        prompt_clue=prompt_clue,
                                        need_decode=True)
          
          
          
          
    # approx_tokenizer, approx_model, target_tokenizer, target_model = load_model(model_list['llama-68m'], model_list['llama2-70b'])
    
    # prompt_clue = "This is answer or related material"
    
    # for ds_name in ["qa_noabs", "qa_reduced"]:
    #     file_path = f'/home/feic/pjs/Speculative-decoding-database/src/output/prompt_this_is_answer_or_related_material_maxtoke_128/sd_with_or_without_db_answer_68_70_{ds_name}'
    #     # if file_path not exist, create it
    #     with open(file_path, 'w') as f:
    #         pass
        
    #     with open(file_path, 'w') as f:
    #         with redirect_stdout(f):
    #             print(f'Arguments: gamma={default_gamma}, max_tokens={default_max_tokens}, temperature={temperature}, top_k={top_k}, top_p={top_p}, random_seed={random_seed}')
    #             print('Model: llama-68m, llama2-70b')
    #             test_sd_with_db_answer( approx_model, 
    #                                     approx_tokenizer, 
    #                                     target_model, 
    #                                     target_tokenizer, 
    #                                     default_max_tokens, 
    #                                     temperature, 
    #                                     top_k, 
    #                                     top_p, 
    #                                     random_seed, 
    #                                     threshold=threshold,
    #                                     dataset_name=ds_name,
    #                                     prompt_clue=prompt_clue,)
                