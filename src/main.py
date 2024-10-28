from parse_arg import parse_arguments
from config import *
from run import run_model_sd, run_db_sd, load_model
import os, sys
from contextlib import redirect_stdout, redirect_stderr

if __name__ == '__main__':
    # Interactive mode
    # args = parse_arguments()
    # torch.random.manual_seed(args.seed)
    # run(input=args.input, 
    #     approx_model_path=args.approx_model_path, 
    #     target_model_path=args.target_model_path, 
    #     verbose=args.verbose, 
    #     is_benchmark_needed=args.benchmark, 
    #     profiling=args.profiling, 
    #     )
    
    
    
    # Non-interactive mode, for testing purposes
    approx_tokenizer, approx_model, target_tokenizer, target_model = load_model(model_list['llama3.2-1b'], model_list['llama3-8b'])
    input = 'Any recommendations for playing video games?'
    
    #     random_seed = [42]
    # default_max_tokens = [50, 100, 150]
    # default_gamma = [4, 5, 6]
    # top_k = [20, 30, 40]
    # top_p = [0.9]
    # temperature = [0.5, 1, 1.5]
    output_dir = '/ibex/user/feic/pjs/Speculative-decoding/Speculative-decoding-database/src/output/output-8-70'
    cnt = 1
    # for seed in random_seed:
    #     for max_tokens in default_max_tokens:
    #         for gamma in default_gamma:
    #             for k in top_k:
    #                 for p in top_p:
    #                     for temp in temperature:
    #                         file_name = f'{cnt}-output_{seed}_{max_tokens}_{gamma}_{k}_{p}_{temp}.txt'
    #                         file_path = os.path.join(output_dir, file_name)
    #                         cnt += 1
    #                         with open(file_path, 'w') as f:
    #                             with redirect_stdout(f), redirect_stderr(f):
    #                                 run_model_sd(input, 
    #                                     approx_model, 
    #                                     approx_tokenizer, 
    #                                     target_model, 
    #                                     target_tokenizer, 
    #                                     verbose=False, 
    #                                     is_benchmark_needed=is_benchmark_needed, 
    #                                     profiling=False, 
    #                                     default_max_tokens=max_tokens, 
    #                                     temperature=temp, 
    #                                     top_k=k, 
    #                                     top_p=p, 
    #                                     default_gamma=gamma, 
    #                                     random_seed=seed)
    #                         print('Done')
    
    for seed in random_seed:
        for max_tokens in default_max_tokens:
            for gamma in default_gamma:
                for k in top_k:
                    for p in top_p:
                        for temp in temperature:
                            run_db_sd(input, target_model, target_tokenizer, default_max_tokens=max_tokens, temperature=temp, top_k=k, top_p=p, random_seed=seed, threshold=0.5)