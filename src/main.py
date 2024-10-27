from parse_arg import parse_arguments
from run import run
import torch

if __name__ == '__main__':
    args = parse_arguments()
    torch.random.manual_seed(args.seed)
    run(input=args.input, 
        approx_model_path=args.approx_model_path, 
        target_model_path=args.target_model_path, 
        verbose=args.verbose, 
        seed=args.seed, 
        benchmark=args.benchmark, 
        profiling=args.profiling, 
        max_tokens=args.max_tokens, 
        gamma=args.gamma)