import argparse
from config import model_list, default_max_tokens, default_gamma


def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    
    parser.add_argument('--input', type=str, default='Any recommendations for playing video games?')
    parser.add_argument('--approx_model_path', type=str, default=model_list['llama3.2-1b'])
    parser.add_argument('--target_model_path', type=str, default=model_list['llama3-8b'])
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=42, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=default_max_tokens, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=default_gamma, help='guess time.')
    
    args = parser.parse_args()
    return args