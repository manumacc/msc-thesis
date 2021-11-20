import sys
sys.path.append('src')

import argparse

import config
from src.experiment import Experiment

arg_parser = argparse.ArgumentParser(description="run active learning loop")
arg_parser.add_argument("name", type=str, help="name of the experiment or base model")
arg_parser.add_argument('--seed', nargs='?', default=None, type=int)

arg_parser.add_argument("--base", action="store_true", dest="train_base")

arg_parser.add_argument("--query", nargs="?", type=str, help="query strategy", dest="query")

arg_parser.add_argument("--ebano-mix-base-strategy", nargs='?', default=None, type=str, dest="ebano_mix_base_strategy")
arg_parser.add_argument("--ebano-mix-query-limit", nargs='?', default=None, type=int, dest="ebano_mix_query_limit")
arg_parser.add_argument("--ebano-mix-augment-limit", nargs='?', default=None, type=int, dest="ebano_mix_augment_limit")
arg_parser.add_argument("--ebano-mix-min-diff", nargs='?', default=None, type=float, dest="ebano_mix_min_diff")
arg_parser.add_argument("--ebano-mix-subset", nargs='?', default=None, type=int, dest="ebano_mix_subset")

arg_parser.add_argument("--dataset-hpc", nargs='?', default=None, type=str, dest="dataset_hpc")

arg_parser.add_argument("--resume-job", nargs='?', default=None, type=str, dest="resume_job")

if __name__ == '__main__':
    args = arg_parser.parse_args()

    experiment = Experiment(config.config_dict)

    if args.train_base and args.query:
        print("Query strategy is ignored when training a base model")

    experiment.run(args.name,
                   seed=args.seed,
                   train_base=args.train_base,
                   query_strategy=args.query,
                   ebano_mix_base_strategy=args.ebano_mix_base_strategy,
                   ebano_mix_query_limit=args.ebano_mix_query_limit,
                   ebano_mix_augment_limit=args.ebano_mix_augment_limit,
                   ebano_mix_min_diff=args.ebano_mix_min_diff,
                   ebano_mix_subset=args.ebano_mix_subset,
                   dataset_hpc=args.dataset_hpc,
                   resume_job=args.resume_job)
