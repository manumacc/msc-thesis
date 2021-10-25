import sys
sys.path.append('src')

import argparse

import config
from src.experiment import Experiment

arg_parser = argparse.ArgumentParser(description="run active learning loop")
arg_parser.add_argument("name", type=str, help="name of the experiment or base model")
arg_parser.add_argument("--query", nargs="?", type=str, help="query strategy", dest="query")
arg_parser.add_argument("--ebano-enable-augment", action="store_true", dest="ebano_augment")
arg_parser.add_argument("--base", action="store_true", dest="train_base")

if __name__ == '__main__':
    args = arg_parser.parse_args()

    experiment = Experiment(config.config_dict)

    if args.train_base and args.query:
        print("Query strategy is ignored when training a base model")

    experiment.run(args.name, query_strategy=args.query, train_base=args.train_base, ebano_augment=args.ebano_augment)
