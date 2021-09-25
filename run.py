import sys
sys.path.append('src')

import argparse

import config
from src.experiment import Experiment

arg_parser = argparse.ArgumentParser(description="Run active learning loop")
arg_parser.add_argument("name", type=str, help="name of the experiment")
arg_parser.add_argument("query", type=str, help="query strategy")

if __name__ == '__main__':
    args = arg_parser.parse_args()

    experiment = Experiment(config.config_dict)
    experiment.run(args.name, args.query)
