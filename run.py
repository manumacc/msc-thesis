import sys
sys.path.append('src')

import argparse

import config
from src.experiment import Experiment

arg_parser = argparse.ArgumentParser(description="run active learning loop")
arg_parser.add_argument("name", type=str, help="name of the experiment or base model")
arg_parser.add_argument('--seed', nargs='?', default=None, type=int)
arg_parser.add_argument("--query", nargs="?", type=str, help="query strategy", dest="query")
arg_parser.add_argument("--resume-job", nargs='?', default=None, type=str, dest="resume_job")

arg_parser.add_argument("--freeze-extractor", action="store_true", dest="freeze_extractor")

# Base model
arg_parser.add_argument("--base", action="store_true", dest="train_base")
arg_parser.add_argument("--base-epochs", nargs="?", default=None, type=int, dest="base_n_epochs")
arg_parser.add_argument("--base-lr-init", nargs="?", default=None, type=float, dest="base_lr_init")
arg_parser.add_argument("--base-reduce-lr-min", nargs="?", default=None, type=float, dest="base_reduce_lr_min")

# AL
arg_parser.add_argument("--loops", nargs="?", default=None, type=int, dest="n_loops")
arg_parser.add_argument("--epochs", nargs="?", default=None, type=int, dest="n_epochs")
arg_parser.add_argument("--lr-init", nargs="?", default=None, type=float, dest="lr_init")
arg_parser.add_argument("--reduce-lr-min", nargs="?", default=None, type=float, dest="reduce_lr_min")

arg_parser.add_argument("--n-query-instances", nargs="?", default=None, type=int, dest="n_query_instances")

arg_parser.add_argument("--ebano-base-strategy", nargs='?', default=None, type=str, dest="ebano_base_strategy")
arg_parser.add_argument("--ebano-query-limit", nargs='?', default=None, type=int, dest="ebano_query_limit")
arg_parser.add_argument("--ebano-augment-limit", nargs='?', default=None, type=int, dest="ebano_augment_limit")
arg_parser.add_argument("--ebano-min-diff", nargs='?', default=None, type=float, dest="ebano_min_diff")
arg_parser.add_argument("--ebano-subset", nargs='?', default=None, type=int, dest="ebano_subset")

arg_parser.add_argument("--ebano-mix-base-strategy", nargs='?', default=None, type=str, dest="ebano_mix_base_strategy")
arg_parser.add_argument("--ebano-mix-query-limit", nargs='?', default=None, type=int, dest="ebano_mix_query_limit")
arg_parser.add_argument("--ebano-mix-augment-limit", nargs='?', default=None, type=int, dest="ebano_mix_augment_limit")
arg_parser.add_argument("--ebano-mix-min-diff", nargs='?', default=None, type=float, dest="ebano_mix_min_diff")
arg_parser.add_argument("--ebano-mix-subset", nargs='?', default=None, type=int, dest="ebano_mix_subset")

arg_parser.add_argument("--dataset-hpc", nargs='?', default=None, type=str, dest="dataset_hpc")
arg_parser.add_argument("--base-model-overwrite", nargs='?', default=None, type=str, dest="base_model_overwrite")

if __name__ == '__main__':
    args = arg_parser.parse_args()

    experiment = Experiment(config.config_dict)

    if args.train_base and args.query:
        print("Query strategy is ignored when training a base model")

    experiment.run(args.name,
                   seed=args.seed,
                   train_base=args.train_base,
                   query_strategy=args.query,
                   ebano_base_strategy=args.ebano_base_strategy,
                   ebano_query_limit=args.ebano_query_limit,
                   ebano_augment_limit=args.ebano_augment_limit,
                   ebano_min_diff=args.ebano_min_diff,
                   ebano_subset=args.ebano_subset,
                   ebano_mix_base_strategy=args.ebano_mix_base_strategy,
                   ebano_mix_query_limit=args.ebano_mix_query_limit,
                   ebano_mix_augment_limit=args.ebano_mix_augment_limit,
                   ebano_mix_min_diff=args.ebano_mix_min_diff,
                   ebano_mix_subset=args.ebano_mix_subset,
                   dataset_hpc=args.dataset_hpc,
                   base_model_overwrite=args.base_model_overwrite,
                   resume_job=args.resume_job,
                   freeze_extractor=args.freeze_extractor,
                   base_n_epochs=args.base_n_epochs,
                   base_lr_init=args.base_lr_init,
                   base_reduce_lr_min=args.base_reduce_lr_min,
                   n_loops=args.n_loops,
                   n_epochs=args.n_epochs,
                   lr_init=args.lr_init,
                   reduce_lr_min=args.reduce_lr_min,
                   n_query_instances=args.n_query_instances)
