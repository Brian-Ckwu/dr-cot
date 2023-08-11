import sys
sys.path.append("../src/dr-cot") # TODO: remove this

import yaml
from pathlib import Path
from argparse import ArgumentParser, Namespace

from utils import dict_to_namespace

def run_experiment(config: Namespace) -> None:
    """Run the experiment with the given configuration."""
    print(config)
    raise NotImplementedError

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--config_path", type=Path, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    run_experiment(config)
