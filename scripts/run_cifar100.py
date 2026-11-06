"""Run an online learner on pre-extracted CIFAR-100 features."""
import argparse
import os

import numpy as np
import yaml

from kfoc.models import build_model
from kfoc.seed import set_seed
from kfoc.stream import run_prequential
from kfoc.tracking import init_run


def make_stream(x: np.ndarray, y: np.ndarray, seed: int):
    """Order the examples into a stream.

    TODO: replace the random order with the non-stationary protocol described in the paper.
    """
    order = np.random.default_rng(seed).permutation(len(y))
    return ((x[i], int(y[i])) for i in order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar100.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    d = cfg["data"]
    data = np.load(os.path.join(d["features_dir"], "train.npz"))
    x, y = data["x"], data["y"]

    m = dict(cfg["model"])
    run = init_run(cfg, enabled=not args.no_wandb)
    model = build_model(m.pop("method"), x.shape[1], int(y.max()) + 1, **m)
    result = run_prequential(model, make_stream(x, y, cfg["seed"]), run)
    print(result)
    if run is not None:
        run.summary.update(result)
        run.finish()


if __name__ == "__main__":
    main()
