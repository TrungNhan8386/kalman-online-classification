"""Run an online learner on a synthetic drifting stream."""
import argparse

import yaml

from kfoc.data import drifting_linear_stream
from kfoc.models import build_model
from kfoc.seed import set_seed
from kfoc.stream import run_prequential
from kfoc.tracking import init_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    d = cfg["data"]
    m = dict(cfg["model"])
    run = init_run(cfg, enabled=not args.no_wandb)
    model = build_model(m.pop("method"), d["dim"], d["num_classes"], **m)
    stream = drifting_linear_stream(
        d["num_steps"], d["dim"], d["num_classes"], d["drift"],
        d["drift_scale"], d["switch_every"], seed=cfg["seed"],
    )
    result = run_prequential(model, stream, run)
    print(result)
    if run is not None:
        run.summary.update(result)
        run.finish()


if __name__ == "__main__":
    main()
