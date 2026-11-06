from typing import Iterator, Tuple

import numpy as np


def drifting_linear_stream(
    num_steps: int,
    dim: int,
    num_classes: int = 2,
    drift: str = "gradual",
    drift_scale: float = 0.05,
    switch_every: int = 1000,
    seed: int = 0,
) -> Iterator[Tuple[np.ndarray, int]]:
    """Yield (x, y) pairs whose labels come from a softmax-linear model with drifting weights.

    Args:
        num_steps: Length of the stream.
        dim: Input dimension.
        num_classes: Number of classes.
        drift: "gradual" (random walk on the weights), "abrupt" (weights resampled
            every ``switch_every`` steps) or "none" (stationary).
        drift_scale: Standard deviation of the per-step random-walk increment.
        switch_every: Period of abrupt weight changes.
        seed: Seed of the generator.

    Yields:
        x of shape (dim,) and an integer label y.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(num_classes, dim))
    for t in range(num_steps):
        if drift == "gradual":
            W = W + drift_scale * rng.normal(size=W.shape)
        elif drift == "abrupt" and t > 0 and t % switch_every == 0:
            W = rng.normal(size=W.shape)
        x = rng.normal(size=dim)
        logits = W @ x
        p = np.exp(logits - logits.max())
        p /= p.sum()
        y = int(rng.choice(num_classes, p=p))
        yield x, y


def load_cifar100(hf_dataset: str = "uoft-cs/cifar100", split: str = "train"):
    """Download CIFAR-100 from Hugging Face.

    Args:
        hf_dataset: Dataset id on the Hugging Face Hub.
        split: "train" or "test".

    Returns:
        A ``datasets.Dataset`` with an ``img`` column and label columns.
    """
    from datasets import load_dataset

    return load_dataset(hf_dataset, split=split)
