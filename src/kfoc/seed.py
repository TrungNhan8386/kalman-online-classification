import random

import numpy as np


def set_seed(seed: int) -> None:
    """Fix random seeds for Python, NumPy and (if installed) PyTorch.

    Args:
        seed: Seed shared by every random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
