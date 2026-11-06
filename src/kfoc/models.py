import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class OnlineSGDClassifier:
    """Baseline: multinomial logistic regression trained with one SGD step per example."""

    def __init__(self, dim: int, num_classes: int, lr: float = 0.1):
        """
        Args:
            dim: Input dimension.
            num_classes: Number of classes.
            lr: SGD learning rate.
        """
        self.W = np.zeros((num_classes, dim))
        self.b = np.zeros(num_classes)
        self.lr = lr

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities for a single input x of shape (dim,)."""
        return softmax(self.W @ x + self.b)

    def update(self, x: np.ndarray, y: int) -> None:
        """Take one SGD step on the cross-entropy loss of (x, y)."""
        g = self.predict_proba(x)
        g[y] -= 1.0
        self.W -= self.lr * np.outer(g, x)
        self.b -= self.lr * g


class KalmanOnlineClassifier:
    """Kalman filter over linear predictor weights with a learned forgetting coefficient.

    This is the core contribution of the paper and must be implemented by the team
    from the paper's description (state-space model and its inference). Expected pieces:
        - state-space model over the weights with a drift transition controlled by
          the forgetting coefficient;
        - Kalman predict / update recursions for the posterior mean and covariance;
        - online learning of the forgetting coefficient;
        - the approximation used to turn the Gaussian model into a classifier.
    """

    def __init__(self, dim: int, num_classes: int, **hparams):
        """
        Args:
            dim: Input (feature) dimension.
            num_classes: Number of classes.
            **hparams: Hyperparameters as named in the paper.
        """
        self.dim = dim
        self.num_classes = num_classes
        self.hparams = hparams

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return the predictive class probabilities for x under the current posterior."""
        raise NotImplementedError("TODO: implement from the paper")

    def update(self, x: np.ndarray, y: int) -> None:
        """Run one predict/update step of the filter and adapt the forgetting coefficient."""
        raise NotImplementedError("TODO: implement from the paper")


def build_model(method: str, dim: int, num_classes: int, **kwargs):
    """Instantiate an online learner by name ("sgd" or "kalman")."""
    if method == "sgd":
        return OnlineSGDClassifier(dim, num_classes, lr=kwargs.get("lr", 0.1))
    if method == "kalman":
        return KalmanOnlineClassifier(dim, num_classes, **kwargs)
    raise ValueError(f"Unknown method: {method}")
