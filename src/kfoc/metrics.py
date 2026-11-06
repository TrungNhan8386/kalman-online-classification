import numpy as np


class OnlineMetrics:
    """Accumulate prequential (test-then-train) metrics over a data stream."""

    def __init__(self):
        self.n = 0
        self.correct = 0
        self.log_loss = 0.0

    def update(self, probs: np.ndarray, y: int) -> None:
        """Record the prediction made for one example before training on it.

        Args:
            probs: Predicted class probabilities, shape (num_classes,).
            y: True label.
        """
        self.n += 1
        self.correct += int(np.argmax(probs) == y)
        self.log_loss += float(-np.log(max(probs[y], 1e-12)))

    def summary(self) -> dict:
        """Return cumulative online accuracy and average log-loss."""
        n = max(self.n, 1)
        return {"online_accuracy": self.correct / n, "avg_log_loss": self.log_loss / n}
