from kfoc.metrics import OnlineMetrics
from kfoc.tracking import log


def run_prequential(model, stream, run=None, log_every: int = 100) -> dict:
    """Evaluate a model test-then-train on a stream and return the final metrics.

    Args:
        model: Object with ``predict_proba(x)`` and ``update(x, y)``.
        stream: Iterable of (x, y) pairs.
        run: Optional W&B run for periodic logging.
        log_every: Logging period in steps.

    Returns:
        Dict with cumulative online accuracy and average log-loss.
    """
    metrics = OnlineMetrics()
    for t, (x, y) in enumerate(stream, start=1):
        metrics.update(model.predict_proba(x), y)
        model.update(x, y)
        if t % log_every == 0:
            log(run, metrics.summary(), step=t)
    return metrics.summary()
