from kfoc.data import drifting_linear_stream
from kfoc.models import OnlineSGDClassifier
from kfoc.seed import set_seed
from kfoc.stream import run_prequential


def test_sgd_beats_chance_on_stationary_stream():
    set_seed(0)
    model = OnlineSGDClassifier(dim=5, num_classes=2, lr=0.1)
    stream = drifting_linear_stream(2000, 5, 2, drift="none", seed=0)
    result = run_prequential(model, stream)
    assert result["online_accuracy"] > 0.6


def test_stream_is_reproducible():
    a = [y for _, y in drifting_linear_stream(100, 5, seed=1)]
    b = [y for _, y in drifting_linear_stream(100, 5, seed=1)]
    assert a == b
