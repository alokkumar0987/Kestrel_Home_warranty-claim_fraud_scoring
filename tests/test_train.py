"""The predictions file must match sample_submission.csv exactly, or training stops."""
import pandas as pd
import pytest

from kestrel_fraud.train import write_predictions

SAMPLE = pd.DataFrame({"claim_id": ["A", "B", "C"], "score": 0.0})


def test_write_predictions_keeps_sample_order(tmp_path):
    scored = pd.DataFrame({"claim_id": ["C", "A", "B"], "score": [0.3, 0.1, 0.2]})
    write_predictions(scored, SAMPLE, tmp_path / "predictions.csv")
    saved = pd.read_csv(tmp_path / "predictions.csv")
    assert list(saved.columns) == ["claim_id", "score"]
    assert saved["claim_id"].tolist() == ["A", "B", "C"] and saved["score"].tolist() == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("scored, problem", [
    (pd.DataFrame({"claim_id": ["A", "B"], "score": [0.1, 0.2]}), "missing scores"),
    (pd.DataFrame({"claim_id": ["A", "B", "C"], "score": [0.1, 0.2, 1.5]}), "scores outside"),
])
def test_write_predictions_refuses_a_bad_file(tmp_path, scored, problem):
    with pytest.raises(ValueError, match=problem):
        write_predictions(scored, SAMPLE, tmp_path / "predictions.csv")
    assert not (tmp_path / "predictions.csv").exists()
