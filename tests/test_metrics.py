import numpy as np
import pytest

from source.metrics import apk, mapk, aupr_threshold, auc_prc_multilabel, calculate_auc


class TestApk:
    def test_perfect_score(self):
        assert apk([1, 2, 3], [1, 2, 3], k=3) == pytest.approx(1.0)

    def test_zero_score(self):
        assert apk([1, 2, 3], [4, 5, 6], k=3) == pytest.approx(0.0)

    def test_partial_match(self):
        score = apk([1, 2], [1, 3, 2], k=3)
        assert 0.0 < score < 1.0

    def test_k_truncation(self):
        score_k1 = apk([1], [1, 2, 3], k=1)
        score_k3 = apk([1], [1, 2, 3], k=3)
        assert score_k1 == pytest.approx(score_k3)

    def test_binary_actual_format(self):
        actual = np.array([1, 0, 1, 0, 0])
        predicted = np.array([0.9, 0.1, 0.8, 0.4, 0.3])
        score = apk(actual, predicted, k=2)
        assert 0.0 <= score <= 1.0


class TestMapk:
    def test_perfect_score(self):
        actual = [[1, 2], [3, 4]]
        predicted = [[1, 2, 5], [3, 4, 6]]
        assert mapk(actual, predicted, k=2) == pytest.approx(1.0)

    def test_zero_score(self):
        actual = [[1, 2], [3, 4]]
        predicted = [[5, 6], [7, 8]]
        assert mapk(actual, predicted, k=2) == pytest.approx(0.0)

    def test_average_of_apk(self):
        actual = [[1], [2]]
        predicted = [[1], [3]]
        result = mapk(actual, predicted, k=1)
        assert result == pytest.approx(0.5)


class TestAuprThreshold:
    def _make_pr_curve(self):
        from sklearn.metrics import precision_recall_curve
        y_true = np.array([1, 0, 1, 0, 1])
        y_score = np.array([0.9, 0.4, 0.8, 0.3, 0.7])
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        return precision, recall, thresholds

    def test_returns_float(self):
        precision, recall, thresholds = self._make_pr_curve()
        t = aupr_threshold(precision, recall, thresholds)
        assert isinstance(t, (float, np.floating))

    def test_threshold_in_range(self):
        precision, recall, thresholds = self._make_pr_curve()
        t = aupr_threshold(precision, recall, thresholds)
        assert thresholds.min() <= t <= thresholds.max()


class TestAucPrcMultilabel:
    def test_perfect_predictions(self):
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[1.0, 0.0], [0.0, 1.0]])
        score = auc_prc_multilabel(y_true, y_pred)
        assert score == pytest.approx(1.0)

    def test_returns_zero_on_degenerate(self):
        y_true = np.array([[0, 0], [0, 0]])
        y_pred = np.array([[0.5, 0.5], [0.5, 0.5]])
        score = auc_prc_multilabel(y_true, y_pred)
        assert score == pytest.approx(0.0)


class TestCalculateAuc:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        assert calculate_auc(y_pred, y_true) == pytest.approx(1.0)

    def test_random_classifier(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = rng.random(100)
        score = calculate_auc(y_pred, y_true)
        assert 0.0 <= score <= 1.0

    def test_2d_input_ravelled(self):
        y_true = np.array([[0, 1], [1, 0]])
        y_pred = np.array([[0.1, 0.9], [0.8, 0.2]])
        score = calculate_auc(y_pred, y_true)
        assert 0.0 <= score <= 1.0
