import unittest
from evaluation.evaluation_metrics import EvaluationMetrics


class TestStringMethods(unittest.TestCase):
    """
        tests the class evaluation.evaluation_metrics.py
    """

    def setUp(self):
        pass


    def test_perfect_metric(self):
        """
            tests that the training scores are calculated correctly for a perfect prediction
        """
        y_true = [[1,0,0],[0,1,0],[0,0,1]]
        y_pred = y_true

        metrics = EvaluationMetrics(y_true=y_true, y_pred=y_pred)   # type: ignore
        
        assert metrics.subset_accuracy == 100
        assert metrics.hamming_score == 1
        assert len(metrics.f1_scores) == 3  # type: ignore
        assert metrics.f1_scores[0] == 1    # type: ignore
        assert metrics.f1_scores[1] == 1    # type: ignore
        assert metrics.f1_scores[2] == 1    # type: ignore


    def test_worst_metric(self):
        """
            tests that the training scores are calculated correctly for a perfect prediction
        """
        y_true = [[1,0,0],[0,1,0],[0,0,1]]
        y_pred = [[0,0,0],[0,0,0],[0,0,0]]
        metrics = EvaluationMetrics(y_true=y_true, y_pred=y_pred)   # type: ignore
        
        assert metrics.subset_accuracy == 0
        assert metrics.hamming_score == 0
        assert len(metrics.f1_scores) == 3  # type: ignore
        assert metrics.f1_scores[0] == 0    # type: ignore
        assert metrics.f1_scores[1] == 0    # type: ignore
        assert metrics.f1_scores[2] == 0    # type: ignore


    def tearDown(self):
        pass
