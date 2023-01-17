import unittest
from evaluation.evaluation_metrics import EvaluationMetrics


class TestEvaluationMetrics(unittest.TestCase):
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

        for class_index in range(len(metrics.f1_scores)):
            assert metrics.f1_scores[class_index] == 1 
            assert metrics.class_accuracies[class_index] == 100 





    def test_worst_metric(self):
        """
            tests that the training scores are calculated correctly for the worst prediction
        """
        y_true = [[1,0,0],[0,1,0],[0,0,1]]
        y_pred = [[0,1,1],[1,0,1],[1,1,0]]
        metrics = EvaluationMetrics(y_true=y_true, y_pred=y_pred)   # type: ignore
        
        assert metrics.subset_accuracy == 0
        assert metrics.hamming_score == 0
        assert len(metrics.f1_scores) == 3  # type: ignore
        
        for class_index in range(len(metrics.f1_scores)):
            assert metrics.f1_scores[class_index] == 0
            assert metrics.class_accuracies[class_index] == 0 


    def tearDown(self):
        pass
