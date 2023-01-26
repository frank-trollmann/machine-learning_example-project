import unittest
import numpy as np

from sklearn.model_selection import train_test_split

from evaluation.multi_run_evaluation import MultiRunEvaluation
from keras.preprocessing.image import ImageDataGenerator
from models.CNN_Builder import CNNBuilder

class TestMultiRunEvaluation(unittest.TestCase):
    """
        tests the dataset class
    """

    def setUp(self):
        pass

    def test_evaluate(self):
        """
            tests running the evaluation
            This contains multiple test cases that are all linked to the evaluation going through and thus have been merged into one method:
                1) running the evaluation works and is not causing an exception
                2) after running the evaluation, the results are available.
        """
        X = np.random.rand(10, 10, 10, 3)
        y = np.random.randint(0, 1, (10, 5))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)   
        train_datagen = ImageDataGenerator(
                            rotation_range=30, 
                            width_shift_range=0.1, 
                            height_shift_range=0.1, 
                            fill_mode="nearest"
                        )
        train_generator = train_datagen.flow(x=X_train, y=y_train)

        cnn_builder = CNNBuilder(
                            convolutional_layers=[1],
                            fully_connected_layers=[1],
                            in_shape=(X.shape[1], X.shape[2], X.shape[3]),
                            out_shape=y.shape[1])
        evaluator = MultiRunEvaluation(model_creation=cnn_builder.build_model)
        
        # test evaluation process runs exception-less
        try:
            evaluator.evaluate(
                            nr_runs=5,
                            epochs=5,
                            early_stopping_patience=5,
                            train_generator=train_generator,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            verbose = 0
                        )
        except Exception:
            self.fail("evaluate() raised an exception unexpectedly!")

        # test evaluation result are available and have right length
        assert len(evaluator.get_training_accuracies()) == 5
        assert len(evaluator.get_test_accuracies()) == 5
        assert len(evaluator.get_training_hamming_scores()) == 5
        assert len(evaluator.get_test_hamming_scores()) == 5

        
    def test_metrict_summary(self):
        summary = MultiRunEvaluation.get_metrics_summary([0,2,4])

        assert summary["min"] == 0
        assert summary["max"] == 4
        assert summary["mean"] == 2
        assert summary["std_dev"] == 2.0

    def tearDown(self):
        pass

