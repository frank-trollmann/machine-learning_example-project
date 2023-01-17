import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class EvaluationMetrics:
    """
        This class calculates and summarizes evaluation metrics based on the predicted and true labels.
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ constructs the metrics by comparing the predicted labels (y_pred) and actual labels (y_true)"""

        if type(y_true) is not np.ndarray:
            y_true = np.array(y_true)
        
        if type(y_pred) is not np.ndarray:
            y_pred = np.array(y_pred)

        # subset accuracy
        self.subset_accuracy = round(accuracy_score(y_true=y_true, y_pred=y_pred) * 100)    # type: ignore

        # hamming score
        self.hamming_score = round(self.calculate_hamming_score(y_true=y_true, y_pred=y_pred), 2)

        # F1 score
        self.f1_scores = f1_score(y_true, y_pred, average=None) # type: ignore
        self.f1_scores = [round(f1_score,2) for f1_score in self.f1_scores]

        # accracy per class
        self.class_accuracies = [accuracy_score(y_true=y_true[:,i],y_pred=y_pred[:,i]) for i in range(len(y_pred[0]))] 
        self.class_accuracies = [round(accuracy *100) for accuracy in self.class_accuracies]


    def print_evaluation_report(self, test_description: str) -> None:
        """ print a summary of the evaluation metrics to command line."""

        print(f"\n{test_description}")
        print("- subset accuracy:", self.subset_accuracy,"%")
        print("- hamming score", self.hamming_score)
        print("- f1-scores: ", self.f1_scores)
        print("- class accuracies: ", self.class_accuracies)

        return


    def calculate_hamming_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ calculates the hamming score between predicted labels y_pred and original laels y_true """

        sum = 0

        for i in range(len(y_true)):
            intersection = len([1 for j in range(len(y_true[i])) if y_true[i][j] == 1 and y_pred[i][j] == 1 ])
            union = len([1 for j in range(len(y_true[i])) if y_true[i][j] == 1 or y_pred[i][j] == 1 ])

            if intersection == 0 and union == 0:
                sum += 1
            else:
                sum += intersection / union
        
        return sum / len(y_true)
