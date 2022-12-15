
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class Evaluation_Metrics():
    """
        This class calculates and summarizes evaluation metrics based on the predicted and true labels.
    """

    def __init__(self, y_true,y_pred):
        """ constructs the metrics by comparing the predicted labels (y_pred) and actual labels (y_true)"""
        # subset accuracy
        self.subset_accuracy = accuracy_score(y_true,y_pred)

        # hamming score
        self.hamming_score = Evaluation_Metrics.hamming_score(y_true,y_pred)

        # F1 score
        self.f1_scores = f1_score(y_true, y_pred, average=None)

    def print_evaluation_report(self, test_description):
        """ print a summary of the evaluation metrics to command line."""
        print("\n"+ test_description)
        print("- subset accuracy:",round(self.subset_accuracy*100),"%")
        print("- hamming score", round(self.hamming_score,2))
        print("- f1-scores: ", self.f1_scores)


    def hamming_score(y_true, y_pred):
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
            
