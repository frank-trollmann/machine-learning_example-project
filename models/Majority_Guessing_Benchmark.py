
import random

class Majority_Guessing_Benchmark:
    """
        A model that always guesses the majority class
    """

    def __init__(self,labels, majority_class):
        """
            constructs a random guessing model.
            labels --  a list of labels to select between.
        """
        self.labels = labels
        self.majority_index = labels.index(majority_class)
    
    def fit(X,y):
        """ Learning method. Nothing to do here. This method is here for interface alignment with other machine learning methods. """
        pass

    def predict_one(self):
        """ Predict one sample"""
        answer = [0]*len(self.labels)
        answer[self.majority_index] = 1

        return answer

    def predict(self, X):
        """ predict a set of samples"""
        y = [self.predict_one() for i in range(len(X))]
        return y