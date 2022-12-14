
import random

class Random_Guessing_Benchmark:
    """
        A model that just guesses one of the options randomly
    """

    def __init__(self,labels):
        """
            constructs a random guessing model.
            labels --  a list of labels to select between.
        """
        self.labels = labels
    
    def fit(X,y):
        """ Learning method. Nothing to do here. This method is here for interface alignment with other machine learning methods. """
        pass

    def predict_one(self):
        """ Predict one sample"""
        answer = [0]*len(self.labels)

        guessed_index = random.randint(0,len(self.labels)-1)
        answer[guessed_index] = 1

        return answer

    def predict(self, X):
        """ predict a set of samples"""
        y = [self.predict_one() for i in range(len(X))]
        return y