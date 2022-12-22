
import keras.backend as kbackend
import  tensorflow as tf

import statistics 

from evaluation.evaluation_metrics import Evaluation_Metrics

class Multi_Run_Evaluation:
    """
        This class carries out an evaluation of a model in multiple runs, recording performance metrics and calculating statistical measures.
    """

    def __init__(self, model_creation):
        """
            inits the evaluation with a specific model.
            does not yet run the evaluation. 

            Parameters:
                model_creation -- A method that can be called to get a new instance of the model
        """

        self.model_creation = model_creation
        self.test_results = []
        self.training_results = []

    def evaluate(self, nr_runs, epochs, early_stopping_patience, train_generator, X_train, y_train, X_test,y_test):
        """
            Trains the model a given number of times with the given training data and configuration.
            Records evaluation scores for each run. 

            Parameters:
                nr_runs -- the number of times the machine the machine learning model to test.
                epochs -- the number of epochs to train
                early_stopping_patience -- how many epochs the early stopping allows to pass by without improvement of the test score before it stops training
                train_generator -- the data generator used to train the model
                X_train -- the training dataset. not used for training directly, but for calculation of training score
                y_train -- the training labels for calculating traning score. 
                X_test -- the test dataset for calculating test scores
                y_test -- the test labels for calculating test scores

        """

        for run in range(nr_runs):
            print("\n running experiment",run+1, "of",nr_runs)
            model = self.model_creation()

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience,restore_best_weights=True)
            history = model.fit_generator(train_generator, epochs=epochs, validation_data=(X_test, y_test),callbacks = [early_stopping])

            y_pred = model.predict(X_train) > 0.1
            self.training_results.append(Evaluation_Metrics(y_train, y_pred))

            y_pred_test = model.predict(X_test) > 0.1
            self.test_results.append(Evaluation_Metrics(y_test, y_pred_test))


    def reset_model(self):
        """
            resets the model with random weights. This is done before each run.
            taken from here: https://github.com/keras-team/keras/issues/341
        """
        
        session = kbackend.get_session()
        for layer in self.model.layers: 
            for v in layer.__dict__:
                v_arg = getattr(layer,v)
                if hasattr(v_arg,'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)
                    #print('reinitializing layer {}.{}'.format(layer.name, v))

    def get_training_accuracies(self):
        """
            returns the accuracy scores of the training runs as list
        """
        return [metric.subset_accuracy for metric in self.training_results]
    
    def get_test_accuracies(self):
        """
            returns the accuracy scores of the test runs as list
        """
        return [metric.subset_accuracy for metric in self.test_results]

    def get_training_hamming_scores(self):
        """
            returns the hamming scores of the training runs as list
        """
        return [metric.hamming_score for metric in self.training_results]
    
    def get_test_hamming_scores(self):
        """
            returns the hamming scores of the test runs as list
        """
        return [metric.hamming_score for metric in self.test_results]


    def print_metrics(self):
        """
            prints the statistical measures of evaluation metrics to command line.
        """
        scheme = "{}:\t {} \t {} \t {} \t {}"

        print(scheme.format("score\t","minimum","maximum","mean","std. dev."))

        training_accuracy = Multi_Run_Evaluation.get_metrics_summary(self.get_training_accuracies())
        print(scheme.format(
                "acc_train",
                training_accuracy["min"],
                training_accuracy["max"],
                training_accuracy["mean"],
                training_accuracy["std_dev"]))
      
        training_hamming = Multi_Run_Evaluation.get_metrics_summary(self.get_training_hamming_scores())
        print(scheme.format(
                "hamming_train",
                training_hamming["min"],
                training_hamming["max"],
                training_hamming["mean"],
                training_hamming["std_dev"]))

        test_accuracy = Multi_Run_Evaluation.get_metrics_summary(self.get_test_accuracies())
        print(scheme.format(
                "acc_test",
                test_accuracy["min"],
                test_accuracy["max"],
                test_accuracy["mean"],
                test_accuracy["std_dev"]))

        test_hamming = Multi_Run_Evaluation.get_metrics_summary(self.get_test_hamming_scores())
        print(scheme.format(
                "hamming_test",
                test_hamming["min"],
                test_hamming["max"],
                test_hamming["mean"],
                test_hamming["std_dev"]))



    def get_metrics_summary(values):
        """
            returns a summary object of the minimum, maximum, mean and standard deviation of a list
        """
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "std_dev": statistics.stdev(values)
        }