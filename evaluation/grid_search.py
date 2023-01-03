
import statistics 
import traceback
import pickle

import numpy as np

from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


from data.dataset import Dataset
from models.CNN_Builder import CNN_Builder
from evaluation.multi_run_evaluation import Multi_Run_Evaluation

# removing some deprecationwanings from output that disturb experiment report.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class Grid_Search:
    """
        This class implements grid search for CNN models. It enables testing different configurations of convolutional and fully connected layers. 
        
        Other training parameters are fixed based on previous experiments (see overfitting_experimentation.ipynb)
    """

    filename = "evaluation/experiment_records/grid_search.pickle"
    test_filename = "tests/grid_search.pickle"


    def __init__(self,in_shape, out_shape, convolutional_options, fully_connected_options, epochs = 100, nr_runs = 10,nr_splits = 10):
        """
            Configures the grid search.

            Each configuration will be tested for nr_runs * nr_splits times with max 100 epochs for each time.

            Parameters:
                in_shape -- the shape of the input of the network
                out_shape -- the shape of the output of the network
                convolutional_options -- A list of options for convolutional layers that are tested. This corresponds to the parameter "convolutional_layers" of the CNNBuilder
                fully_connected_options -- A list of options for fully connected layers that are tested. This corresponds to the parameter "convolutional_layers" of the CNNBuilder
                epochs -- nr of epochs to run
                nr_runs -- the number of runs to execute for each split
                nr_splits -- the number of splits for the dataset.
        """
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.convolutional_options = convolutional_options
        self.fully_connected_options = fully_connected_options

        self.epochs =  epochs
        self.nr_runs = nr_runs
        self.nr_splits = nr_splits

    def run(self, X,y):
        """
            Runs grid search.            
            The  search tests all combinations of convolutional and fully connected shapes. For each of them it performs 10-fold cross validation and uses the average hamming score of the test set over all splits as comparison score.

            Parameters:
                X -- the training data. 
                y -- labels of the training data.
            Returns:
                A list of all tried combinations and their scores as tuple (cnn_configuration, fully connected conciguration, score). The list is sorted by score.
        """

        print("\n\nrunning grid search...")
        kfold = KFold(n_splits = self.nr_splits, shuffle = True)

        # list of all tried configurations and scores. will be sorted and returned. 
        self.scores = []
        for cnn_config in self.convolutional_options:
            for fc_config in self.fully_connected_options:
                print("\nTesting configuration: ")
                print("  Convolutional part:",cnn_config)
                print("  Connected part: ",fc_config)

                test_hammings = []
                split_counter = 1

                for train_indices,test_indices in kfold.split(X):

                    try:
                        print("\n split {}/{}".format(split_counter,self.nr_splits))
                        split_counter+=1

                        X_train = X[train_indices]
                        y_train = y[train_indices]
                        X_test = X[test_indices]
                        y_test = y[test_indices]
                        train_datagen = ImageDataGenerator(
                                                rotation_range=30, 
                                                width_shift_range=0.1, 
                                                height_shift_range=0.1, 
                                                fill_mode="nearest")
                        train_generator = train_datagen.flow(X_train,y_train)

                        cnn_builder = CNN_Builder(convolutional_layers=cnn_config,
                                fully_connected_layers=fc_config,
                                in_shape=(self.in_shape),
                                out_shape=self.out_shape)


                        evaluator = Multi_Run_Evaluation(cnn_builder.build_model)
                        evaluator.evaluate( nr_runs=self.nr_runs, 
                                            epochs=self.epochs, 
                                            early_stopping_patience=5, 
                                            train_generator= train_generator, 
                                            X_train=X_train, 
                                            y_train=y_train, 
                                            X_test=X_test,
                                            y_test=y_test,
                                            verbose = 0)

                        mean_test_hamming = Multi_Run_Evaluation.get_metrics_summary(evaluator.get_test_hamming_scores())["mean"]
                        test_hammings.append(mean_test_hamming)

                    except Exception as e:
                        print("an exception occurred:")
                        traceback.print_exc()
                        print("resuming with next test")
                
                score = statistics.mean(test_hammings)
                print("score: ",score)

                self.scores.append((cnn_config,fc_config,score))
                self.scores.sort(key=lambda tup: tup[2], reverse = True)
                self.store_to_file()

        self.scores.sort(key=lambda tup: tup[2], reverse = True)
        return self.scores

    def store_to_file(self,test = False):
        """
            store the results of the last run into a file on hard drive
            
            Parameters:
                test -- Whether we are in test mode. In test mode the file path is changed to avoid overwriting actual results.
        """
        Grid_Search.store_results(self.scores,test)


    def store_results(results, test = False):
        """
            Class method for storing results to a file.

            Parameters:
                results -- the results to be stored
                test -- Whether we are in test mode. In test mode the file path is changed to avoid overwriting actual results.
        """
        filename = Grid_Search.filename
        if test:
            filename = Grid_Search.test_filename

        with open(filename,"wb") as f:
            pickle.dump(results,f,pickle.HIGHEST_PROTOCOL)
        
    def load_results(test = False):
        """
            Class method for loading results from a file.
            
            Parameters:
                test -- Whether we are in test mode. In test mode the file path is changed to avoid overwriting actual results.
        """
        filename = Grid_Search.filename
        if test:
            filename = Grid_Search.test_filename

        with open(filename,"rb") as f:
            results = pickle.load(f)
            return results


