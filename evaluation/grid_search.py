import statistics 
import traceback
import pickle

import warnings
import os
import numpy as np


from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator

from models.CNN_Builder import CNNBuilder
from evaluation.multi_run_evaluation import MultiRunEvaluation

# Removing some deprecationwanings from output that disturb experiment report.
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class GridSearch:
    """
        This class implements grid search for CNN models. It enables testing different configurations of convolutional and fully connected layers. 
        
        Other training parameters are fixed based on previous experiments (see overfitting_experimentation.ipynb)
    """

    filename = "evaluation/experiment_records/grid_search-{}_{}.pickle"
    test_filename = "tests/grid_search-{}_{}.pickle"


    def __init__(self,in_shape, out_shape, convolutional_options, fully_connected_options, epochs = 100, nr_runs = 10,nr_splits = 10,test = False):
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
        
        self.test = test

        self.scores = list()

    def run(self, X, y) -> list:
        """
            Runs grid search.            
            The  search tests all combinations of convolutional and fully connected shapes. For each of them it performs 10-fold cross validation and uses the average hamming score of the test set over all splits as comparison score.
            The search stores checkpoints in evaluation/experiment_records/grid_search.pickle

            Parameters:
                X -- the training data. 
                y -- labels of the training data.
            Returns:
                A list of all tried combinations and their scores as tuple (cnn_configuration, fully connected conciguration, score). The list is sorted by score.
        """
        print("\n\n RUNNING GRID SEARCH ...")
        results = {}

        checkpoint = self.load_checkpoint()
        print("checkpoint: ",checkpoint)
        if checkpoint is not None:
            results = checkpoint
            print("Existing checkpoint loaded with ", len(results), "entries")

        kfold = KFold(n_splits=self.nr_splits, shuffle=True)

        for cnn_config in self.convolutional_options:
            for fc_config in self.fully_connected_options:
                print("\nTesting configuration:")
                print(f"    - Convolutional part: {cnn_config}")
                print(f"    - Connected part: {fc_config}")


                # check to avoid retesting a tested combination:
                config_key = "("+str(cnn_config) + "," + str(fc_config) + ")"
                if config_key in results.keys():
                    print("Skipping - already tested in checkpoint")
                    continue

                test_hammings = []
                split_counter = 1

                for train_indices, test_indices in kfold.split(X=X):
                    try:
                        print(f"\nSplit {split_counter}/{self.nr_splits}")
                        split_counter+=1

                        X_train = X[train_indices]
                        y_train = y[train_indices]
                        X_test = X[test_indices]
                        y_test = y[test_indices]

                        train_datagen = ImageDataGenerator(
                            rotation_range=30, 
                            width_shift_range=0.1, 
                            height_shift_range=0.1, 
                            fill_mode="nearest"
                        )
                        train_generator = train_datagen.flow(X_train,y_train)

                        cnn_builder = CNNBuilder(
                            convolutional_layers=cnn_config,
                            fully_connected_layers=fc_config,
                            in_shape=(self.in_shape),
                            out_shape=self.out_shape
                        )

                        evaluator = MultiRunEvaluation(model_creation=cnn_builder.build_model)
                        evaluator.evaluate(
                            nr_runs=self.nr_runs,
                            epochs=self.epochs,
                            early_stopping_patience=5,
                            train_generator= train_generator,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            verbose = 0
                        )

                        mean_test_hamming = MultiRunEvaluation.get_metrics_summary(values=evaluator.get_test_hamming_scores())["mean"]
                        test_hammings.append(mean_test_hamming)

                        # cleanup. 
                        del X_train
                        del y_train
                        del X_test
                        del y_test
                        del train_datagen
                        del train_generator
                        del cnn_builder
                        del evaluator

                    except Exception as e:
                        print("An exception occurred:")
                        traceback.print_exc()
                        print("Resuming with next test...")

                score = statistics.mean(test_hammings)
                print(f"Score: {score}")
                
                # update results and make a checkpoint
                results[config_key] = (cnn_config,fc_config,score)
                self.store_checkpoint(results)
                

        scores = list(results.values())

        scores.sort(key=lambda tup: tup[2], reverse = True)
        return scores

    def store_checkpoint(self,checkpoint):
        """
            store a checkpint of the current results to a file where it can be loaded from if the experiment is interrupted
            
            Parameters:
                checkpoint -- The checkpoint data to be stored.
        """
        filename = GridSearch.filename.format(str(self.convolutional_options),str(self.fully_connected_options))
        if self.test:
            filename = GridSearch.test_filename.format(str(self.convolutional_options),str(self.fully_connected_options))
        
        with open(filename,"wb") as f:
            pickle.dump(checkpoint,f,pickle.HIGHEST_PROTOCOL)
    
    def load_checkpoint(self):
        """
            load the last checkpoint
        """
        filename = GridSearch.filename.format(str(self.convolutional_options),str(self.fully_connected_options))
        if self.test:
            filename = GridSearch.test_filename.format(str(self.convolutional_options),str(self.fully_connected_options))

        if os.path.exists(filename):
            with open(filename,"rb") as f:
                results = pickle.load(f)
                return results
        else:
            print("Checkpoint file " + filename + " does not exist") 
            return None

    def remove_checkpoint(self):
        """
            clean up the checkpoint data if it is not needed any more.
        """
        filename = GridSearch.filename.format(str(self.convolutional_options),str(self.fully_connected_options))
        if self.test:
            filename = GridSearch.test_filename.format(str(self.convolutional_options),str(self.fully_connected_options))
        
        if os.path.exists(filename):
            os.remove(filename)
