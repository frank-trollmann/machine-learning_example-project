import unittest
import numpy as np
import os

from evaluation.grid_search import GridSearch



class TestGridSearch(unittest.TestCase):
    """
        Tests the class evaluation/grid_search.py
    """
    def setUp(self):
        # Run a grid search with random input and a very simple net.
        # Even in this simple form, this step will take a while.
        X = np.random.rand(10, 10, 10, 3)
        y = np.random.randint(0, 1, (20, 5))       
        self.grid_search = GridSearch(
            in_shape= (X.shape[1], X.shape[2], X.shape[3]),
            out_shape= y.shape[1],
            convolutional_options=[[1], [2]],
            fully_connected_options=[[2], [4]],
            epochs=5,
            nr_runs=2,
            nr_splits=2,
            test=True
                            )
        self.results = self.grid_search.run(X,y)


    def test_result_number_search(self):
        """
            test that the number of results is the same as the combination of 2 x 2 options.
        """
        assert len( self.results) == 4

    

    def test_checkpoint(self):
        """
            test that the last checkpoint exists and is complete (has 4 elements)
        """
        last_checkpoint = self.grid_search.load_checkpoint()
        assert len(last_checkpoint) == 4
    

    def tearDown(self):
        self.grid_search.remove_checkpoint()

