import unittest
import numpy as np
import os

from data.dataset import Dataset
from evaluation.grid_search import Grid_Search



class TestGridSearch(unittest.TestCase):
    """
        tests the class evaluation/grid_search.py
    """
    def setUp(self):
        # run a grid search with random input and a very simple net.
        # Even in this simple form, this step will take a while.
        X = np.random.rand(10,10,10,3)
        y = np.random.randint(0,1,(20,5))       
        self.grid_search = Grid_Search(
                                in_shape = (X.shape[1],X.shape[2],X.shape[3]),
                                out_shape = y.shape[1],
                                convolutional_options = [ [1],
                                                          [2]], 
                                fully_connected_options = [ [2],
                                                            [4]], 
                                epochs = 5, 
                                nr_runs = 2,
                                nr_splits=2
                            )
        self.grid_search.run(X,y)

    def test_result_number_search(self):
        """
            test that the number of results is the same as the combination of 2 x 2 options.
        """
        assert len(self.grid_search.scores) == 4
    

    def test_file_storage(self):
        """
            test that storing to file and loading from file works
        """
        self.grid_search.store_to_file(test=True)
        results = Grid_Search.load_results(test=True)

        self.assertCountEqual(self.grid_search.scores,results)

        os.remove(Grid_Search.test_filename)



    def tearDown(self):
        pass

