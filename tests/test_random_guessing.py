import unittest
from models.Random_Guessing_Benchmark import RandomGuessingBenchmark


class TestRandomGuessing(unittest.TestCase):
    """
        tests the CNN builder
    """

    def setUp(self):
        self.labels = ["Red","Green","Blue"]
        self.X_test = [None]*10
        self.model = RandomGuessingBenchmark(self.labels)


    def test_dimensions(self):
        """
            tests that the the result has the right dimensions (number of samples, length of prediction vector)
        """
        results = self.model.predict(X=self.X_test)

        assert len(results) == len(self.X_test)
        assert len(results[0]) == len(self.labels)

    
    def test_results(self):
        """
            tests that results are always guessing exactly one class
        """

        results = self.model.predict(X=self.X_test)

        for result in results:
            assert sum(result) == 1



    def tearDown(self):
        pass
