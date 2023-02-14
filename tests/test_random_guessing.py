import unittest
from models.random_guessing_benchmark import RandomGuessingBenchmark


class TestRandomGuessing(unittest.TestCase):
    """
        tests the CNN builder
    """

    def setUp(self):
        self.labels = ["Red","Green","Blue"]
        self.X_test = [None]*10
        self.model = RandomGuessingBenchmark(self.labels)
        self.model.fit(X=None, y=None)  # type: ignore


    def test_dimensions(self):
        """
            tests that the the result has the right dimensions (number of samples, length of prediction vector)
        """
        results = self.model.predict(X=self.X_test) # type: ignore

        assert len(results) == len(self.X_test)
        assert len(results[0]) == len(self.labels)

    
    def test_results(self):
        """
            tests that results are always guessing exactly one class
        """

        results = self.model.predict(X=self.X_test) # type: ignore

        for result in results:
            assert sum(result) == 1



    def tearDown(self):
        pass
