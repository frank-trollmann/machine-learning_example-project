import unittest
from data.dataset import Dataset


class TestDataset(unittest.TestCase):
    """
        tests the dataset class
    """

    def setUp(self):
        self.dataset = Dataset(test_mode = True)


    def test_download_and_remove(self):
        """
            test interaction between dataset download and check for downloading
        """
        # before download, dataset should not contain 
        assert self.dataset.is_downloaded() == False

        # check that after download data and images are available
        self.dataset.download()

        assert self.dataset.is_downloaded() == True
        assert self.dataset.get_original_data() is not None
        assert self.dataset.get_image(pokemon_name="pikachu") is not None
        assert self.dataset.get_image(pokemon_name="blacephalon") is not None
        assert self.dataset.get_image(pokemon_name="not a pokemon") is None
        assert len(self.dataset.get_labels()) == 18

        # check that remove_all correctly cleans this data
        self.dataset.remove_all()

        assert self.dataset.is_downloaded() == False


    def test_automatic_download(self):
        """
            test automatic download of get_original_data
        """
        # before download, dataset should not contain 
        assert self.dataset.is_downloaded() == False

        # check that get_original_data downloads correctly.
        data = self.dataset.get_original_data()

        assert self.dataset.is_downloaded() == True
        assert data is not None

        # check that remove_all correctly cleans this data
        self.dataset.remove_all()

        assert self.dataset.is_downloaded() == False


    def test_save_and_load_prepared_data(self):
        """
            tests interactions with processed data.
        """
        # check that data isn't there yet
        assert self.dataset.has_prepared_data() == False

        self.dataset.get_original_data()

        # store data and test that we can load it
        self.dataset.store_prepared_data(X="TestX", y="Testy")

        assert self.dataset.has_prepared_data() == True

        X, y = self.dataset.get_prepared_data()

        assert X == "TestX"
        assert y == "Testy"

        # check that remove_all correctly cleans this data
        self.dataset.remove_all()
        assert self.dataset.has_prepared_data() == False



    def tearDown(self):
        self.dataset.remove_all()

