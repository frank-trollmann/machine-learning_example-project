import shutil
import pathlib
import unittest

from keras.models import Sequential
from keras.layers import (
    Input,
    Dense,
    Activation,
    Conv2D,
    MaxPooling2D,
    Flatten
) 

from data.dataset import Dataset
from helpers.models import save_model, load_model


class TestModelVersioning(unittest.TestCase):
    """
        tests the dataset class
    """

    def setUp(self):        
        self.model_name = "Sequential_CNN_Test_Model"
        self.model_version = 1
        self.model_dir_version = 1
        self.model = Sequential(name=self.model_name)
        self.dataset = Dataset()

        self.model.add(Input(shape=(4,)))
        self.model.add(Dense(2, activation="relu"))


    def test_save_model(self):
        """
            test interaction between dataset download and check for downloading
        """
        save_model(
            model=self.model,
            model_name=self.model_name,
            model_version=self.model_version,
            model_dir_version=self.model_dir_version
        )

        assert pathlib.Path(f"data/models/{self.model_name}_V_{self.model_dir_version}/{self.model_name}_v_{self.model_version}.h5").is_file() == True


    def test_save_same_model(self):
        save_model(
            model=self.model,
            model_name=self.model_name,
            model_version=self.model_version,
            model_dir_version=self.model_dir_version
        )
        assert pathlib.Path(f"data/models/{self.model_name}_V_{self.model_dir_version}/{self.model_name}_v_{self.model_version}.h5").is_file() == True


    def test_load_model(self):
        """
            test automatic download of get_original_data
        """
        save_model(
            model=self.model,
            model_name=self.model_name,
            model_version=self.model_version,
            model_dir_version=self.model_dir_version
        )

        test_model = load_model(
            model_dir_name="Sequential_CNN_Test_Model_V_1",
            model_name="Sequential_CNN_Test_Model_v_1"
        )

        assert test_model.name == self.model_name  # type: ignore
        assert isinstance(test_model, Sequential)


    def tearDown(self):
        shutil.rmtree(path=f"data/models/{self.model_name}_V_{self.model_dir_version}")
