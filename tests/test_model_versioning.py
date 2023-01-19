import shutil
import pathlib
import unittest
import os

from sklearn import tree
from keras.models import Sequential
from keras.layers import (
    Input,
    Dense
) 

from data.dataset import Dataset
from models.model_persistence import ROOT_DIR, save_model, load_model


class TestModelVersioning(unittest.TestCase):
    """
        tests the dataset class
    """

    def setUp(self):   
        self.model_dir = str(ROOT_DIR / pathlib.Path("tests/snapshots/"))        

        self.tf_model_name = "Sequential_CNN_TEST_Model"
        self.sk_model_name = "DTC_CreiterionGini_TEST_MODEL"
        self.model_version = 1
        self.model_dir_version = 1

        self.tf_model = Sequential(name=self.tf_model_name)
        self.tf_model.add(Input(shape=(4,)))
        self.tf_model.add(Dense(2, activation="relu"))
        self.tf_model.compile(loss="binary_crossentropy", optimizer="adam", run_eagerly=True)

        self.sk_model = tree.DecisionTreeClassifier()

        

    def test_save_sk_model(self):
        """
            test interaction between dataset download and check for downloading
        """
        save_model(
            ml_library="scikit-learn",
            model=self.sk_model,
            model_name=self.sk_model_name,
            model_version=1,
            model_dir_version=1,
            base_dir = self.model_dir
        )

        assert pathlib.Path(f"{self.model_dir}/{self.sk_model_name}_V_{self.model_dir_version}/{self.sk_model_name}_v_{self.model_version}.pkl").is_file() == True


    def test_save_tf_model(self):
        """
            test interaction between dataset download and check for downloading
        """
        save_model(
            ml_library="tensorflow",
            model=self.tf_model,
            model_name=self.tf_model_name,
            model_version=1,
            model_dir_version=1,
            base_dir = self.model_dir
        )

        assert pathlib.Path(f"{self.model_dir}/{self.tf_model_name}_V_{self.model_dir_version}/{self.tf_model_name}_v_{self.model_version}.h5").is_file() == True


    def test_overwriting_same_sk_model_with_same_filename(self):
        save_model(
            ml_library="scikit-learn",
            model=self.sk_model,
            model_name=self.sk_model_name,
            model_version=1,
            model_dir_version=1,
            base_dir = self.model_dir
        )
        save_model(
            ml_library="scikit-learn",
            model=self.sk_model,
            model_name=self.sk_model_name,
            model_version=1,
            model_dir_version=1,
            base_dir = self.model_dir
        )

        model_files = [model_file.name for model_file in pathlib.Path(f"{self.model_dir}/{self.sk_model_name}_V_{self.model_dir_version}").iterdir()]

        assert len(model_files) == 1


    def test_load_sk_model(self):
        """
            test automatic download of get_original_data
        """
        save_model(
            ml_library="scikit-learn",
            model=self.sk_model,
            model_name=self.sk_model_name,
            model_version=1,
            model_dir_version=1,
            base_dir = self.model_dir
        )

        sk_loaded_model = load_model(
            ml_library="scikit-learn",
            model_name=self.sk_model_name,
            model_version=1,
            model_dir_version=1,
            base_dir = self.model_dir
        )

        assert isinstance(sk_loaded_model, tree.BaseDecisionTree)
        assert sk_loaded_model.get_params() == self.sk_model.get_params()


    def test_load_tf_model(self):
        """
            test automatic download of get_original_data
        """
        save_model(
            ml_library="tensorflow",
            model=self.tf_model,
            model_name=self.tf_model_name,
            model_version=1,
            model_dir_version=1,
            base_dir = self.model_dir
        )

        tf_loaded_model = load_model(
            ml_library="tensorflow",
            model_name=self.tf_model_name,
            model_version=1,
            model_dir_version=1,
            base_dir = self.model_dir
        )

        assert isinstance(tf_loaded_model, Sequential)
        assert tf_loaded_model.name == self.tf_model_name


    def tearDown(self):
        if os.path.exists(path=self.model_dir):
            shutil.rmtree(path= pathlib.Path(self.model_dir))
