import shutil
import pathlib
import unittest

from sklearn import tree
from keras.models import Sequential
from keras.layers import (
    Input,
    Dense
) 

from data.dataset import Dataset
from helpers.models import MODEL_DIR, save_model, load_model


class TestModelVersioning(unittest.TestCase):
    """
        tests the dataset class
    """

    def setUp(self):        
        self.tf_model_name = "Sequential_CNN_TEST_Model"
        self.sk_model_name = "DTC_CreiterionGini_TEST_MODEL"
        self.model_version = 1
        self.model_dir_version = 1
        self.tf_model = Sequential(name=self.tf_model_name)
        self.sk_model = tree.DecisionTreeClassifier()

        self.tf_model.add(Input(shape=(4,)))
        self.tf_model.add(Dense(2, activation="relu"))
        self.tf_model.compile(loss="binary_crossentropy", optimizer="adam", run_eagerly=True)


    def test_save_sk_model(self):
        """
            test interaction between dataset download and check for downloading
        """
        save_model(
            ml_library="scikit-learn",
            model=self.sk_model,
            model_name=self.sk_model_name,
            model_version=1,
            model_dir_version=1
        )

        assert pathlib.Path(f"{MODEL_DIR}/{self.sk_model_name}_V_{self.model_dir_version}/{self.sk_model_name}_v_{self.model_version}.pkl").is_file() == True


    def test_save_tf_model(self):
        """
            test interaction between dataset download and check for downloading
        """
        save_model(
            ml_library="tensorflow",
            model=self.tf_model,
            model_name=self.tf_model_name,
            model_version=1,
            model_dir_version=1
        )

        assert pathlib.Path(f"{MODEL_DIR}/{self.tf_model_name}_V_{self.model_dir_version}/{self.tf_model_name}_v_{self.model_version}.h5").is_file() == True


    def test_overwriting_same_sk_model_with_same_filename(self):
        save_model(
            ml_library="scikit-learn",
            model=self.sk_model,
            model_name=self.sk_model_name,
            model_version=1,
            model_dir_version=1
        )
        save_model(
            ml_library="scikit-learn",
            model=self.sk_model,
            model_name=self.sk_model_name,
            model_version=1,
            model_dir_version=1
        )

        model_files = [model_file.name for model_file in pathlib.Path(f"{MODEL_DIR}/{self.sk_model_name}_V_{self.model_dir_version}").iterdir()]

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
            model_dir_version=1
        )

        sk_loaded_model = load_model(
            ml_library="scikit-learn",
            model_name=self.sk_model_name,
            model_version=1,
            model_dir_version=1
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
            model_dir_version=1
        )

        tf_loaded_model = load_model(
            ml_library="tensorflow",
            model_name=self.tf_model_name,
            model_version=1,
            model_dir_version=1
        )

        assert isinstance(tf_loaded_model, Sequential)
        assert tf_loaded_model.name == self.tf_model_name


    def tearDown(self):
        if pathlib.Path(f"{MODEL_DIR}/{self.sk_model_name}_V_{self.model_dir_version}/{self.sk_model_name}_v_{self.model_version}.pkl").is_file():
            shutil.rmtree(path=pathlib.Path(f"{MODEL_DIR}/{self.sk_model_name}_V_{self.model_dir_version}"))
        if pathlib.Path(f"{MODEL_DIR}/{self.tf_model_name}_V_{self.model_dir_version}/{self.tf_model_name}_v_{self.model_version}.h5").is_file():
            shutil.rmtree(path=pathlib.Path(f"{MODEL_DIR}/{self.tf_model_name}_V_{self.model_dir_version}"))
