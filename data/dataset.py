import os
import shutil
import pandas as pd
import numpy as np
import typing

from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
from zipfile import ZipFile


class Dataset:
    """
        This class is a convenience wrapper for dataset functions. It taks care of interaction with the file system and downloading the dataset.

    """

    def __init__(self, test_mode = False):
        """
            constructor. 
            If test_case is true, this will use different folders so unit tests don't interfere with any downloaded data.
        """

        self.DATASET_DIR_PATH = "data/dataset/"
        self.ZIP_PATH = "data/"
        self.ZIP_NAME = "pokemon-images-and-types.zip"
        self.IMAGE_FOLDER = "data/dataset/images/images/"

        if test_mode:
            self.DATASET_DIR_PATH = "tests/dataset/"
            self.ZIP_PATH = "tests/"
            self.IMAGE_FOLDER = "tests/dataset/images/images/"


    def download(self) -> None:
        """ download the dataset from Kaggle and unzip it to data/dataset"""

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('vishalsubbiah/pokemon-images-and-types', path=self.ZIP_PATH)

        zip_file = ZipFile(self.ZIP_PATH + self.ZIP_NAME)
        zip_file = zip_file.extractall(path = self.DATASET_DIR_PATH)

        os.remove(self.ZIP_PATH + self.ZIP_NAME)


    def is_downloaded(self) -> bool:
        """ checks if the data alread has been downloaded """

        return os.path.exists(self.DATASET_DIR_PATH + "pokemon.csv")


    def remove_all(self) -> None:
        """ removes all files associated with this dataset from the computer"""

        if os.path.exists(self.ZIP_PATH + self.ZIP_NAME):
            os.remove(self.ZIP_PATH + self.ZIP_NAME)
        if os.path.exists(self.DATASET_DIR_PATH):
            shutil.rmtree(self.DATASET_DIR_PATH)


    def get_original_data(self) -> pd.DataFrame:
        """
            Loads the original pokemon.csv as pandas self.
            If the dataset is not yet loaded, it will be loaded automatically.
        """

        if not self.is_downloaded():
            self.download()
        
        return pd.read_csv(self.DATASET_DIR_PATH + "pokemon.csv")


    def get_image(self, pokemon_name: str) -> typing.Any:
        """
            loads the image of a pokemon.
            tests whether a corresponding image exists either as png or as jpg.
        """

        if not self.is_downloaded():
            self.download()
        
        pokemon_png_path = self.IMAGE_FOLDER + pokemon_name + ".png"
        pokemon_jpg_path = self.IMAGE_FOLDER + pokemon_name + ".jpg"
        if os.path.exists(path=pokemon_png_path):
            img = Image.open(fp=pokemon_png_path)
            img = img.convert(mode="RGBA")
            return img
        elif os.path.exists(path=pokemon_jpg_path):
            return Image.open(fp=pokemon_jpg_path)
        else:
            return None


    def get_labels(self) -> list[str]:
        """
            returns a list of all labels occurring in Type1 or Type.
            This should be used to provide a definite list and order of labels for all notebooks.
        """

        data = self.get_original_data()
        labels = list(np.unique(data["Type1"], return_counts=False))

        labels.extend(np.unique(data[data["Type2"].notnull()]["Type2"], return_counts=False))

        return sorted(set(labels))


    def store_prepared_data(self, X: np.ndarray | str, y: np.ndarray | str) -> None:
        """
            Stores the X and y dataset after data preparation on hard drive.
        """

        np.save(self.DATASET_DIR_PATH + "prepared_X.npy", X)
        np.save(self.DATASET_DIR_PATH + "prepared_y.npy", y)


    def has_prepared_data(self) -> bool:
        """ Checks if prepared data is ready to be loaded. """

        return os.path.exists(self.DATASET_DIR_PATH + "prepared_X.npy") and os.path.exists(self.DATASET_DIR_PATH + "prepared_y.npy")


    def get_prepared_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
            Loads the prepared dataset from disk.
            Throws an exception if the data is not present on disk.
            Returns a tuple of X and y
        """

        if not self.has_prepared_data():
            raise Exception("Cannot load prepared data as it has not been stored. Please make sure to run data_preparation.ipynb before doing the machine learning.")
        
        X = np.load(self.DATASET_DIR_PATH + "prepared_X.npy")
        y = np.load(self.DATASET_DIR_PATH + "prepared_y.npy")

        return X, y
