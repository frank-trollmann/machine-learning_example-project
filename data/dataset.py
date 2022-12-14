
from kaggle.api.kaggle_api_extended import KaggleApi

from zipfile import ZipFile
import os
import shutil

from PIL import Image
import pandas 
import numpy as np


class Dataset:
    """
        This class is a convenience wrapper for dataset functions. It taks care of interaction with the file system and downloading the dataset.

    """

    DATASET_DIR_PATH = "data/dataset/"
    ZIP_PATH = "data/"
    ZIP_NAME = "pokemon-images-and-types.zip"
    IMAGE_FOLDER = "data/dataset/images/images/"
    

    def __init__(self, test_mode = False):
        """
            constructor. 
            If test_case is true, this will use different folders so unit tests don't interfere with any downloaded data.
        """
        if test_mode:
            Dataset.DATASET_DIR_PATH = "tests/dataset/"
            Dataset.ZIP_PATH = "tests/"
            Dataset.IMAGE_FOLDER = "tests/dataset/images/images/"
        pass


    def download(self):
        """ download the dataset from Kaggle and unzip it to data/dataset"""

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('vishalsubbiah/pokemon-images-and-types', path=Dataset.ZIP_PATH)

        zip_file = ZipFile(Dataset.ZIP_PATH + Dataset.ZIP_NAME)
        zip_file = zip_file.extractall(path = Dataset.DATASET_DIR_PATH)

        os.remove(Dataset.ZIP_PATH + Dataset.ZIP_NAME)


    def is_downloaded(self):
        """ checks if the data alread has been downloaded """
        return os.path.exists(Dataset.DATASET_DIR_PATH + "pokemon.csv")


    def remove_all(self):
        """ removes all files associated with this dataset from the computer"""
        if os.path.exists(Dataset.ZIP_PATH + Dataset.ZIP_NAME):
            os.remove(Dataset.ZIP_PATH + Dataset.ZIP_NAME)
        if os.path.exists(Dataset.DATASET_DIR_PATH):
            shutil.rmtree(Dataset.DATASET_DIR_PATH)

    def get_original_data(self):
        """
            Loads the original pokemon.csv as pandas dataset.
            If the dataset is not yet loaded, it will be loaded automatically.
        """
        if not self.is_downloaded():
            self.download()
        
        return pandas.read_csv(Dataset.DATASET_DIR_PATH + "pokemon.csv")

    def get_image(self,pokemon_name):
        """
            loads the image of a pokemon.
            tests whether a corresponding image exists either as png or as jpg.
        """
        if not self.is_downloaded():
            self.download()
        
        png_name = Dataset.IMAGE_FOLDER + pokemon_name + ".png"
        jpg_name = Dataset.IMAGE_FOLDER + pokemon_name + ".jpg"
        if os.path.exists(png_name):
            img = Image.open(png_name)
            img = img.convert('RGBA')
            return img
        elif os.path.exists(jpg_name):
            return Image.open(jpg_name)
        else:
            return None

    def get_labels(self):
        """
            returns a list of all labels occurring in Type1 or Type.
            This should be used to provide a definite list and order of labels for all notebooks.
        """
        data = self.get_original_data()
        labels = list(np.unique(data["Type1"], return_counts=False))
        labels.extend(np.unique(data[data["Type2"].notnull()]["Type2"], return_counts=False))
        return sorted(set(labels))


    def store_prepared_data(self,X,y):
        """
            Stores the X and y dataset after data preparation on hard drive.
        """
        np.save(Dataset.DATASET_DIR_PATH + "prepared_X.npy",X)
        np.save(Dataset.DATASET_DIR_PATH + "prepared_y.npy",y)

    def has_prepared_data(self):
        """ Checks if prepared data is ready to be loaded. """
        return os.path.exists(Dataset.DATASET_DIR_PATH + "prepared_X.npy") and os.path.exists(Dataset.DATASET_DIR_PATH + "prepared_y.npy")

    def get_prepared_data(self):
        """
            Loads the prepared dataset from disk.
            Throws an exception if the data is not present on disk.
            Returns a tuple of X and y
        """
        if not self.has_prepared_data():
            raise Exception("Cannot load prepared data as it has not been stored. Please make sure to run data_preparation.ipynb before doing the machine learning.")
        
        X = np.load(Dataset.DATASET_DIR_PATH + "prepared_X.npy")
        y = np.load(Dataset.DATASET_DIR_PATH + "prepared_y.npy")
        return X, y
    
    