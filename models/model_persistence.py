import os
import pathlib
import pickle
import tensorflow as tf
import typing

ROOT_DIR = pathlib.Path().parent.resolve()
MODEL_DIR = str(ROOT_DIR / pathlib.Path("models/snapshots/"))


def construct_filepath(base_dir: str,
                        model_dir_name: str,
                        model_fname: str,
                        extension: str ) -> pathlib.Path:
    """
        constructs the filepath for storing a specific model

        Parameters:
            base_dir -- The base directory of the path
            model_dir_name -- The directory to place the model in
            model_fname -- the filename of the model
            extension -- the extension of the model file
    """
    return pathlib.Path(base_dir) / pathlib.Path(model_dir_name) / pathlib.Path(model_fname + f"{extension}")


def save_model( ml_library: typing.Literal["tensorflow", "scikit-learn"],
                model: tf.keras.Model | typing.Any,
                model_name: str,
                model_version: int,
                model_dir_version: int,
                base_dir: str = MODEL_DIR ) -> None:
    """
        Stores a model in a file in the project folder.

        Parameters:
            ml_library -- a string representing which library te model comes from ("tensorflow" or "scikit-learn")
            model -- the model to store
            model_name -- the name this model is referred to
            model_version -- the version of the model
            model_dir_version -- a representation of the version for the model directory
            base_dir -- the directory to store the model in. If this is left blank, models will be stored in MODEL_DIR 
    """
    model_dir_name = model_name + "_V_" + str(model_dir_version)
    model_fname = model_name + "_v_" + str(model_version)
    model_dir_path = pathlib.Path(base_dir) / pathlib.Path(model_dir_name)
    if not os.path.exists(path=base_dir):
        os.mkdir(path=base_dir)
    if not os.path.exists(path=model_dir_path):
        os.mkdir(path=model_dir_path)

    print(f"Storing the model {model_name} in:\n{model_dir_path}")

    # use specific safe function in case of tensorflow
    if ml_library.lower() == "tensorflow":

        model_filepath = construct_filepath(
            base_dir=base_dir,
            model_dir_name=model_dir_name,
            model_fname=model_fname,
            extension=".h5"
        )
        model.save(filepath=model_filepath)

    # use pickle in case of scikit-learn or other libraries
    else:
        model_filepath = construct_filepath(
            base_dir=base_dir,
            model_dir_name=model_dir_name,
            model_fname=model_fname,
            extension=".pkl"
        )
        pickle.dump(model, open(file=model_filepath, mode="wb"))
    
    print("The model is successfully saved:\n")
    print(f"    - Model Name: {model_name}")
    print(f"    - Model Saving Path: {model_filepath}")


def load_model(ml_library: typing.Literal["tensorflow", "scikit-learn"],
                model_name: str,
                model_version: int,
                model_dir_version: int,
                base_dir: str = MODEL_DIR,
                custom_objects: dict | None = None ):
    """
        Loads a model from file. 

        Parameters:
            ml_library -- a string representing which library the model comes from ("tensorflow" or "scikit-learn")
            model_name -- the name this model is referred to
            model_version -- the version of the model
            model_dir_version -- a representation of the version for the model directory
            base_dir -- the directory to store the model in. If this is left blank, models will be stored in MODEL_DIR 
            custom_objects -- custom objects to load the model with in case of tensorflow.
    """
    model_dir_name = model_name + "_V_" + str(model_dir_version)
    model_fname = model_name + "_v_" + str(model_version)
    model_dir_path = pathlib.Path(base_dir) / pathlib.Path(model_dir_name)

    print(f"Loading {ml_library} model from:\n{model_dir_path}")
    
    if ml_library.lower() == "tensorflow":
        path = construct_filepath(
                base_dir=base_dir,
                model_dir_name=model_dir_name,
                model_fname=model_fname,
                extension=".h5"
            )
        return tf.keras.models.load_model( path, custom_objects=custom_objects)

    else:
        path = construct_filepath(
                base_dir=base_dir,
                model_dir_name=model_dir_name,
                model_fname=model_fname,
                extension=".pkl"
                )
        return pickle.load(file=open(path,mode="rb"))


