import enum
import os
import pathlib
import pickle
import tensorflow as tf
import typing

ROOT_DIR = pathlib.Path().parent.resolve()
MODEL_DIR = str(ROOT_DIR / pathlib.Path("data/models/"))


def set_filepath(
    base_dir: str,
    model_dir_name: str,
    model_fname: str,
    extension: str
) -> pathlib.Path:
        return pathlib.Path(base_dir) / pathlib.Path(model_dir_name) / pathlib.Path(model_fname + f"{extension}")


def save_model(
    ml_library: typing.Literal["tensorflow", "scikit-learn"],
    model: tf.keras.Model | typing.Any,
    model_name: str,
    model_version: int,
    model_dir_version: int,
    base_dir: str = MODEL_DIR
) -> None:
    model_dir_name = model_name + "_V_" + str(model_dir_version)
    model_fname = model_name + "_v_" + str(model_version)
    model_dir_path = pathlib.Path(base_dir) / pathlib.Path(model_dir_name)

    print(f"Saving the new model {model_name} in:\n{model_dir_path}")

    if ml_library.lower() == "tensorflow":

        model_filepath = set_filepath(
            base_dir=base_dir,
            model_dir_name=model_dir_name,
            model_fname=model_fname,
            extension=".h5"
        )

        if os.path.exists(path=model_dir_path):
            model.save(filepath=model_filepath)
        else:
            os.mkdir(path=model_dir_path)
            model.save(filepath=model_filepath)

        print("The model is successfully saved:\n")
        print(f"    - Model Name: {model_name}")
        print(f"    - Model Saving Path: {model_filepath}")

        return

    elif ml_library.lower() == "scikit-learn":
        model_filepath = set_filepath(
            base_dir=base_dir,
            model_dir_name=model_dir_name,
            model_fname=model_fname,
            extension=".pkl"
        )

        if os.path.exists(path=model_dir_path):
            pickle.dump(model, open(file=model_filepath, mode="wb"))
        else:
            os.mkdir(path=model_dir_path)
            pickle.dump(model, open(file=model_filepath, mode="wb"))   # type: ignore

        print("The model is successfully saved:\n")
        print(f"    - Model Name: {model_name}")
        print(f"    - Model Saving Path: {model_filepath}")

        return

    raise Exception("Uuppss.. Nothing happens. Please re-check all your arguments!")


def load_model(
    ml_library: typing.Literal["tensorflow", "scikit-learn"],
    model_name: str,
    model_version: int,
    model_dir_version: int,
    base_dir: str = MODEL_DIR,
    custom_objects: dict | None = None
):
    model_dir_name = model_name + "_V_" + str(model_dir_version)
    model_fname = model_name + "_v_" + str(model_version)
    model_dir_path = pathlib.Path(base_dir) / pathlib.Path(model_dir_name)

    print(f"Loading {ml_library} model from:\n{model_dir_path}")

    if ml_library.lower() == "tensorflow":
        return tf.keras.models.load_model(
            filepath=set_filepath(
                base_dir=base_dir,
                model_dir_name=model_dir_name,
                model_fname=model_fname,
                extension=".h5"
            ),
            custom_objects=custom_objects
        )

    elif ml_library.lower() == "scikit-learn":
        return pickle.load(file=open(
            file=set_filepath(
                base_dir=base_dir,
                model_dir_name=model_dir_name,
                model_fname=model_fname,
                extension=".pkl"
                ),
                mode="rb"
            )
        )

    raise Exception("Uuppss.. Nothing happens. Please re-check all your arguments!")
