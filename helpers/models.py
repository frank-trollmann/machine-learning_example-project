import os
import pathlib
import shutil
import tensorflow as tf

ROOT_DIR = pathlib.Path().parent.resolve()
MODEL_DIR = ROOT_DIR / pathlib.Path("data/models/")


def save_model(
    model: tf.keras.Model,
    model_name: str,
    model_version: int,
    model_dir_version: int
) -> None:
    fname = model_name + "_v_" + str(model_version) + ".h5"
    dir_name = model_name + "_V_" + str(model_dir_version)
    model_dir_path = MODEL_DIR / pathlib.Path(dir_name)
    model_filepath = model_dir_path / pathlib.Path(fname)

    print(f"Saving the new model {model_name} in {model_dir_path}. . .")

    if os.path.exists(path=model_dir_path):
        model.save(filepath=model_filepath)
    else:
        os.mkdir(path=model_dir_path)
        model.save(filepath=model_filepath)

    print("The model is successfully saved:\n")
    print(f"    - Model Name: {model_name}")
    print(f"    - Model Saving Path: {model_filepath}")

    return


def load_model(
    model_dir_name: str,
    model_name: str,
    custom_objects: dict | None = None
):
    return tf.keras.models.load_model(
        filepath=MODEL_DIR / pathlib.Path(model_dir_name) / pathlib.Path(model_name + ".h5"),
        custom_objects=custom_objects
    )
