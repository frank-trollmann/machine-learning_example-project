import numpy as np
from sklearn.model_selection import train_test_split

from data.dataset import Dataset
from evaluation.grid_search import GridSearch

"""
    This script runs grid search for the hyper parameter selection and stores the result in file "evaluation/experiment_records/grid_search.pickle".

    Since this script is not in the main folder, it needs to be called as a module to resolve it's dependencies.
    Call it with: python -m scripts.hyper_parameter_selection.

    This script assumes data has already been prepared. It will not run unless you execute data_preparation.ipynb at least once to create and store the prepared dataset.

"""

def main():
    dataset = Dataset()

    X, y = dataset.get_prepared_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 7)
    y_train = np.array(object=[y_train[i] for i in range(len(y_train))])

    grid_search = GridSearch(
        in_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),  # type: ignore
        out_shape=y_train.shape[1],
        convolutional_options=[
            [8],
            [16],
            [32],
            [64],
            [8,4],
            [16,8],
            [32,16],
            [64,32]
        ],
        fully_connected_options=[
            [10],
            [25],
            [50],
            [100],
            [250],
            [500],
            [1000],
            [10,10],
            [25,25],
            [50,50],
            [100,100],
            [250,250],
            [500,500],
            [1000,1000]
        ],
        epochs = 100, 
        nr_runs = 5,
        nr_splits = 5 
    )
    grid_search.run(X=X_train, y=y_train)



if __name__ == "__main__":
    main()
