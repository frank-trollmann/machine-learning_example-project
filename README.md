

# Machine Learning Example project

For most beginners, a machine learning project consists of a single Jupyter notebook. While this setup is great for experimenting with small scale machine learning projects, bigger projects require a more structured approach to machine learning code. 

This project serves as an example for a more complex machine learning project and how it can be set up. This is part of the learning unit XXX. 
Please feel free to use this project as a reference for your own machine learning project. 


## Machine Learning Problem

The problem adressed in this project is pokemon classification. This is based on a Kaggle Dataset (see [here](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)) that records each pokemon image and type (multiple are possible). Our goal is to learn a model that predicts the types of a pokemon based on it's image. This means our machine learning problem has the following form:
* Input: An image
* Output: A set of classes.

## How to set up?

This project is designed to be edited and run on your own computer. The machine learning problem is small enough that training should run on an average notebook in a reasonable amount of time. We recommend using Visual Studio Code as programming environment as it has good support for both Python files (.py) and Notebook files (.ipynb).

The recommended setup for this project is to be used in VSCode alongside a dedicated virtual environment. Below you can get information on how to connect everything correctly.


### Virtual Environment

Virtual environments are essential to deal with different versions of libraries. It is strongly recommended for you to create a virtual environment for this project. If you don't know how, you can find out in the [documentation](https://python.land/virtual-environments/virtualenv)).

Don't forget to activate the virtual environmet whenever you want to run code or manage dependencies.


### Project Dependencies

All the dependencies you need should be listed in requirements.txt. 

After creating a fresh environment you should activate it and make sure pip is up to date (*pip install --upgrade pip*)

Now you can install all dependencies with *pip install -r requirements.txt*. 

### Jupyter Notebook Plugin

VSCode has a plugin for Jupyter Notebooks. [Here](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) is the documentation that walks you through how to install it.

Note that you will have to put in some additional effort to connect your notebook with the virtual environment you have created. Jupyter is already installed as part of the requirements.txt, but you will need to install a Kernel that the Notebook plugin can use (details see [here](https://anbasile.github.io/posts/2017-06-25-jupyter-venv/)).  

This requires the following steps:
1) Install a kernel in your virtual environment by running *ipython kernel install --user --name=project_name*  feel free to give it a better name.
2) Make sure to restart VSCode to give it a chance to find the Kernel
3) Open a Juypter notebook and select te kernel in the top right. 

If you encounter any issues, or if the kernel doesn't appear, make sure you restarted VSCode (maybe get a coffee after you do. It took a few minutes for me to find the kernel). If that still didn't help, try substituting step 1 with this command: *python3 -m ipykernel install --user --name=projectname*. This tip is taken from [here](https://stackoverflow.com/questions/58119823/jupyter-notebooks-in-visual-studio-code-does-not-use-the-active-virtual-environm).

### Kaggle API

We will download our data from Kaggle via source code. For this to work you will need to be registered with Kaggle and download your API key configuration. This requires the following steps:
1) Make a free kaggle account.
2) generate an API key on *https://www.kaggle.com/(yourusername)/account*
3) This should automatically download a $kaggle.json$ with your API access. 
4) Place this file into your home directory into the $.kaggle$ folder

After completing these steps, the Kaggle API will find your access credentials and use them automatially when accessing datasets. The kaggle API is already installed from the requirements.txt


## Testing
The project comes with automated tests located in the folder "test". To run them execute *python -m unittest* in the main folder of this project.

## Notebook Dependencies

The main entry points of this project are jupyter notebooks. Each notebook has a specific purpose. Some notebooks depend on each other (e.g., the data preparation creates a dataset for machine learning and needs to be run before any notebook containing machine learning.)

Here is a summary of the notebooks and dependencies:
1) *data_exploration.ipynb*: This notebook aims to understand the dataset by exploring it.
2) *data_preparation.ipynb*: This notebook converts the initial dataset into X and y data for machine learning. This dataset needs to be run once for the following notebooks to function. 
3) *model_selection.ipynb*: This notebook runs experiments to determine which type of model is most suited to our problem.


## project structure

The project is structured as follows:
* *notebooks* contains the jupyter notebooks of this project
* *data* contains datasets and python code around loading and storing them.
* *models* contains code for creating machine learning models.
* *evaluation* contains code for collecting and evaluating evaluation metrics. It also contains notes on evaluation of specific experiments. 
* *test* contains automated tests for the python code in this project.



