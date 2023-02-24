<h1 align=center><strong>Machine Learning Example project</strong></h1>

For most beginners, a machine learning project consists of a single Python notebook. While this setup is great for experimenting with small scale Machine Learning projects, bigger projects require a more structured approach to remain understandable and maintainable. In this regard, Machine Learning is not too different from traditional normal Software Engineering.

This test project has been created for the learning unit "Complex Machine Learning Projects" at CODE University of applied sciences. It's purpose it to be a showcase for how a Machine Learning project can be structured. The problem we tackle with this project has been selected with the requirement that certain experiments can be run in a university classroom setting and thus may seem like it is too simply to warrant some of the more elaborate structures and tests we use. This is onpurpose, as this project is more about showing off processes and structures than it is about the specific Machine Learning problem. 

While most of the project should stand as a reference on it's own, we have included the commented slidesets of the Learning Unit in the documentation folder. 

# Machine Learning Problem

The problem adressed in this project is Pokémon classification. This is based on a Kaggle Dataset (see [here](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)) that records one image and up to two types per Pokémon. Our goal is to learn a model that predicts the types of a Pokémon based on it's image. This is a multi-label classification problem of the following form:
* Input: An image
* Output: A set of classes.

# Project Structure

Here we describe the structure of the project. 

## Folder Structure

The project is structured along the following folders:
* *data*:  code for handling data (storing, loading, processing), as well as the dataset itself (after it is downloaded).
* *models*: code for handling models (creating, storing, loading) as well as the stored models.
* *evaluation*: code for evaluating a model
* *notebooks:* Python notebooks
* *scripts:* executable scripts
* *tests:* automated tests for the Python code in this project.


## Notebooks

The main entry points of this project are Python notebooks. Each notebook has a specific purpose. Some notebooks depend on each other (e.g., the data preparation creates a dataset for machine learning and needs to be run before any notebook containing machine learning.)

Here is a summary of the notebooks and dependencies:
1) *data_exploration.ipynb*: aims to understand the dataset by exploring it.
2) *data_preparation.ipynb*: converts the initial dataset into X and y data for machine learning. This dataset needs to be run once for the following notebooks to function. 
3) *model_selection.ipynb*: runs experiments to determine which type of model is most suited to our problem.
4) *overfitting_experimentation.ipynb*: experiments with several ways to reduce overfitting in order to determine which are effective on our problem.
5) *hyper_parameter_selection.ipynb*: evaluates Grid Search results to determine the best hyper-parameters for our problem.
6) *manuael_model_selection.ipynb*: an experimentation notebook that can be executed to test different model configurations.
7) *evaluation.ipyb*: evaluates the model found by hyper-paratemer selection.

## Scripts

The project contains one Script: *hyper_parameter_selection.py*. This script runs grid search on the structure of a convolutional neural network. The results are written into a .pickle file in the experiment records. 

# Project Setup

This project requires Python 3.10 and is designed to be edited and run on your own computer. The machine learning problem is small enough that training should run on an average notebook in a reasonable amount of time. We recommend using Visual Studio Code as programming environment as it has good support for both Python files (.py) and Notebook files (.ipynb).

The recommended setup for this project is to be used in VSCode alongside a dedicated virtual environment. Below you can get information on how to connect everything correctly.

## Virtual Environment

Virtual environments are essential to deal with different versions of libraries. It is strongly recommended for you to create a virtual environment for this project. If you don't know how, you can find out in the [documentation](https://python.land/virtual-environments/virtualenv)).

Please make sure to create a virtual environment with Python 3.10 and don't forget to activate the virtual environmet whenever you want to run code or manage dependencies.

## Project Dependencies

All the dependencies you need should be listed in requirements.txt. 

After creating a fresh environment you should activate it and make sure pip is up to date (*pip install --upgrade pip*)

Now you can install all dependencies with *pip install -r requirements.txt*. 

## Jupyter Notebook Plugin

VSCode has a plugin for Jupyter Notebooks. [Here](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) is the documentation that walks you through how to install it.

Note that you will have to put in some additional effort to connect your notebook with the virtual environment you have created. Jupyter is already installed as part of the requirements.txt, but you will need to install a Kernel that the Notebook plugin can use (details see [here](https://anbasile.github.io/posts/2017-06-25-jupyter-venv/)).  

This requires the following steps:
1) Install a kernel in your virtual environment by running *ipython kernel install --user --name=project_name*  feel free to give it a better name.
2) Make sure to restart VSCode to give it a chance to find the Kernel
3) Open a Juypter notebook and select the kernel in the top right. 

If you encounter any issues, or if the kernel doesn't appear, make sure you restarted VSCode (maybe get a coffee after you do. It took a few minutes for me to find the kernel). If that still didn't help, try substituting step 1 with this command: *python3 -m ipykernel install --user --name=projectname*. This tip is taken from [here](https://stackoverflow.com/questions/58119823/jupyter-notebooks-in-visual-studio-code-does-not-use-the-active-virtual-environm).

## Kaggle API

We will download our data from Kaggle via source code. For this to work you will need to be registered with Kaggle and download your API key configuration. This requires the following steps:
1) Make a free kaggle account.
2) generate an API key on *https://www.kaggle.com/(yourusername)/account*
3) This should automatically download a $kaggle.json$ with your API access. 
4) Place this file into your home directory into the $.kaggle$ folder.
5) (Not for Windows users!) For security reason, please run from your terminal: `chmod 600 ~/.kaggle/kaggle.json`. This command ensures that your Kaggle API secured and unreadable by others.

After completing these steps, the Kaggle API will find your access credentials and use them automatially when accessing datasets. The kaggle API is already installed from the requirements.txt

## Testing
The project comes with automated tests located in the folder "test". To run them execute *python -m unittest* in the main folder of this project.

## Setup in other environments.

The documentation folder contains additional direction on how to set up this project with environments other than VSCode. The folder contains documents for the following environments:
- Deepnote - [link](documentation/Deepnote%20Setup.md)

