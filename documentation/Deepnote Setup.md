# Setup Instructions - Deepnote
This document contains instructions on how to set up this project with Deepnote. We will discuss what Deepnote is and why it is useful and then continue to describe how to connect the different aspects of the project so that it’s notebooks and scripts can be run via Deepnote.

This is not a Deepnote tutorial – we will focus on the technical setup of the project and not explain the user interfaces of Deepnote in-depth.

## What is Deepnote? 
[Deepnote](http://Deepnote.com) is a browser-based programming environment that specializes on Machine Learning. It gives you access to a project running on a cloud server. The project contains a file system and has the ability to edit and run normal Python files as well as Python notebooks. 

At the time of writing this, Deepnote comes with a generous offer for teaching purposes of 16Gb RAM servers to be used free of charge. This can be used to run your Machine Learning experiments in the cloud, if you don’t want them to run on your local computer.  

## Connecting Git
Deepnote has git libraries installed. If you want to clone a project you can do so by creating a new terminal and cloning the project. From the command line you can use git as you would with any other command line.
It is strongly recommended to clone your repository directly into the main directory (called “work”) and not into a subdirectory. This will result in Deepnote being able to find your project files easily which helps it integrate with your project (e.g., it can find the requirements.txt file and install it directly).

## Python Version
The Test project uses language concepts from Python 3.10. You have to configure Deepnote to use this version of Python as otherwise you will get Python errors. You can do this in the settings under the point “Environment”.


## Installing Libraries
Deepnote comes with some machine learning libraries already installed. If you want to install others there are several ways to do so (documented [here](https://deepnote.com/docs/custom-environments)).

If you have cloned the repository into main folder of the project, then Deepnote will automatically install dependencies specified in the file requirements.txt. This happens on server startup, so if you find that after cloning the repository the libraries are not yet installed, you may have to restart the machine manually. 

## Connecting Kaggle
The Kaggle API requires to know your username and API key. To make these known within the Deepnote environment, you can use integrations. Unfortunately, there is no direct Kaggle integration but you can work around that with an environment variable integration. You need to define the following two environment variables:
-	KAGGLE_USERNAME: your username
-	KAGGLE_KEY: your API key.
Don’t forget to connect this integration with your project. Afterwards, you should be able to import the dataset without any issues. 


## Using Notebooks
Deepnote has it’s own environment for editing notebooks. When you open a notebook from the notebooks file, you can select “Move this file to notebooks” on the top of the screen. This will attach make the file appear in the “Notebooks” area where it will be editable and executable. 

Note that the working directory changing method we used in VSCode does not work on Deepnote. You have to manually set the working directory to the root directory of your project. There should be a small folder icon next to the notebook that will let you do this. 

Since we are using this way to change our working directory in Deepnote, you should remove the following part of the import cell of the notebook: 

> if not "working_directory_corrected" in vars():
>
> &nbsp;&nbsp;&nbsp;&nbsp;%cd ..
>
> &nbsp;&nbsp;&nbsp;&nbsp; working_directory_corrected = True
If you don't do this your interactions with the file system will not work correctly as Paths are resolved differently than in VSCode.


## Running Scripts
If you have Python scripts you need to run, you can do so via the command line. Deepnote lets you create your own terminals to get access to the linux bash on your virtual machine. In the exampe projects, scripts are contained in the folder "scripts". 

Kaggle enables you to create terminals in the terminals area. You can create one and use Python command line from there.

Unfortunately, Deepnote integrations don't extend to the command line, meaning you will have to set your KAGGLE_USERNAME and  KAGGLE_KEY manually using the export command.


