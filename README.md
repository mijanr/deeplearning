# Deep Learning
This repository contains the implementations of various deep learning algorithms and projects that I have worked on.

## Contents
- [Codes](codes)
    - [Datasets](codes/Datasets.py)
    - [Models](codes/models)
        - [Feed Forward NN](codes/models/FeedForwardNN.py)
        - [Convolutional Neural Network](codes/models/CNN.py)
    - [Examples](codes/examples)
        - [Linear Regression](codes/examples/LogisticRegression.py)
        - [Multiclass Classification (CNN)](codes/examples/MultiClassClassification_CNN.py)


## Requirements
`requirements.yml` file contains the list of all the packages required to run the code in this repository. requirements.yml is generated using the following command:
```
conda env export --no-builds | grep -v "prefix" > requirements.yml
```
To create a conda environment using the `requirements.yml` file, run the following command:
```
conda env create -f requirements.yml
```