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
environment.yml file contains the list of all the required packages. You can create a conda environment using the following command:
```bash
conda env create -f environment.yml
```

To update the environment with the latest packages, you can run the following command:
```bash
conda env update -f environment.yml
```

Activate the environment using the following command:
```bash
conda activate deepl
```