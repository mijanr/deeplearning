from typing import Tuple
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

class Datasets:
    def __init__(self, task:str="classification"):
        """
        task: str
            The type of task: classification or regression
        """
        self.task = task
    
    def get_classification_data(self, dataset_name:str="iris", split:float=0.7):
        # load the dataset
        try:
            iris = sns.load_dataset(dataset_name)
        except:
            raise Exception("Dataset not found")
        # encode the labels
        iris['species'] = iris['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})
        X = iris.drop('species', axis=1).values
        y = iris['species'].values
        # split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
        # convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        return X_train, X_test, y_train, y_test

    def get_data(self, dataset_name:str="iris", split:float=0.7):
        if self.task == "classification":
            return self.get_classification_data(dataset_name="iris", split=0.7)
        elif self.task == "regression":
            pass

if __name__ == "__main__":
    task = "classification"
    dataset_name = "iris"
    split = 0.7

    # load the dataset
    dataset = Datasets(task=task)
    X_train, X_test, y_train, y_test = dataset.get_data(dataset_name=dataset_name, split=split)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
