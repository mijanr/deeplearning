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
    
    def get_classification_data(self, dataset_name:str="iris", split:float=0.7, batch_size:int=32)->Tuple[DataLoader, DataLoader]:
        """
        dataset_name: str
            The name of the dataset
        split: float    
            The ratio of train and test data
        batch_size: int
            The batch size
        """
        
        if dataset_name == "iris":
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

            # datasets and dataloaders
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        elif dataset_name == "mnist":
            # load mnist dataset
            try:
                from torchvision.datasets import MNIST
            except:
                raise Exception("Dataset not found")
            
            # define the transforms
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            
            # load the dataset
            train_dataset = MNIST(root='data/', train=True, transform=transform, download=True) 
            test_dataset = MNIST(root='data/', train=False, transform=transform, download=True)
            
            # datasets and dataloaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def get_data(self, dataset_name:str="iris", split:float=0.7, batch_size:int=32)->Tuple[DataLoader, DataLoader]:
        """
        dataset_name: str
            The name of the dataset
        split: float
            The ratio of train and test data
        """
        if self.task == "classification":
            return self.get_classification_data(dataset_name=dataset_name, split=split, batch_size=batch_size)
        elif self.task == "regression":
            pass

if __name__ == "__main__":
    task = "classification"
    dataset_name = "mnist" # "mnist", "iris"
    split = 0.7
    batch_size = 32

    # load the dataset
    dataset = Datasets(task=task)
    train_loader, test_loader = dataset.get_data(dataset_name=dataset_name, split=split, batch_size=batch_size)
    
    # print shapes
    data, target = next(iter(train_loader))
    print(f"Data shape: {data.shape}")
    print(f"Target shape: {target.shape}")
