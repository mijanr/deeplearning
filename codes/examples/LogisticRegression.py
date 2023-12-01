# necessary imports
import os
import sys
import git

repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Datasets import Datasets
from models import FeedForwardNN as ffnn

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

# load the dataset
task = "classification"
dataset_name = "iris"
split = 0.7
batch_size = 32

dataset = Datasets(task=task)
train_loader, test_loader = dataset.get_data(dataset_name=dataset_name, split=split, batch_size=batch_size)

# print the shapes of the data
data, target = next(iter(train_loader))
print(f"Data shape: {data.shape}")
print(f"Target shape: {target.shape}")

# define the model
input_dim = data.shape[1]
hidden_dim = 64
output_dim = len(torch.unique(target))
model = ffnn.FeedForwardNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=0.2)

# define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 500
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        # forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print the results
        if epoch % 10 == 0 and i == 0:
            print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# evaluate the model
with torch.no_grad():
    correct = 0
    total = 0
    y_pred = []
    y_test = []
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_pred.extend(predicted.cpu().numpy().tolist())
        y_test.extend(labels.cpu().numpy().tolist())
print(f"Test Accuracy: {100*correct/total:.2f}%")

# save the classification report 
save_path = repo_path + "/results/FeedForwardNN_iris.txt"
# classification report
cl_report = classification_report(y_test, y_pred)
# save the classification report
with open(save_path, "w") as f:
    f.write(cl_report)







