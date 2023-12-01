# necessary imports
import os
import sys
import git

repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Datasets import Datasets
from models import CNN as cnn

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

# load the dataset
task = "classification"
dataset_name = "mnist"
split = 0.7 # for mnist, split is not used
batch_size = 32

dataset = Datasets(task=task)
train_loader, test_loader = dataset.get_data(dataset_name=dataset_name, split=split, batch_size=batch_size)

# print the shapes of the data
data, target = next(iter(train_loader))
print(f"Data shape: {data.shape}")
print(f"Target shape: {target.shape}")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# define the model
in_ch = data.shape[1]
num_classes = len(torch.unique(target))
model = cnn.CNN(in_ch=in_ch, num_classes=num_classes).to(device)

# define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 5
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        # send the data to the device
        features = features.to(device)
        labels = labels.to(device)
        # print(features.shape)
        # print(labels.shape)

        # forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print the loss
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    for features, labels in test_loader:
        # send the data to the device
        features = features.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # append the labels
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(predicted.cpu().numpy().tolist())

print(f"Test accuracy of the model on the {total} test images: {100 * correct / total}%")

# save the classification report
save_path = repo_path + "/results/CNN_mnist.txt"
# classification report
cl_report = classification_report(y_true, y_pred)
with open(save_path, "w") as f:
    f.write(cl_report)