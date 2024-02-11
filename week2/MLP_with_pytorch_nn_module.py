import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm


class MNIST_Logistic_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 500)
        # self.dropout = nn.Dropout(0.2)
        self.lin2 = nn.Linear(500, 10)

    def forward(self, x):
        a1 = self.lin1(x)
        a2 = F.relu(a1)
        # a2 = self.dropout(a2)
        a3 = self.lin2(a2)

        return a3


# Load the data
mnist_train = datasets.MNIST(
    root="./datasets", train=True, transform=transforms.ToTensor(), download=True
)
mnist_test = datasets.MNIST(
    root="./datasets", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Training
# Instantiate model
model = MNIST_Logistic_Regression().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Iterate through train set minibatchs
for images, labels in tqdm(train_loader):
    # Move data to device
    images = images.to(device)
    labels = labels.to(device)

    # Zero out the gradients
    optimizer.zero_grad()

    # Forward pass
    x = images.view(-1, 28 * 28)
    y = model(x)
    loss = criterion(y, labels)
    # Backward pass
    loss.backward()
    optimizer.step()

## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs
    for images, labels in tqdm(test_loader):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        x = images.view(-1, 28 * 28)
        y = model(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print("Test accuracy: {}".format(correct / total))
