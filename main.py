import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import random_split, DataLoader, TensorDataset

from tqdm import trange


MNIST_ROOT = "/checkpoint/wesbz/Datasets"
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
batch_size = 128

torch.manual_seed(seed)

transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

mnist = MNIST(MNIST_ROOT, train=True, download=True, transform=transform)
transform_loader = DataLoader(mnist, batch_size=60_000, shuffle=False)
tfmd_data = next(iter(transform_loader))
mnist_dataset = TensorDataset(tfmd_data[0].to(device), tfmd_data[1].to(device))

train_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

mnist_test = MNIST(MNIST_ROOT, train=False, download=True, transform=transform)
transform_loader = DataLoader(mnist_test, batch_size=10_000, shuffle=False)
tfmd_data = next(iter(transform_loader))
test_data = TensorDataset(tfmd_data[0].to(device), tfmd_data[1].to(device))
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


torch.manual_seed(seed)
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=2, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(in_features=1152, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=10)
)

model.to(device)


lr = 1e-1
epochs = 20
# Training
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 / (epoch+1))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)

hist_train_acc = []
hist_train_loss = []

with trange(epochs) as t:
    for _ in t:

        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = y_pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_loss /= len(train_loader)

        # print(f"Train accuarcy: {100. * correct / total:.3f}%")
        train_acc = 100. * correct / total
        hist_train_acc.append(100. * correct / total)
        hist_train_loss.append(train_loss)

        # scheduler.step()

        t.set_postfix(train_acc=train_acc)

test_loss = 0.0
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = criterion(y_pred, y)

        test_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

test_loss /= len(test_loader)

print(f"Test accuracy: {100. * correct / total}%")