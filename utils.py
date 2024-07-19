from torchvision import datasets, transforms
import torch
import random
import numpy as np

train_dataset = datasets.MNIST(root='./data', train=True, download=True,transform=transforms.ToTensor())
val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, criterion, optimizer, train_loader, device):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        # forward
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs, labels=labels)
        loss = outputs[0]
        loss = loss.mean()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)

def val_one_epoch(model, criterion, val_loader, device):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs[0], labels)
            epoch_loss += loss.item()

            _, predicted = torch.max(outputs[0], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = epoch_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy