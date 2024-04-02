#!/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
torch
"""
import argparse
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
import torch
from torchvision import datasets
import torchvision.models as models


parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--model_name", type=str, default='alexnet', help="model name for GPU train."
)
args = parser.parse_args()

models_zoo = {'resnet18': models.resnet18,
              'alexnet': models.alexnet,
              'squeezenet1_0': models.squeezenet1_0,
              'resnet34': models.resnet34,
              'resnet101': models.resnet101,
              'vgg11': models.vgg11,
              'vgg11_bn': models.vgg11_bn,
              'vgg16': models.vgg16}

if __name__ == "__main__":
    model = models_zoo[args.model_name]
    model = model()
    learning_rate = 1e-3
    batch_size = 2
    epochs = 1
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    training_data = Subset(training_data, range(0, 400))
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_data = Subset(test_data, range(0, 80))
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
