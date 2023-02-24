import torch
import argparse
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

from dataloader import train_loader, valid_loader
from utils.save_model import save_model
from utils.save_plots import save_plots

import numpy as np
import pandas as pd
import os
import glob
from tqdm.auto import tqdm
import time


# ------------ Argument Parser ------------ #
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=20, help='Epochs to train the network')
args = vars(parser.parse_args())

# ------------ Set Parameters ------------ #
lr = 1e-3
epochs = args['epochs']
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computing on : {device}\n")

# ------------ Initialize the VGG16 model ------------ #
model = torchvision.models.vgg16(pretrained=True)
model.to(device)
print(f"Employing generic VGG16 model as our backbone.\n")
print(model)

# ------------ Parameters count ------------
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters : {total_params}\n")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Training parameters : {trainable_params}\n")

# ------------ Define Optimizer ------------
optimizer = optim.Adam(model.parameters(), lr=lr)

# ------------ Define Loss Function ------------
ce_loss = nn.CrossEntropyLoss()


def compute_loss(pred, target):
    loss = ce_loss(pred, target)

    return loss


# ------------ Training Phase ------------
def train(model, train_loader, optimizer):

    model.train()
    print(f"Training Phase begins!")

    curr_train_loss = 0.0
    correct_preds = 0
    counter = 0

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1

        image, labels = data
        image = image.to(device)
        labels = labels.to(device)

        # zeroing the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(image)

        # now, calculate loss
        loss = compute_loss(outputs, labels)
        curr_train_loss += loss.item()

        # and, calculate accuracy
        _, preds = torch.max(outputs.data, 1)
        correct_preds += (preds == labels).sum().item()

        # backpropagation
        loss.backward()

        # update the optimizer parameters
        optimizer.step()

    # loss and accuracy for the complete epoch
    epoch_loss = curr_train_loss / counter
    epoch_acc = 100. * (correct_preds / len(train_loader.dataset))
    return epoch_loss, epoch_acc


# ------------ Validation Phase ------------
def validate(model, test_loader):
    # Evaluation mode
    model.eval()
    print('Validation Phase!')

    curr_valid_loss = 0.0
    correct_preds = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(image)

            # compute loss
            loss = compute_loss(outputs, labels)
            curr_valid_loss += loss.item()

            # calculate accuracy
            _, preds = torch.max(outputs.data, 1)
            correct_preds += (preds == labels).sum().item()

    # loss and accuracy for the complete epoch
    epoch_loss = curr_valid_loss / counter
    epoch_acc = 100. * (correct_preds / len(test_loader.dataset))
    return epoch_loss, epoch_acc


# Losses and Accuracies for every epoch
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

for epoch in range(epochs):
    print(f"Progress: Epoch {epoch+1} / {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)

    print(f"Training loss: {train_epoch_loss:.4f}, training acc: {train_epoch_acc:.4f}")
    print(f"Validation loss: {valid_epoch_loss:.4f}, validation acc: {valid_epoch_acc:.4f}")
    print('-'*50)
    time.sleep(5)

    if (epoch+1) % 10 == 0:
        # Save the model weights
        save_model(epoch, model, optimizer, ce_loss)

# save the loss and accuracy plots
save_plots(train_acc, valid_acc, train_loss, valid_loss)
print('-'*50)
print('Training is complete')


