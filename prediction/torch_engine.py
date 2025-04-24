import torch.nn.functional as F
from torch.optim import adam
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
import numpy as np
import torch
from collections import Counter
import copy
import matplotlib.pyplot as plt
import pathlib as pl
from datetime import datetime
import os

def train(model, dataset, epochs=200, loss_function=F.cross_entropy, optimizer=adam, scheduler=None, working_dir=None, device='cuda', validation=0.25):
    if working_dir is not None:
        if working_dir.is_dir():
            raise ValueError(f'working directory {working_dir} already exists, cant use.')
        else:
            os.mkdir(working_dir)
    else:
        working_dir = pl.Path(f'training_from_{datetime.now().strftime("%d/%m/%Y%H:%M:%S")}')
        os.mkdir(working_dir)
    ## Prepare Dataloaders for training and validation if required
    if validation is not None:
        labels = [data.y.item() for data in dataset]
        train_idx, val_idx = train_test_split(
            range(len(dataset)), stratify=labels, test_size=0.2, random_state=42
        )
        train_set = [dataset[i] for i in train_idx]
        train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
        val_set = [dataset[i] for i in val_idx]
        val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=True)
    else:
        train_set = [dataset[i] for i in train_idx]
        train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)

    ## Prepare class weight balancing weights
    labels = [data.y.item() for data in train_set]
    label_counts = Counter(labels)
    num_classes = len(label_counts)  # or len(label_counts)
    counts = torch.tensor([label_counts[i] for i in range(num_classes)], dtype=torch.float)
    torch_weights = 1.0 / (counts + 1e-6)  # Avoid divide by zero
    torch_weights = torch_weights / torch_weights.sum()  # Normalize (optional but nice)

    ## Set up Tracking variables
    best_model = None
    best_loss = np.inf
    train_losses = []
    val_losses = []

    ### Training and Validation loop
    model.train()
    for epoch in range(epochs):
        print(f'== epoch {epoch}/{epochs}')
        # train step
        model.train()
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_function(out, batch.y, weight=torch_weights.to(device))
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step()
        # validation step
        if validation:
            model.eval()
            for batch in val_loader:
                batch.to(device)
                out = model(batch)
                loss = loss_function(out, batch.y, weight=torch_weights.to(device))
                val_losses.append(loss.item())
                if loss < best_loss:
                    best_loss=loss
                    best_model=copy.deepcopy(model)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    filler = '' if validation is None else ' and validation'
    plt.title(f'Training{filler} loss per epoch')
    plt.savefig(working_dir/'training_losses.png')
    return best_model, best_loss