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
from .evaluation import classification_evaluation, regression_evaluation
from tqdm import tqdm

def train(model, dataset, epochs=200, loss_function=F.cross_entropy, optimizer=adam, scheduler=None, working_dir=None, device='cuda', validation=0.25, batch_size='all', use_target_index=False):
    if working_dir is not None:
        if working_dir.is_dir():
            pass
            #raise ValueError(f'working directory {working_dir} already exists, cant use.')
        else:
            os.mkdir(working_dir)
    else:
        working_dir = pl.Path(f'training_from_{datetime.now().strftime("%d%m%Y%H%M%S")}')
        os.mkdir(working_dir)
    ## Prepare Dataloaders for training and validation if required
    if validation is not None:
        labels = [data.rano.item() for data in dataset]
        train_idx, val_idx = train_test_split(
            range(len(dataset)), stratify=labels, test_size=validation)
        train_set = [dataset[i] for i in train_idx]
        batch_size = batch_size if isinstance(batch_size, int) else len(train_set)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_set = [dataset[i] for i in val_idx]
        val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=True)

        train_distribution = {k:v/len(train_idx) for k,v in Counter([data.rano.item() for data in train_set]).items()}
        print(f"""train set class distribution {train_distribution}""")
        test_distribution = {k:v/len(val_idx) for k,v in Counter([data.rano.item() for data in val_set]).items()}
        print(f"""validation set class distribution {test_distribution}""")

    else:
        train_set = [dataset[i] for i in train_idx]
        train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)

    ## Prepare class weight balancing weights
    labels = [data.rano.item() for data in train_set]
    label_counts = Counter(labels)
    num_classes = len(label_counts)  # or len(label_counts)
    counts = torch.tensor([label_counts[i] for i in range(num_classes)], dtype=torch.float)
    torch_weights = 1.0 / (counts + 1e-6)  # Avoid divide by zero
    torch_weights = (torch_weights / torch_weights.sum())  # Normalize (optional but nice)
    print("Balancing class weight in loss function using:",torch_weights)

    ## Set up Tracking variables
    best_model = None
    best_loss = np.inf
    train_losses = []
    
    val_losses = []

    ### Training and Validation loop
    model.train()
    for epoch in tqdm(range(epochs)):
        batch_losses=[]
        #print(f'== epoch {epoch}/{epochs}')
        # train step
        model.train()
        for batch in train_loader:
            size = batch.y.shape[1]
            ## current weight vector        assign the weight at [0]              pad 0 to ignore other features
            torch_weights_ = torch.tensor([[torch_weights.tolist()[sample.item()]]+[0]*(size-1) for sample in batch.rano], dtype=torch.float) if use_target_index else torch_weights # balance weights according to rano class even for regression
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch) 
            out = out[batch.target_index] if use_target_index else out # apply prediction mask to output, this basically selects the predicted value for the target node
            #print(out.shape, batch.y.shape, torch_weights_.shape)
            loss = loss_function(out, batch.y, weight=torch_weights_.to(device)) # compute loss with weights
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step()
        train_losses.append(sum(batch_losses)/len(batch_losses))
        # validation step
        if validation:
            model.eval()
            for batch in val_loader:
                torch_weights_ = torch.tensor([[torch_weights.tolist()[sample.item()]]+[0]*(size-1) for sample in batch.rano], dtype=torch.float) if use_target_index else torch_weights
                batch = batch.to(device)
                out = model(batch)
                out = out[batch.target_index] if use_target_index else out
                loss = loss_function(out, batch.y, weight=torch_weights_.to(device))
                val_losses.append(loss.item())
                if loss < best_loss:
                    best_loss=loss
                    best_model=copy.deepcopy(model)
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    filler = '' if validation is None else ' and validation'
    plt.title(f'Training{filler} loss per epoch')
    plt.savefig(working_dir/'training_losses.png')
    plt.close()
    plt.clf()
    return best_model, best_loss

def test_classification(model, dataset, device='cuda'):
    loader = DataLoader(dataset, batch_size=len(dataset))
    model.eval()
    for batch in loader:
        batch.to(device)
        out = model(batch)
        assignment = out.argmax(dim=1).cpu().numpy()
    return classification_evaluation(assignment, batch.y.cpu().numpy())

def test_regression(model, dataset, device='cuda'):
    loader = DataLoader(dataset, batch_size=len(dataset))
    model.eval()
    for batch in loader:
        batch.to(device)
        out = model(batch)
        assignment = out[batch.target_index].detach().cpu().numpy()
    return regression_evaluation(assignment, batch.y.cpu().numpy())
