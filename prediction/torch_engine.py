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
import csv

def train(model, dataset, epochs=200, loss_function=F.cross_entropy, optimizer=adam, scheduler=None, working_dir=None, device='cuda', validation=0.25, batch_size='all', use_target_index=False, verbose=False, weighted_loss=True):
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

    ## save experiment setup
    with open(working_dir/'setup.txt', 'w') as file:
        if validation:
            file.write(f"Training with validation, using {validation*100:.2f}% of training data as validation set\n")
            file.write(f"""train set class distribution {train_distribution}\n""")
            file.write(f"""validation set class distribution {test_distribution} \n""")
        if weighted_loss:
            file.write(f"""Balancing class weight in loss function using: {torch_weights}, weighted by rano class even if regression is task\n""")
        else:
            file.write("not using any weighting in loss\n")
        file.write(f"""Optimizer: {optimizer} \n""")
        file.write(f"""LR Scheduler: {scheduler} \n""")
        file.write(f"""Loss Function: {loss_function.__name__} \n""")
        file.write(f"""Epochs: {epochs} \n""")
        file.write(f"""Training Batche size: {batch_size} \n""")
        file.write(f"Model config: {model}")
        file.write(f"Paradigm: using {dataset.used_timepoints} to predict {dataset.target_name}")



    ## Set up Tracking variables
    best_model = None
    best_loss = np.inf
    train_losses = []
    val_losses = []

    ### Training and Validation loop
    for epoch in tqdm(range(epochs)):
        batch_losses=[]
        #print(f'== epoch {epoch}/{epochs}')
        # train step
        model.train()
        for batch in train_loader:
            
            ## current weight vector        assign the weight at [0]              pad 0 to ignore other features
            if weighted_loss: 
                size = batch.y.shape[0]
                torch_weights_ = torch.tensor([[torch_weights.tolist()[sample.item()]]+[0]*(size-1) for sample in batch.rano], dtype=torch.float).to(device) if use_target_index else torch_weights.to(device) # balance weights according to rano class even for regression
            else: torch_weights_ = None
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch) 
            out = out[batch.target_index] if use_target_index else out # apply prediction mask to output, this basically selects the predicted value for the target node
            #print(out.shape, batch.y.shape, torch_weights_.shape)
            loss = loss_function(out, batch.y, weight=torch_weights_) if weighted_loss else loss_function(out, batch.y) # compute loss with weights
            if verbose: print('Batch Loss:', loss.item())
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step()
        
        # loss tracking
        if verbose: print('Avg Epoch Loss:', sum(batch_losses)/len(batch_losses))
        train_losses.append(sum(batch_losses)/len(batch_losses))

        # validation step
        if validation:
            model.eval()
            for batch in val_loader:
                if weighted_loss: 
                    size=batch.y.shape[0]
                    torch_weights_ = torch.tensor([[torch_weights.tolist()[sample.item()]]+[0]*(size-1) for sample in batch.rano], dtype=torch.float).to(device) if use_target_index else torch_weights.to(device)
                else: torch_weights_ = None
                batch = batch.to(device)
                out = model(batch)
                out = out[batch.target_index] if use_target_index else out
                loss = loss_function(out, batch.y, weight=torch_weights_) if weighted_loss else loss_function(out, batch.y)
                if verbose: print('Validation loss:', loss.item())
                val_losses.append(loss.item())
                if loss < best_loss:
                    best_loss=loss
                    best_model=copy.deepcopy(model)

        if epoch%100==0: ## incremental reporting
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
            best_model.save(working_dir)
            
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
    best_model.save(working_dir)
    return best_model, best_loss

def test_classification(model, dataset, working_dir, device='cuda'):
    loader = DataLoader(dataset, batch_size=len(dataset))
    model.eval()
    with open(working_dir/'assignments.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=['prediction', 'target', 'confidence'], delimiter=';')
        writer.writeheader()
        for batch in loader:
            batch.to(device)
            y = batch.y.cpu().numpy()
            out = model(batch)
            confidence = out.detach.cpu().numpy()
            assignment = out.argmax(dim=1).cpu().numpy()
            for pd, gt, conf in zip(assignment, y, confidence):
                res = {'prediction': pd, 'target': gt, 'confidence':conf}
                writer.writerow(res)
    return classification_evaluation(assignment, y)

def test_regression(model, dataset, working_dir, device='cuda', regression_decoder_function=None):
    model.eval()
    assignment = []
    y = []
    with open(working_dir/'regressions.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=["prediction", "target", 'data_row', 'Lesion ID'])
        writer.writeheader()
        for sample in dataset:
            res = {}
            res["data_row"] = sample.row
            res["Lesion ID"] = sample.id
            decoder = sample.decoder
            target = eval(decoder.format(sample.y.item()))
            y.append([target])
            sample.to(device)
            out = model(sample)
            out = out[sample.target_index].detach().cpu().item()
            out = eval(decoder.format(out))
            assignment.append([out])
            if regression_decoder_function is not None: assignment = regression_decoder_function(assignment)
            res[f"prediction"] = out
            res[f"target"] = target
            writer.writerow(res)

    return regression_evaluation(assignment, y)
