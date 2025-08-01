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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

def train(model, dataset, epochs=200, loss_function=F.cross_entropy, optimizer=adam, scheduler=None, working_dir=None, device='cuda', validation=0.25, batch_size='all', verbose=False, weighted_loss=True, patience=20):
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
            range(len(dataset)), stratify=labels, test_size=validation, random_state=42)
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
    if weighted_loss: print("Balancing class weight in loss function using:",torch_weights)

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
        if hasattr(dataset, 'used_timepoints'): file.write(f"Paradigm: using {dataset.used_timepoints} to predict {dataset.target_name}")
        elif hasattr(dataset, 'used_timedelta'): file.write(f"Paradigm: using noisy data with a cutoff time at {dataset.used_timedelta} days after treatment to predict {dataset.target_name}, at 360 days after treatment")



    ## Set up Tracking variables
    best_model = None
    best_loss = np.inf
    train_losses = []
    val_losses = []
    pat = 0
    ### Training and Validation loop
    for epoch in tqdm(range(epochs)):
        if pat == patience:
            print(f"{patience} epochs passed without improvement, terminating training.")
            break
        batch_losses=[]
        #print(f'== epoch {epoch}/{epochs}')
        # train step
        model.train()
        for batch in train_loader:
            ## current weight vector        assign the weight at [0]              pad 0 to ignore other features
            if weighted_loss and loss_function.__name__ != 'cross_entropy': 
                    size=batch.y.shape[0]
                    torch_weights_ = torch.tensor(
                        [torch_weights[sample.item()] for sample in batch.rano],
                        dtype=torch.float
                    ).to(device)
            elif weighted_loss:
                torch_weights_ = torch_weights.to(device)
            else: torch_weights_ = None
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch) 
            if loss_function.__name__ in ['binary_cross_entropy', 'binary_cross_entropy_with_logits']: 
                out = out.squeeze()
                batch.y = batch.y.float()
            #print(out.shape, batch.y.shape, torch_weights_.shape)
            if loss_function.__name__ == 'ldam_loss':
                loss = loss_function(out, batch.y, train_distribution)
            else:
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
                if weighted_loss and loss_function.__name__ != 'cross_entropy': 
                    size=batch.y.shape[0]
                    torch_weights_ = torch.tensor(
                        [torch_weights[sample.item()] for sample in batch.rano],
                        dtype=torch.float
                    ).to(device)
                elif weighted_loss:
                    torch_weights_ = torch_weights.to(device)
                    
                else: torch_weights_ = None
                batch = batch.to(device)
                out = model(batch)
                if loss_function.__name__ in ['binary_cross_entropy', 'binary_cross_entropy_with_logits']: 
                    out = out.squeeze()
                    batch.y = batch.y.float()
                if loss_function.__name__ == 'ldam_loss':
                    loss = loss_function(out, batch.y, train_distribution)
                else:
                    loss = loss_function(out, batch.y, weight=torch_weights_) if weighted_loss else loss_function(out, batch.y) # compute loss with weights
                if verbose: print('Validation loss:', loss.item())
                val_losses.append(loss.item())
                if loss < best_loss:
                    best_loss=loss
                    best_model=copy.deepcopy(model)
                    pat = 0
                else:
                    pat += 1


        if epoch%20==0 and epoch!=0 : ## incremental reporting
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

def train_regression(model, dataset, epochs=200, loss_function=F.cross_entropy, optimizer=adam, scheduler=None, working_dir=None, device='cuda', validation=0.25, batch_size='all', use_target_index=False, verbose=False, weighted_loss=True):
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
            if loss_function.__name__ == 'binary_cross_entropy': 
                out = out.argmax(dim=1)
            print(torch_weights_)
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
                if loss_function.__name__ == 'binary_cross_entropy': out = out.argmax(dim=1)
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

def test_classification(model, dataset, working_dir, device='cuda', rano_encoding=None, binary=False):
    loader = DataLoader(dataset, batch_size=len(dataset))
    model.eval()
    # with open(working_dir/'assignments.csv', 'w') as file:
        # writer = csv.DictWriter(file, fieldnames=['id', 'prediction', 'target', 'confidence'], delimiter=';')
        # writer.writeheader()
    for batch in loader:
        batch.to(device)
        y = batch.y.cpu().numpy()
        out = model(batch)
        confidence = out.detach().cpu().numpy()
        ids = batch.id
        if not binary: 
            assignment = out.argmax(dim=1).cpu().numpy()
            y_bin = label_binarize(y, classes=[0,1,2,3])
            probs = np.exp(confidence)
            fpr, tpr, roc_auc = {}, {}, {}
            n_classes = y_bin.shape[1]
            plt.figure(figsize=(8, 6))
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multiclass ROC Curve (One-vs-Rest)')
            plt.legend(loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(working_dir/'roc.png')
            plt.close()
            plt.clf()
        else: 
            assignment = out.detach().squeeze()
            confidence = torch.sigmoid(assignment)
            fpr, tpr, thresholds = roc_curve(y, confidence.cpu().numpy())
            roc_auc = auc(fpr, tpr)
            # optimal_idx = np.argmax(tpr - fpr)
            # optimal_threshold = thresholds[optimal_idx]
            # print(f"Optimal threshold (Youden's J): {optimal_threshold:.3f}")
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(working_dir/'roc.png')
            plt.close()
            plt.clf()

            assignment = (confidence >= 0.5).long().cpu().numpy()
            confidence = confidence.cpu().numpy()

            # for id, pd, gt, conf in zip(batch.id, assignment, y, confidence):
            #     res = {
            #         'id': id if isinstance(id, str) else id.item(),
            #         'prediction': pd, 'target': gt, 'confidence':conf}
            #     writer.writerow(res)
    return classification_evaluation(rano_pd=assignment, rano_gt=y, rano_encoding=None, ids=ids, rano_proba=confidence, out=working_dir)

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
