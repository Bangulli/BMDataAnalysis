import data as d
import pandas as pd
import torch
import numpy as np
from collections import Counter
from torch_geometric.loader import DataLoader
from prediction import GraphClassificationModel, GCN, GAT, classification_evaluation
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
#from torch_geometric.nn.models import GraphSAGE, GCN


BMGDS = d.BrainMetsGraphDataset(
    '/mnt/nas6/data/Target/BMPipeline_DEVELOPMENT_runs/task_502_PARSED_METS_mrct1000_nobatch/csv_nn_only_valid/features.csv',
    used_timepoints = ['t0', 't1', 't2', 't3'], 
    ignored_suffixes = ('_timedelta_days', '_rano', 'Lesion ID'), 
    rano_encoding = {'CR':0, 'PR':1, 'SD':2, 'PD':3},
    target_name = 't6_rano',
    extra_features = None,
    transforms = None,
    )

# Step 2: Extract labels (assume `y` is 1D tensor)
labels = [data.y.item() for data in BMGDS]
label_counts = Counter(labels)
num_classes = len(label_counts)  # or len(label_counts)
counts = torch.tensor([label_counts[i] for i in range(num_classes)], dtype=torch.float)
torch_weights = 1.0 / (counts + 1e-6)  # Avoid divide by zero
torch_weights = torch_weights / torch_weights.sum()  # Normalize (optional but nice)

# Step 3: Stratified split into train, val, test
train_idx, test_idx = train_test_split(
    range(len(BMGDS)), stratify=labels, test_size=0.2, random_state=42
)
# train_idx, val_idx = train_test_split(
#     train_idx, stratify=[labels[i] for i in train_idx], test_size=0.25, random_state=42
# )  # 0.25 x 0.8 = 0.2 for val

# Step 4: Subset your dataset manually
train_dataset = [BMGDS[i] for i in train_idx]
#val_dataset = [BMGDS[i] for i in val_idx]
test_dataset = [BMGDS[i] for i in test_idx]

# Step 5: Create loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = GraphClassificationModel(GCN(BMGDS.get_node_size(), BMGDS.get_node_size(), 10, 4)).to(device)
model = GCN(4, BMGDS.get_node_size()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
best_acc = 0
best_rep = None
for epoch in range(200):
    print(f'== epoch {epoch}/200')
    # train step
    model.train()
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, batch.y, weight=torch_weights.to(device))
        #print(loss)
        loss.backward()
        optimizer.step()
    # validation step
    model.eval()
    for batch in test_loader:
        batch.to(device)
        pred = model(batch).argmax(dim=1).cpu().numpy()
        gt = batch.y.cpu().numpy()
        res = classification_evaluation(gt, pred)
        acc = res['balanced_accuracy']
        if acc > best_acc:
            best_acc = acc
            best_rep = res['classification_report']
print(f"Best model achieved accuracy {best_acc:4f}")
print(best_rep)

####    RUNS ON A DIFFERENT DATASET CONSTURCTOR!!!
"""
def __init__(self, 
                 csv_path, 
                 used_timepoints: list = ['t0', 't1', 't2'], 
                 ignored_suffixes: tuple = ('_timedelta_days', '_rano', 'Lesion ID'), 
                 rano_encoding = {'CR':0, 'PR':1, 'SD':2, 'PD':3},
                 target_name = 't6_rano',
                 extra_features = None,
                 transforms = None,
                 ):
        self.table = pd.read_csv(csv_path, index_col='Lesion ID')
        self.used_timepoints = used_timepoints
        self.ignored_suffixes = ignored_suffixes
        self.rano_encoding = rano_encoding
        self.target_name = target_name
        self.extra_features = extra_features
        self.transforms = transforms
######### preprocess data
        self.table = self.table[~self.table[target_name].isna()]
        self.table = self.table.fillna(0)
        self._encode_rano()
######### preprocess data

######### normalize data
        ## normalize follow up volumes
        used_timepoints = [c+'_volume' for c in used_timepoints]
        self.table[used_timepoints[1:]] = self.table[used_timepoints[1:]].div(self.table[used_timepoints[0]], axis=0)
        ## normalize init volume
        self.table[used_timepoints[0]]=zscore(self.table[used_timepoints[0]])
        ## normalize radiomics
        for tp in self.used_timepoints:
            for col in self.table.columns:
                if col.startswith(f"{tp}_radiomics"):
                    self.table[col]=zscore(pd.to_numeric(self.table[col], errors='coerce')) # try to parse every value to floats
        ## normalize extra data
        if extra_features is not None:
            for col in extra_features:
                self.table[col]=zscore(pd.to_numeric(self.table[col], errors='coerce')) # try to parse every value to floats
        ## drop any broken columns
        for col in self.table.columns:
            if self.table[col].dtype == 'object':
                print('found broken column:', col)
                self.table.drop(columns=col, inplace=True)
        print(self.table.describe())
        self.table.info()
######### normalize data
"""