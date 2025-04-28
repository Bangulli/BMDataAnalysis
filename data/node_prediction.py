from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch
import pandas as pd
from scipy.stats import zscore
import numpy as np
import copy

def row2graph(row: pd.Series, timepoints: list, extra_features: list, y_name:str, fully_connected=True, direction=None, ignore_types: tuple = ('_timedelta_days', '_rano', 'Lesion ID'), verbose=False):
    # prepare variables and data
    row = pd.to_numeric(row, errors='coerce') # saveguard against missparsed data
    row.fillna(0, inplace=True) # saveguard against nan in data
    edge_index = []
    edge_weights = []
    node_values = []
    edge_attributes = []
    # create a graph where each node is connected to each other node with a direct edge and its corresponding value
    if fully_connected: 
        timepoints = [*timepoints, y_name]
        for i, tp1 in enumerate(timepoints):
            feature_names = [f for f in row.index.tolist() if tp1 in f and not f.endswith(ignore_types)] # this filter works
            if extra_features:
                feature_names += extra_features
            features = row[feature_names].to_list() # extract extra data and timepoint data from row, ignoring time column though
            targets = copy.deepcopy(features) if i == len(timepoints)-1 else None
            features = list(np.ones_like(features)*-1) if i == len(timepoints)-1 else features # remove the target values
            node_values.append(features)
            for j, tp2 in enumerate(timepoints):
                if i == j:
                    continue
                delta = row[f"{tp2}_timedelta_days"] - row[f"{tp1}_timedelta_days"]
                if direction is None:
                    edge_index.append([i, j])
                    edge_weights.append([abs(delta)/365.25]) # value is the timedelta normalized by one year (compensating for switch years)
                    edge_attributes.append(delta/365.25)
                elif direction == 'future':
                    if delta>0:
                        edge_index.append([i, j])
                        edge_weights.append([abs(delta)/365.25])
                        edge_attributes.append(delta/365.25)
                elif direction == 'past':
                    if delta<0:
                        edge_index.append([i, j])
                        edge_weights.append([abs(delta)/365.25])
                        edge_attributes.append(delta/365.25)
                else:
                    raise RuntimeError(f"Unknown direction argument {direction}")
    # create a sparse graph where each node is only connected to its direct neighbors in time
    else:
        timepoints = [*timepoints, y_name]
        for i, tp in enumerate(timepoints):
            feature_names = [f for f in row.index.tolist() if tp in f and not f.endswith(ignore_types)] # this filter works
            if extra_features:
                feature_names += extra_features
            features = row[feature_names].to_list() # extract extra data and timepoint data from row, ignoring time column though
            targets = copy.deepcopy(features) if i == len(timepoints)-1 else None
            features = list(np.ones_like(features)*-1) if i == len(timepoints)-1 else features # remove the target values
            node_values.append(features)
            # edge case t0
            if i == 0:
                # positive in time
                if direction in [None, 'future']:
                    edge_index.append([i, i+1])
                    edge_weights.append([abs(row[f"{timepoints[i+1]}_timedelta_days"] - row[f"{tp}_timedelta_days"])/365.25])
                    edge_attributes.append((row[f"{timepoints[i+1]}_timedelta_days"] - row[f"{tp}_timedelta_days"])/365.25)
            # edge case t-1
            elif i == len(timepoints)-1:
                # negative in time
                if direction in [None, 'past']:
                    edge_index.append([i, i-1])
                    edge_weights.append([abs(row[f"{tp}_timedelta_days"] - row[f"{timepoints[i-1]}_timedelta_days"])/365.25])
                    edge_attributes.append((row[f"{tp}_timedelta_days"] - row[f"{timepoints[i-1]}_timedelta_days"])/365.25)
            # normal case
            else:
                # negative in time
                if direction in [None, 'past']:
                    edge_index.append([i, i-1])
                    edge_weights.append([abs(row[f"{tp}_timedelta_days"] - row[f"{timepoints[i-1]}_timedelta_days"])/365.25])
                    edge_attributes.append((row[f"{tp}_timedelta_days"] - row[f"{timepoints[i-1]}_timedelta_days"])/365.25)
                # positive in time
                if direction in [None, 'future']:
                    edge_index.append([i, i+1])
                    edge_weights.append([abs(row[f"{timepoints[i+1]}_timedelta_days"] - row[f"{tp}_timedelta_days"])/365.25])
                    edge_attributes.append((row[f"{timepoints[i+1]}_timedelta_days"] - row[f"{tp}_timedelta_days"])/365.25)

    
    
    # parse to torch
    edge_index = torch.tensor(edge_index, dtype = torch.long).t().contiguous()
    x = torch.tensor(node_values, dtype=torch.float)
    y = torch.tensor([targets], dtype=torch.float) # for graph level target needs shape [1, *]
    rano = torch.tensor([row[f"{y_name}_rano"]], dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    edge_attributes = torch.tensor(edge_attributes, dtype=torch.float)
    # parse to graph Data object
    data = Data(x=x, edge_index=edge_index, edge_weights=edge_weights, y=y, edge_attr=edge_attributes)
    data.target_index = torch.tensor([x.shape[0]-1])
    data.rano = rano
    # Debug and validation
    if verbose: print(data, data.validate())
    else: data.validate(raise_on_error=True)
    if verbose: print('Edge info', data.edge_weights, data.edge_index)
    if verbose: print('Node Info', data.x, data.y)
    return data

class BrainMetsNodeRegression(Dataset):
    def __init__(self, 
                 df, 
                 used_timepoints: list = ['t0', 't1', 't2'], 
                 ignored_suffixes: tuple = ('_timedelta_days', '_rano', 'Lesion ID'), 
                 rano_encoding = {'CR':0, 'PR':1, 'SD':2, 'PD':3},
                 target_name = 't6',
                 fully_connected = True,
                 extra_features = None,
                 transforms = None,
                 direction = None,
                 ):
        self.table = df
        self.used_timepoints = used_timepoints
        self.ignored_suffixes = ignored_suffixes
        self.rano_encoding = rano_encoding
        self.target_name = target_name
        self.extra_features = extra_features
        self.transforms = transforms
        self.fully_connected = fully_connected
        self.direction = direction
        self._encode_rano()

    def __len__(self):
        return len(self.table)
    
    def len(self):
        return self.__len__()
    
    def __getitem__(self, idx):
        row = self.table.iloc[idx, :]
        graph = row2graph(row=row, timepoints=self.used_timepoints, extra_features=self.extra_features, y_name=self.target_name, ignore_types=self.ignored_suffixes, fully_connected=self.fully_connected, direction=self.direction)
        if self.transforms:
            graph = self.transforms(graph)
        return graph
    
    def get(self, idx):
        return self.__getitem__(idx)

    def _encode_rano(self):
        rano_cols = [c for c in self.table.columns if c.endswith('_rano')]
        self.table[rano_cols] = self.table[rano_cols].applymap(lambda x: self.rano_encoding.get(x, x))

    def get_node_size(self):
        g = self[0]
        return g.x.shape[-1]
    
#class MetTSGraphInMemory()