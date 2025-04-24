from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch
import pandas as pd
from scipy.stats import zscore

def row2graph(row: pd.Series, timepoints: list, extra_features: list, y_name:str, fully_connected, ignore_types: tuple = ('_timedelta_days', '_rano', 'Lesion ID'), verbose=False):
    # prepare variables and data
    row = pd.to_numeric(row, errors='coerce') # saveguard against missparsed data
    row.fillna(0, inplace=True) # saveguard against nan in data
    edge_index = []
    edge_values = []
    node_values = []
    # create a graph where each node is connected to each other node with a direct edge and its corresponding value
    if fully_connected: 
        for i, tp1 in enumerate(timepoints):
            feature_names = [f for f in row.index.tolist() if tp1 in f and not f.endswith(ignore_types)] # this filter works
            if extra_features:
                feature_names += extra_features
            features = row[feature_names].to_list() # extract extra data and timepoint data from row, ignoring time column though
            node_values.append(features)
            for j, tp2 in enumerate(timepoints):
                if i == j:
                    continue
                edge_index.append([i, j])
                edge_values.append([abs(row[f"{tp2}_timedelta_days"] - row[f"{tp1}_timedelta_days"])/365.25]) # value is the timedelta normalized by one year (compensating for switch years)
    # create a sparse graph where each node is only connected to its direct neighbors in time
    else:
        for i, tp in enumerate(timepoints):
            feature_names = [f for f in row.index.tolist() if tp1 in f and not f.endswith(ignore_types)] # this filter works
            if extra_features:
                feature_names += extra_features
            features = row[feature_names].to_list() # extract extra data and timepoint data from row, ignoring time column though
            node_values.append(features)
            # edge case t0
            if i == 0:
                # positive in time
                edge_index.append([i, i+1])
                edge_values.append([abs(row[f"{timepoints[i+1]}_timedelta_days"] - row[f"{tp}_timedelta_days"])/365.25])
            # edge case t-1
            elif i == len(timepoints)-1:
                # negative in time
                edge_index.append([i-1, i])
                edge_values.append([abs(row[f"{tp}_timedelta_days"] - row[f"{timepoints[i-1]}_timedelta_days"])/365.25])
            # normal case
            else:
                # negative in time
                edge_index.append([i-1, i])
                edge_values.append([abs(row[f"{tp}_timedelta_days"] - row[f"{timepoints[i-1]}_timedelta_days"])/365.25])
                # positive in time
                edge_index.append([i, i+1])
                edge_values.append([abs(row[f"{timepoints[i+1]}_timedelta_days"] - row[f"{tp}_timedelta_days"])/365.25])
    
    # parse to torch
    edge_index = torch.tensor(edge_index, dtype = torch.long).t().contiguous()
    x = torch.tensor(node_values, dtype=torch.float)
    y = torch.tensor([row[y_name]], dtype=torch.long) # for graph level target needs shape [1, *]
    edge_values = torch.tensor(edge_values, dtype=torch.float)
    # parse to graph Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_values, y=y)
    # Debug and validation
    if verbose: print(data, data.validate())
    else: data.validate(raise_on_error=True)
    if verbose: print('Edge info', data.edge_weight, data.edge_index)
    if verbose: print('Node Info', data.x, data.y)
    return data

class BrainMetsGraphDataset(Dataset):
    def __init__(self, 
                 df, 
                 used_timepoints: list = ['t0', 't1', 't2'], 
                 ignored_suffixes: tuple = ('_timedelta_days', '_rano', 'Lesion ID'), 
                 rano_encoding = {'CR':0, 'PR':1, 'SD':2, 'PD':3},
                 target_name = 't6_rano',
                 fully_connected = True,
                 extra_features = None,
                 transforms = None,
                 ):
        self.table = df
        self.used_timepoints = used_timepoints
        self.ignored_suffixes = ignored_suffixes
        self.rano_encoding = rano_encoding
        self.target_name = target_name
        self.extra_features = extra_features
        self.transforms = transforms
        self.fully_connected = fully_connected
        self._encode_rano()

    def __len__(self):
        return len(self.table)
    
    def len(self):
        return self.__len__()
    
    def __getitem__(self, idx):
        row = self.table.iloc[idx, :]
        graph = row2graph(row, self.used_timepoints, self.extra_features, self.target_name, self.ignored_suffixes)
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