from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch
import pandas as pd
import random
from scipy.stats import zscore

def row2graph(row: pd.Series, id, timedelta_cutoff: int, extra_features: list, y_name:str, fully_connected=True, direction=None, ignore_types: tuple = ('_timedelta_days', '_rano', 'Lesion ID'), verbose=False, random_period=False):
    # prepare variables and data
    row = pd.to_numeric(row, errors='coerce') # saveguard against missparsed data
    row.fillna(0, inplace=True) # saveguard against nan in data
    edge_index = []
    edge_weights = []
    node_values = []
    edge_attributes = []
    tds = [c for c in row.index.tolist() if "timedelta_days" in c]
    timepoints = [tp.split('_')[0] for tp in tds if row[tp]<=timedelta_cutoff and row[tp]!=0] + ['t0']
    timepoints.sort()
    if random_period:
        cutoff = random.randint(1, len(timepoints))
        timepoints=timepoints[:cutoff]
    # create a graph where each node is connected to each other node with a direct edge and its corresponding value
    if fully_connected: 
        if len(timepoints) >= 2:
            for i, tp1 in enumerate(timepoints):
                feature_names = [f for f in row.index.tolist() if tp1 ==  f.split('_')[0] and not f.endswith(ignore_types)] # this filter works
                if extra_features:
                    feature_names += extra_features
                features = row[feature_names].to_list() # extract extra data and timepoint data from row, ignoring time column though
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
        else:
            feature_names = [f for f in row.index.tolist() if timepoints[0] ==  f.split('_')[0] and not f.endswith(ignore_types)] # this filter works
            if extra_features:
                feature_names += extra_features
            features = row[feature_names].to_list() # extract extra data and timepoint data from row, ignoring time column though
            node_values.append(features)
            edge_index = [[0,0]]
            edge_attributes = [0]
            edge_weights = [[0]]
    # create a sparse graph where each node is only connected to its direct neighbors in time
    else:
        if len(timepoints) >= 2:
            for i, tp in enumerate(timepoints):
                feature_names = [f for f in row.index.tolist() if tp == f.split('_')[0] and not f.endswith(ignore_types)] # this filter works
                if extra_features:
                    feature_names += extra_features
                features = row[feature_names].to_list() # extract extra data and timepoint data from row, ignoring time column though
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
        else:
            feature_names = [f for f in row.index.tolist() if timepoints[0] ==  f.split('_')[0] and not f.endswith(ignore_types)] # this filter works
            if extra_features:
                feature_names += extra_features
            features = row[feature_names].to_list() # extract extra data and timepoint data from row, ignoring time column though
            node_values.append(features)
            edge_index = [[0,0]]
            edge_attributes = [0]
            edge_weights = [[0]]
    
    # parse to torch
    edge_index = torch.tensor(edge_index, dtype = torch.long).t().contiguous()
    x = torch.tensor(node_values, dtype=torch.float)
    y = torch.tensor([row[y_name]], dtype=torch.long) # for graph level target needs shape [1, *]
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    edge_attributes = torch.tensor(edge_attributes, dtype=torch.float)
    # parse to graph Data object
    data = Data(x=x, edge_index=edge_index, edge_weights=edge_weights, y=y, edge_attr=edge_attributes)
    data.rano = y
    #data.id = row['Lesion ID']
    data.id = id
    # Debug and validation
    if verbose: print(data, data.validate())
    else: data.validate(raise_on_error=True)
    if verbose: print('Edge info', data.edge_weights, data.edge_index)
    if verbose: print('Node Info', data.x, data.y)
    return data

class NoisyBrainMetsGraphClassification(Dataset):
    def __init__(self, 
                 df, 
                 used_timedelta: 300, 
                 ignored_suffixes: tuple = ('_timedelta_days', '_rano', 'Lesion ID'), 
                 rano_encoding = {'CR':0, 'PR':1, 'SD':2, 'PD':3},
                 target_name = 'target_rano',
                 fully_connected = True,
                 extra_features = None,
                 transforms = None,
                 direction = None,
                 random_period=False
                 ):
        self.table = df
        self.used_timedelta = used_timedelta
        self.ignored_suffixes = ignored_suffixes
        self.rano_encoding = rano_encoding
        self.target_name = target_name
        self.extra_features = extra_features
        self.transforms = transforms
        self.fully_connected = fully_connected
        self.direction = direction
        self.random_period = random_period
        self._encode_rano()

    def __len__(self):
        return len(self.table)
    
    def len(self):
        return self.__len__()
    
    def __getitem__(self, idx):
        row = self.table.iloc[idx, :]
        id = row['Lesion ID']
        graph = row2graph(row=row, id=id, timedelta_cutoff=self.used_timedelta, extra_features=self.extra_features, y_name=self.target_name, ignore_types=self.ignored_suffixes, fully_connected=self.fully_connected, direction=self.direction, random_period=self.random_period)
      
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