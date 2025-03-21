import core
import torch
from torch.utils.data import Dataset
import os
import pathlib as pl


class BrainMets(Dataset):
    def __init__(self, source_dir: pl.Path, transform=None):
        self.source_dir = source_dir
        self.patients = [p for p in os.listdir(self.source_dir) if p.startswith('sub-PAT')]
        self.preparsed_mets = self._check_data_structure()
        self.transform = transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.transform:
            raise NotImplementedError('The dataset does not yet support transforms')
        return 
    
    def _check_data_structure(self):
        """
        Utility that checks the type of dataset
        retruns true if the dataset is already parsed metastases
        """
        pats = []
        for pat in self.patients:
            files = [f.startswith('Metastasis') for f in os.listdir(self.source_dir/pat) if (self.source_dir/pat/f).is_dir()]
            pats.append(all(files))
        return all(pats)