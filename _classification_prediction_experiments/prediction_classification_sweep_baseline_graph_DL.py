import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from prediction import *
import pandas as pd
import pathlib as pl
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from visualization import *
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightgbm import LGBMClassifier
import torch_geometric.transforms as T
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import data as d
import pandas as pd
import torch
import numpy as np
from collections import Counter
from torch_geometric.loader import DataLoader
from prediction import GCN, GAT, classification_evaluation
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

class AddNoise(T.BaseTransform):
    def __init__(self, p=0.1):
        self.p = p
    def __call__(self, data):
        data.x = data.x + self.p * torch.randn_like(data.x)
        # data.edge_weights = data.edge_weights + 0.01 * torch.randn_like(data.edge_weights)
        # data.edge_attr = data.edge_attr + 0.01 * torch.randn_like(data.edge_attr)
        return data
    
class FeatureDropout(T.BaseTransform):
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, data):
        if data.x is not None:
            mask = torch.rand_like(data.x) > self.p
            data.x = data.x * mask
        return data

class FocalLoss(nn.Module):
    # source: https://datascientistsdiary.com/implementing-focal-loss-in-pytorch-for-class-imbalance/
    def __init__(self, alpha=0.25, gamma=5.0, reduction='mean'):
        """
        Custom implementation of Focal Loss in PyTorch.
        
        Parameters:
        alpha (float): Weighting factor for the rare class (default 0.25).
        gamma (float): Modulating factor to down-weight easy examples (default 2.0).
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the Focal Loss between logits and targets.

        Parameters:
        inputs (Tensor): Model predictions (logits) of shape (batch_size, num_classes).
        targets (Tensor): Ground truth labels of shape (batch_size,).

        Returns:
        Tensor: Computed Focal Loss.
        """
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
def focal_loss(input, targets, weight=None):
    loss = FocalLoss()
    return loss(input, targets)

class LDAMLoss(nn.Module):
    # https://torcheeg.readthedocs.io/en/latest/_modules/torcheeg/trainers/imbalance/ldam.html
    def __init__(self,
                 class_frequency: List[int],
                 max_margin: float = 0.5,
                 weight: Tensor = None,
                 scaling: float = 30):
        '''
        Label-distribution-aware margin (LDAM) loss for imbalanced datasets.

        - Paper: Cao K, Wei C, Gaidon A, et al. Learning imbalanced datasets with label-distribution-aware margin loss[J]. Advances in neural information processing systems, 2019, 32.
        - URL: https://proceedings.neurips.cc/paper_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf
        - Related Project: https://github.com/kaidic/LDAM-DRW

        Args:
            class_frequency (List[int]): The frequency of each class in the dataset.
            max_margin (float): The maximum margin. (default: :obj:`0.5`)
            weight (Tensor): The weight of each class. (default: :obj:`None`)
            scaling (float): The scaling factor. (default: :obj:`30`)
        '''
        super(LDAMLoss, self).__init__()
        max_key = max(class_frequency.keys())  # Ensure list is long enough
        result = [None] * (max_key + 1)
        for k, v in class_frequency.items():
            result[k] = v
        class_frequency = result
        margin_list = 1.0 / np.sqrt(np.sqrt(class_frequency))
        margin_list = margin_list * (max_margin / np.max(margin_list))
        self.register_buffer('margin_list', torch.tensor(margin_list).float().to('cuda'))
        assert scaling > 0, "scaling should be greater than 0."
        self.scaling = scaling
        if not weight is None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        index = torch.zeros_like(input)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        index_bool = index.bool()

        batch_m = torch.matmul(self.margin_list[None, :],
                               index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = input - batch_m

        output = torch.where(index_bool, x_m, input)
        return F.cross_entropy(self.scaling * output,
                               target,
                               weight=self.weight)

def ldam_loss(input, target, weight):
    loss = LDAMLoss(weight)
    return loss(input, target)


if __name__ == '__main__':
########## setup
    options=[
         {'prediction': '1v3', 
         'selection': None, 
         'model': 'SimplestGCN', 
         'loss': 'cross_entropy',  
         'feats': ['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex',	'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics'],
         'transforms': T.Compose([AddNoise(0.5), FeatureDropout(0.1)]),
         'fully_connect': True,
         'direction': 'past',
         'loss_balance': True,
         'noise_level': 0.5},

    ]

    for i, config in enumerate(options):
        data = pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn/features.csv')
        prediction_type = config['prediction']
        feature_selection = config['selection']
        method = config['model']
        loss_func = config['loss']
        output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/output/graph_ml_lvl_4-ftuning')
        used_features = config['feats']
        categorical =  ['Sex',	'Primary_loc_1', 'lesion_location', 'Primary_hist_1']


        if prediction_type == 'binary':
            rano_encoding={'CR':0, 'PR':0, 'SD':1, 'PD':1}
            num_out=1
            classes = ['resp', 'non-resp']
            if loss_func == 'cross_entropy':
                loss = F.binary_cross_entropy_with_logits
            elif loss_func == 'focal':
                loss = focal_loss
            elif loss_func == 'ldam':
                loss = ldam_loss
        elif prediction_type == '1v3':
            rano_encoding={'CR':0, 'PR':1, 'SD':1, 'PD':1}
            num_out=1
            classes=['CR', 'non-CR']
            if loss_func == 'cross_entropy':
                loss = F.binary_cross_entropy_with_logits
            elif loss_func == 'focal':
                loss = focal_loss
            elif loss_func == 'ldam':
                loss = ldam_loss
        else:
            rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}
            classes = list(rano_encoding.keys())
            num_out=4
            loss = F.cross_entropy

        if feature_selection == 'LASSO':
            eliminator = d.LASSOFeatureEliminator(alpha=0.1)
        elif feature_selection == 'correlation':
            eliminator = d.FeatureCorrelationEliminator()
        elif feature_selection == 'model':
            eliminator = d.ModelFeatureEliminator()
        else:
            eliminator = None

        data_prefixes = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"] # used in the training method to select the features for each step of the sweep
        volume_cols = [c+'_volume' for c in data_prefixes] # used to normalize the volumes
        rano_cols = [elem+'_rano' for elem in data_prefixes] # used in the training method to select the current targets

        output = output_path/f'classification/{prediction_type}/featuretypes={used_features}_selection={feature_selection}/{method}_{loss_func}_exp{i}'
        os.makedirs(output, exist_ok=True)
        with open(output/'exp_config.csv', 'w') as file:
            file.write(str(config))

        train_data, test_data = d.load_prepro_data(data,
                                            categorical=categorical,
                                            fill=0,
                                            used_features=used_features,
                                            test_size=0.2,
                                            drop_suffix=eliminator,
                                            prefixes=data_prefixes,
                                            target_suffix='rano',
                                            normalize_suffix=[f for f in used_features if f!='volume'],
                                            rano_encoding=rano_encoding,
                                            time_required=True,
                                            interpolate_CR_swing_length=1,
                                            drop_CR_swing_length=2,
                                            normalize_volume='std',
                                            save_processed=output.parent/'encoding_test_used_data.csv')
        extra_data = [c for c in train_data.columns if not (c.startswith('ignored') or c.split('_')[0] in data_prefixes)]
        print("using extra data cols", extra_data)
        dist = Counter(test_data['t6_rano'])

        os.makedirs(output, exist_ok=True)
        with open(output/'used_feature_names.txt', 'w') as file:
            file.write("Used feature names left in the dataframe:\n")
            for c in train_data.columns:
                file.write(f"   - {c}\n")
            file.write("NOTE: rano columns are used as targets not as prediction")

        ## class weight definition for torch
        labels = [row['t6_rano'] for i, row in train_data.iterrows()]
        label_counts = Counter(labels)
        num_classes = len(label_counts)  # or len(label_counts)
        counts = torch.tensor([label_counts[i] for i in range(num_classes)], dtype=torch.float)
        torch_weights = 1.0 / (counts + 1e-6)  # Avoid divide by zero
        torch_weights = torch_weights / torch_weights.sum()  # Normalize (optional but nice)



        train_transforms = config['transforms']
        test_transforms = None#T.Compose([T.LocalDegreeProfile(), T.GDC()])

        all_results = {}
        for i in range(2, len(data_prefixes)):
            key_year = f"1yr_{data_prefixes[:i]}->t6_rano"
            # make datasets
            dataset_train = d.BrainMetsGraphClassification(train_data,
                used_timepoints = data_prefixes[:i], 
                ignored_suffixes = ('_timedelta_days', '_rano', 'Lesion ID'), 
                rano_encoding = rano_encoding,
                target_name = 't6_rano',
                extra_features = extra_data,
                fully_connected=config['fully_connect'],
                direction=config['direction'],
                transforms = train_transforms,
                )
            dataset_test = d.BrainMetsGraphClassification(test_data,
                used_timepoints = data_prefixes[:i], 
                ignored_suffixes = ('_timedelta_days', '_rano', 'Lesion ID'), 
                rano_encoding = rano_encoding,
                target_name = 't6_rano',
                extra_features = extra_data,
                fully_connected=config['fully_connect'],
                transforms = test_transforms,
                direction = config['direction']
                )
            
            experiment_dir = output/key_year
            os.makedirs(experiment_dir, exist_ok=True)

            # init training variables
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if method == 'SimplestGCN':
                model = SimplestGCN(num_out, dataset_train.get_node_size()).to(device)
            elif method == 'GCN':
                model = GCN(num_out, dataset_train.get_node_size()).to(device)
            else:
                raise RuntimeError(f"Unrecognized method name, cant resolve correct model for {method}")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=20,       # First restart after 10 epochs
                T_mult=2,     # Increase period between restarts by this factor
                eta_min=1e-6  # Minimum LR
            )

            # run training
            best_model, best_loss = torch_engine.train(model, 
                                            dataset= dataset_train, 
                                            loss_function=loss,
                                            epochs=1000,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            working_dir=experiment_dir,
                                            device=device,
                                            validation=0.25,
                                            batch_size=128,
                                            weighted_loss=config['loss_balance']
                                            )
            print(f"Best model achieved loss {best_loss:4f}")

            # evaluate
            best_res = torch_engine.test_classification(best_model, dataset_test, experiment_dir, device, rano_encoding, num_out==1)
            print(f"""Best model achieved a class weight balanced accuracy {best_res['balanced_accuracy']:4f}""")
            print(best_res['classification_report'])
            all_results[key_year] = best_res

            # plot
            plot_prediction_metrics(best_res, experiment_dir)
        plot_prediction_metrics_sweep(all_results, output, classes=classes, distribution=dist)