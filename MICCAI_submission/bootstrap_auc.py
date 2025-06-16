import matplotlib.pyplot as plt
import re
import pathlib as pl
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score

def load_expert(source):
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    results = {}
    for fold in folds:   
        assignments = {} 
        for file in [f for f in os.listdir(source) if (source/f).is_dir()]:
            assignments[file] = pd.read_csv(source/file/fold/'assignments.csv')
        results[fold] = assignments
    return results

def load_general(source):
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    results = {}
    for fold in folds:
        files = [f for f in os.listdir(source/fold) if (source/fold/f).is_dir()]
        assignments = {}
        for f in files:
            assignments[f] = pd.read_csv(source/fold/f/'assignments.csv')
        results[fold] = assignments  
    return results

def combine_folds(data):
    res = {}
    for fold in ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']:
        tps = data[fold]
        for k, v in tps.items():
            if not k in list(res.keys()):
                res[k] = v
            else:
                res[k] = pd.concat([res[k], v], axis=0, ignore_index=True)
    return res


if __name__ == '__main__':
    resamples = 1000
    source = pl.Path('''/home/lorenz/BMDataAnalysis/MICCAI_submission/GML_time-specific/classification/binary/featuretypes=['Age@Onset', 'Weight', 'Height'] - ['Sex', 'Primary_loc_1', 'lesion_location', 'Primary_hist_1'] - ['volume', 'radiomics']''')
    assignments = load_expert(source)
    combined = combine_folds(assignments)
    # Initialize the master plot
    plt.figure(figsize=(8, 9))
    colors = plt.cm.tab10.colors  # or any other colormap if you need more than 10

    for idx, (k, v) in enumerate(combined.items()):
        aucs = []
        for i in range(resamples):
            sample_df = v.sample(n=len(v), replace=True)
            aucs.append(roc_auc_score(sample_df['target'], y_score=sample_df['confidence']))

        fpr, tpr, thresholds = roc_curve(v['target'], v['confidence'])
        roc_auc = auc(fpr, tpr)
        lower = np.percentile(aucs, 2.5)
        upper = np.percentile(aucs, 97.5)

        # Plot individual ROC
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f}); CI95% = {lower:.2f} - {upper:.2f}")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        k_safe = re.sub(r"[\[\]',<>]", "", k).replace(':', '-')
        plt.savefig(source / f'''{k_safe}_roc.png''')
        plt.close()

        # Add to master plot
        color = colors[idx % len(colors)]
        plt.figure(1)  # switch back to master plot
        plt.plot(fpr, tpr, lw=2,
                label=f"{k} (AUC={roc_auc:.2f}; CI={lower:.2f}-{upper:.2f})",
                color=color)

    # Finalize and save the master ROC plot
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Methods')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Place the legend below the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize='small')
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(source / "combined_roc.png")
    plt.close()

