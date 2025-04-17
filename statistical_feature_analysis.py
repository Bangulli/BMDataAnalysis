import pathlib as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
folder_name = 'csv_nn_multiclass_reseg_only_valid'
source = pl.Path(f'/mnt/nas6/data/Target/task_524-504_REPARSED_METS_mrct1000_nobatch/{folder_name}/features.csv')
output = pl.Path(f'/home/lorenz/BMDataAnalysis/output/{folder_name}')
os.makedirs(output, exist_ok=True)

d = pd.read_csv(source, index_col=None)
data_cols = ["0", "60", "120", "180", "240", "300", "360"]

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin=corr.min(axis=None).min(), vmax=corr.max(axis=None).min())
plt.title('Feature Correlation')
plt.xlabel('Feature Name')
plt.ylabel('Feature Name')
plt.savefig(output/'feature_correlations.png')
plt.close()
plt.clf()

