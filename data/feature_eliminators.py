import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.feature_selection import SelectFromModel

class FeatureCorrelationEliminator:
    def __init__(self, threshold, plot=True):
        self.threshold = threshold
        self.plot = plot
    
    def __call__(self, data, target=None):
        ### check for feature correlation
        # Compute the correlation matrix
        corr = data.corr()
        if self.plot:
            # Generate a mask for the upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))

            # Set up a larger figure size
            f, ax = plt.subplots(figsize=(30, 25))  # try 30x25 or even larger if needed

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(230, 20, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                        square=True, linewidths=.5,
                        cbar_kws={"shrink": .5},
                        vmin=corr.values.min(), vmax=corr.values.max())

            # Optional: rotate tick labels for readability
            plt.xticks(rotation=90, fontsize=8)
            plt.yticks(rotation=0, fontsize=8)

            plt.title('Feature Correlation')
            plt.xlabel('Feature Name')
            plt.ylabel('Feature Name')
            plt.tight_layout()  # fits everything within figure bounds
            plt.savefig('feature_correlations.png', dpi=300)
            plt.close()

        # Threshold for correlation
        threshold = self.threshold

        # Absolute correlation matrix
        corr_matrix = corr.abs()

        # Upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find columns with correlation above threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        if self.plot:
            tp_data_filtered = data.drop(columns=to_drop)
            
            corr = tp_data_filtered.corr()

            # Generate a mask for the upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))

            # Set up a larger figure size
            f, ax = plt.subplots(figsize=(30, 25))  # try 30x25 or even larger if needed

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(230, 20, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                        square=True, linewidths=.5,
                        cbar_kws={"shrink": .5},
                        vmin=corr.values.min(), vmax=corr.values.max())

            # Optional: rotate tick labels for readability
            plt.xticks(rotation=90, fontsize=8)
            plt.yticks(rotation=0, fontsize=8)

            plt.title('Feature Correlation')
            plt.xlabel('Feature Name')
            plt.ylabel('Feature Name')
            plt.tight_layout()  # fits everything within figure bounds
            plt.savefig('feature_correlations_after_elimination.png', dpi=300)
            plt.close()
        print(f'== FeatureCorrelationEliminator wants to remove the following columns:')
        [print(f"   - {c}") for c in to_drop]
        return to_drop
    
class LASSOFeatureEliminator:
    def __init__(self, alpha=0.5):
        self.alpha=alpha
    def __call__(self, data, target):
        lasso = Lasso(alpha=self.alpha)
        print(f"== fitting LASSO predictor with radiomics feature names in:")
        [print(f"   - {c}") for c in data.columns if 'radiomics' in c]
        print(f"== fitting LASSO predictor with feature names in:")
        [print(f"   - {c}") for c in data.columns if not 'radiomics' in c]
        lasso.fit(data.fillna(0), target)
        eliminate_mask = lasso.coef_==0
        
        to_drop = data.columns[eliminate_mask]
        # print(f'== LASSOFeatureEliminator wants to remove the following columns:')
        # [print(f"   - {c}") for c in to_drop]
        print("LASSO Coefficients:", lasso.coef_)
        print("Non-zero Coefs:", np.count_nonzero(lasso.coef_))
        print(f"== LSSOFeatureEliminator wants to keep the following columns:")
        [print(f"   - {c}") for c in data.columns[lasso.coef_!=0]]
        return list(to_drop)
    
class ModelFeatureEliminator:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, data, target):
        selector = SelectFromModel(self.model).fit(data.fillna(0), target)
        to_keep = selector.get_feature_names_out()
        to_drop = [f for f in data.columns if f not in to_keep]
        print(f'== ModelFeatureEliminator wants to remove the following columns:')
        [print(f"   - {c}") for c in to_drop]
        return list(to_drop)
