import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from prediction import regression_evaluation
import pandas as pd
import pathlib as pl
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, GammaRegressor, TweedieRegressor
from sklearn.model_selection import train_test_split
from visualization import plot_regression_metrics
from sklearn.model_selection import GridSearchCV, train_test_split
import data as d
from lightgbm import LGBMRegressor
import numpy as np
import csv
import os
from sklearn.base import BaseEstimator, RegressorMixin
import copy
def evaluate(model, features, targets, data, output, tag='test'):
        preds = model.predict(data[features])
        gt = data[targets]

        gt_decoded = []
        preds_decoded = []
        off_threshold = 100
        very_off = []
        with open(output/f'{tag}_regressions.csv', 'w') as file:
            writer = csv.DictWriter(file, fieldnames=["prediction_decoded", "target_decoded", "prediction", "target", 'data_row', 'Lesion ID', 'features'])
            writer.writeheader()
            for i in range(len(data)):
                sample = data.iloc[i,:]
                res = {}

                if normalization is not None:
                    decoder = sample['ignored_vol_normalizer']
                    
                    target = eval(decoder.format(gt[i]))
                    out = eval(decoder.format(preds[i]))

                    #print('Decoding with:', decoder, 'obtaining:', gt[i], '->', target, for the ground truth value)
                    preds_decoded.append(out)
                    gt_decoded.append(target)

                else:
                     target=gt[i]
                     out=preds[i]

                res[f"prediction_decoded"] = out if normalization is not None else 'NA'
                res[f"target_decoded"] = target if normalization is not None else 'NA'
                res["prediction"] = preds[i]
                res["target"] = gt[i]
                res["features"] = sample[features].to_dict()
                res["Lesion ID"] = sample.name
                writer.writerow(res)

                if abs(target-out)>off_threshold:
                     very_off.append({'id': sample.name, 'target': target, 'pred': out, 'discrepancy': abs(out-target)})
        
        if any(very_off):
            with open(output/f'{tag}_large_discrepancies.csv', 'w') as file:
                writer = csv.DictWriter(file, fieldnames=list(very_off[0].keys()))
                writer.writeheader()
                writer.writerows(very_off)

        if normalization is not None: plot_regression_metrics(regression_evaluation(gt_decoded, preds_decoded, None), output, tag)
        else: plot_regression_metrics(regression_evaluation(gt, preds, None), output, tag=tag)

def evaluate_tX(ref, targets, data, output, tag='t5'):
        
        gt = data[targets]
        if ref != 'zeros': preds = data[ref]
        else: preds = np.zeros_like(gt)
        

        gt_decoded = []
        preds_decoded = []
        off_threshold = 100
        very_off = []
        with open(output/f'{tag}_regressions.csv', 'w') as file:
            writer = csv.DictWriter(file, fieldnames=["prediction_decoded", "target_decoded", "prediction", "target", 'data_row', 'Lesion ID', 'features'])
            writer.writeheader()
            for i in range(len(data)):
                sample = data.iloc[i,:]
                res = {}

                if normalization is not None:
                    decoder = sample['ignored_vol_normalizer']
                    
                    target = eval(decoder.format(gt[i]))
                    out = eval(decoder.format(preds[i]))

                    #print('Decoding with:', decoder, 'obtaining:', gt[i], '->', target, for the ground truth value)
                    preds_decoded.append(out)
                    gt_decoded.append(target)

                else:
                     target=gt[i]
                     out=preds[i]

                res[f"prediction_decoded"] = out if normalization is not None else 'NA'
                res[f"prediction_decoded"] = res[f"prediction_decoded"] if ref != 'zeros' else 0
                res[f"target_decoded"] = target if normalization is not None else 'NA'
                res["prediction"] = preds[i]
                res["prediction"] = res["prediction"] if ref != 'zeros' else 'NA'
                res["target"] = gt[i]
                res["features"] = sample[features].to_dict()
                res["Lesion ID"] = sample.name
                writer.writerow(res)

                if abs(target-out)>off_threshold:
                     very_off.append({'id': sample.name, 'target': target, 'pred': out, 'discrepancy': abs(out-target)})

        if any(very_off):
            with open(output/f'{tag}_large_discrepancies.csv', 'w') as file:
                writer = csv.DictWriter(file, fieldnames=list(very_off[0].keys()))
                writer.writeheader()
                writer.writerows(very_off)

        if ref == 'zeros': preds_decoded = np.zeros_like(preds_decoded)
        if ref == 'zeros': preds = np.zeros_like(preds)

        if normalization is not None: plot_regression_metrics(regression_evaluation(gt_decoded, preds_decoded, None), output, tag)
        else: plot_regression_metrics(regression_evaluation(gt, preds, None), output, tag=tag)

class NonZeroRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, classifier):
        self.regressor = regressor
          
        self.classifier = classifier

    def fit(self, X, y):
        X0, y0 = X, copy.deepcopy(y.astype(float))
        y0[y0!=0]=1
        print(y0)
        print(X0.shape, y0.shape)

        XR, yR = X[y!=0], y[y!=0]
        print(XR.shape, yR.shape)
        self.classifier.fit(X0, y0)
        self.regressor.fit(XR, yR)
        return self
    
    def predict(self, X):
        y0 = self.classifier.predict(X)
        for sample in range(len(X)):
             if y0[sample] == 1:
                 y0[sample] = self.regressor.predict(X.iloc[[sample]])
        return y0
            
        

if __name__ == '__main__':
    for normalization in ['frac->log']:#['3root->std', 'std', 'log', 'frac->3root', 'max', 'frac', None]: # 
        folder_name = 'baselines'
        method_name = 'reg_linear'
        #normalization = '3root+std'
        model = LinearRegression() #NonZeroRegressor(regressor=SVR(), classifier=SVC())
        data = pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/229_patients faulty/PARSED_METS_task_502/csv_nn/features.csv')
        valid_char_norm = normalization.replace('->', '+') if normalization is not None else normalization
        output = pl.Path(f'''/home/lorenz/BMDataAnalysis/output/{folder_name}/regression_{method_name}/prepro_{valid_char_norm}''')
        os.makedirs(output, exist_ok=True)
        
        data_cols = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"]


        train, test = d.load_prepro_data(data,
                                        used_features=['volume'],
                                        test_size=0.2,
                                        drop_suffix=None,
                                        prefixes=data_cols,
                                        target_suffix='rano',
                                        normalize_suffix=None,
                                        rano_encoding={ 'CR': 0,'PR': 1,'SD': 2,'PD': 3 },
                                        time_required=False,
                                        interpolate_CR_swing_length=1,
                                        drop_CR_swing_length=2,
                                        normalize_volume=normalization,
                                        save_processed=data.parent/f"features_new_{valid_char_norm}.csv")


        features = [c for c in train.columns if c.split('_')[0] in ["t0", "t1", "t2", "t3", "t4", "t5"] and 'volume' in c.split('_')[1]]
        print('Input features are:', features)
        target = 't6_volume'
        print('Target variable is:', target)

        print(train[features])
        model.fit(train[features], train[target])

        evaluate(model, features, target, test, output, 'test')
        evaluate(model, features, target, train, output, 'train')
        evaluate_tX('t5_volume', 't6_volume', test, output, 't5_as_pred')
        evaluate_tX('zeros', 't6_volume', test, output, 'zeros_as_pred')
        print(output)
        