from prediction import regression_evaluation
import pandas as pd
import pathlib as pl
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from visualization import plot_regression_metrics
from sklearn.model_selection import GridSearchCV, train_test_split
import data as d
from lightgbm import LGBMRegressor
import numpy as np
import csv
import os
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

        if normalization is not None: plot_regression_metrics(regression_evaluation(gt_decoded, preds_decoded), output, tag)
        else: plot_regression_metrics(regression_evaluation(gt, preds), output, tag=tag)

def evaluate_tX(ref, targets, data, output, tag='t5'):
        preds = data[ref]
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

        if normalization is not None: plot_regression_metrics(regression_evaluation(gt_decoded, preds_decoded), output, tag)
        else: plot_regression_metrics(regression_evaluation(gt, preds), output, tag=tag)

if __name__ == '__main__':
    for normalization in ['3root->std', 'frac->3root', None, 'max', 'log', 'std', 'frac']:
        folder_name = 'baselines'
        method_name = 'reg_Linear'
        #normalization = '3root+std'
        model = LinearRegression()
        data = f'/mnt/nas6/data/Target/BMPipeline_DEVELOPMENT_runs/task_502_PARSED_METS_mrct1000_nobatch/csv_nn_only_valid/features.csv'
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
                                        normalize_volume=normalization)


        features = [c for c in train.columns if c.split('_')[0] in ["t0", "t1", "t2", "t3", "t4", "t5"] and 'volume' in c.split('_')[1]]
        print('Input features are:', features)
        target = 't6_volume'
        print('Target variable is:', target)

        print(train[features])
        model.fit(train[features], train[target])

        evaluate(model, features, target, test, output, 'test')
        evaluate(model, features, target, train, output, 'train')
        evaluate_tX('t5_volume', 't6_volume', test, output, 't5_as_pred')
        