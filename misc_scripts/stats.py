import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib as pl
import os
import re
import csv
from scipy import stats
from sklearn.metrics import auc, roc_curve, roc_auc_score
from statsmodels.stats.multitest import multipletests
import PrettyPrint
from PrettyPrint.figures import Table


def load_expert(source, value_map):
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    results = {}
    for fold in folds:   
        assignments = {} 
        for file in [f for f in os.listdir(source) if (source/f).is_dir()]:
            assignments[file] = pd.read_csv(source/file/fold/'assignments.csv')
        results[fold] = assignments
    for idx, (k, v) in enumerate(results.items()):
        for idx2, (k2, v2) in enumerate(v.items()):
            if value_map is not None: v2['target'] = v2['target'].map(value_map)
            if any(v2['target'].isna()):
                raise ValueError('Mishap')
    return results

def load_general(source, value_map):
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    results = {}
    for fold in folds:
        files = [f for f in os.listdir(source/fold) if (source/fold/f).is_dir()]
        assignments = {}
        for f in files:
            assignments[f] = pd.read_csv(source/fold/f/'assignments.csv')
        results[fold] = assignments  
    for idx, (k, v) in enumerate(results.items()):
       for idx2, (k2, v2) in enumerate(v.items()):
            if value_map is not None: v2['target'] = v2['target'].map(value_map)
            if any(v2['target'].isna()):
                raise ValueError('Mishap')
    return results

def get_aucs(data):
    res = {}
    for fold in ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']:
        tps = data[fold]
        aucs = {}
        for k, v in tps.items():
            aucs[k] = roc_auc_score(v['target'], v['confidence'])
        res[fold] = aucs
    return res


if __name__ == '__main__':
    tp_mapping={
        '''1yr_['t0']->t6_rano''': '''1yr_t0:0->target_rano''',
        '''1yr_['t0', 't1']->t6_rano''':'''1yr_t0:90->target_rano''',
        '''1yr_['t0', 't1', 't2']->t6_rano''':'''1yr_t0:150->target_rano''',
        '''1yr_['t0', 't1', 't2', 't3']->t6_rano''':'''1yr_t0:210->target_rano''',
        '''1yr_['t0', 't1', 't2', 't3', 't4']->t6_rano''':'''1yr_t0:270->target_rano''',
        '''1yr_['t0', 't1', 't2', 't3', 't4', 't5']->t6_rano''':'''1yr_t0:330->target_rano'''
    }
    models = {}
    models['best_classic_cr'] = load_general(value_map=None, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/classic_experts_5fold_best_model_assignment_bugfix/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location']_selection=None/LGBM'''))
    models['best_classic_resp'] = load_general(value_map={1:0,0:1}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/classic_experts_5fold_best_model_assignment_bugfix/classification/binary/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics', 'deep']_selection=LASSO/LogisticRegression'''))
    models['best_gml_noisy_general_cr'] = load_general(value_map={'PD':1, 'CR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_general_noisy_5fold/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp2'''))
    models['best_gml_noisy_general_resp'] = load_general(value_map={'PD':1, 'PR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_general_noisy_5fold/classification/binary/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp9'''))
    models['best_gml_resamp_general_cr'] = load_general(value_map={'PD':1, 'CR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_general_5fold/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp2'''))
    models['best_gml_resamp_general_resp'] = load_general(value_map={'PD':1, 'PR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_general_5fold/classification/binary/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp9'''))
    models['best_gml_noisy_ts_cr'] = load_expert(value_map={'PD':1, 'CR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_experts_noisy_5fold_bugfix/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp2'''))
    models['best_gml_noisy_ts_resp'] = load_expert(value_map={'PD':1, 'PR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_experts_noisy_5fold_bugfix/classification/binary/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp7'''))
    models['best_gml_resamp_ts_cr'] = load_expert(value_map={'PD':1, 'CR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_experts_5fold/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp2'''))
    models['best_gml_resamp_ts_resp'] = load_expert(value_map={'PD':1, 'PR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_experts_5fold/classification/binary/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp2'''))

    model_aucs = {}
    for k, v in models.items():
        model_aucs[k] = get_aucs(v)
    
    ### reorganize data
    model_aucs_reorg = {}
    with open('/home/lorenz/BMDataAnalysis/final_output/aucs.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=['model', 'tp_config', 'auc_mean', 'auc_std', 'auc_fold_1', 'auc_fold_2', 'auc_fold_3', 'auc_fold_4', 'auc_fold_5'])
        writer.writeheader()
        for model_name, model_auc in model_aucs.items():
            reorg = {}
            logline = {}   
            for k, v in tp_mapping.items():
                logline['model'] = model_name
                logline['tp_config'] = k
                for i, (fold, fold_auc) in enumerate(model_auc.items()):
                    keys = list(fold_auc.keys())
                    if i == 0:
                        reorg[k] = []
                    if k in keys:
                        reorg[k].append(fold_auc[k])
                        logline[f'auc_fold_{i+1}'] = fold_auc[k]
                    else:
                        reorg[k].append(fold_auc[v])
                        logline[f'auc_fold_{i+1}'] = fold_auc[v]
                logline['auc_mean'] = np.mean(reorg[k])
                logline['auc_std'] = np.std(reorg[k])
                writer.writerow(logline)
            model_aucs_reorg[model_name] = reorg
    

    ### test performance significance
    print(f"RUNNING EXPERIMENTS FOR MODEL PERFORMANCE ACROSS TIMEPOINTS")
    printer = PrettyPrint.Printer(log_type="txt", log_prefix='stats_crosstimepoint', location='/home/lorenz/BMDataAnalysis/final_output')
    for model_name, model_auc in model_aucs_reorg.items():
        printer(f"Testing model {model_name} with a wilcoxon test")
        tp_keys = list(model_auc.keys())
        for j in range(0, len(model_auc)-1):
            p_vals = []
            for i in range(1, len(model_auc)):
                printer.tagged_print("Testing", f"AUC significance between {tp_keys[j]} and {tp_keys[i]}", PrettyPrint.Default())
                printer(f"Aucs across folds are:\n{tp_keys[j]}: {model_auc[tp_keys[j]]}\n{tp_keys[i]}: {model_auc[tp_keys[i]]}")
                stat, p = stats.wilcoxon(model_auc[tp_keys[j]], model_auc[tp_keys[i]])
                printer.tagged_print("Results", f"stat={stat}, p={p}", PrettyPrint.Success())
                p_vals.append(p)
               
            printer("--------------------")
        printer("-------------------------------------")

    # ### Testing timepoint performance across models
    # print(f"RUNNING EXPERIMENTS FOR TIMEPOINT PERFORMANCE ACROSS MODELS")
    # printer = PrettyPrint.Printer(log_type=None, log_prefix='stats_crossmodels', location='/home/lorenz/BMDataAnalysis/final_output')
    # for tp, _ in tp_mapping.items():
    #     printer(f"Testing model {tp} with a wilcoxon test")
    #     model_keys = list(model_aucs_reorg.keys())
    #     for j in range(0, len(model_keys)-1):
    #         p_vals = []
    #         for i in range(1, len(model_keys)):
    #             printer.tagged_print("Testing", f"AUC significance between {model_keys[j]} and {model_keys[i]} at tp config {tp}", PrettyPrint.Default())
    #             print(f"Aucs across folds are:\n", model_auc[tp_keys[i-1]], "\n", model_auc[tp_keys[1]])
    #             stat, p = stats.wilcoxon(model_aucs_reorg[model_keys[j]][tp], model_aucs_reorg[model_keys[i]][tp])
    #             printer.tagged_print("Results", f"stat={stat}, p={p}", PrettyPrint.Success())
    #             p_vals.append(p)
    #         printer("--------------------")
    #     printer("-------------------------------------")