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
from PrettyPrint.figures import Table, ProgressBar


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
            if value_map is not None: v2['prediction'] = v2['prediction'].map(value_map)
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
            if value_map is not None: v2['prediction'] = v2['prediction'].map(value_map)
            if any(v2['target'].isna()):
                raise ValueError('Mishap')
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
    N_PERMUTATIONS = 1000
    

    tp_mapping={
        '''1yr_['t0']->t6_rano''': '''1yr_t0:0->target_rano''',
        '''1yr_['t0', 't1']->t6_rano''':'''1yr_t0:90->target_rano''',
        '''1yr_['t0', 't1', 't2']->t6_rano''':'''1yr_t0:150->target_rano''',
        '''1yr_['t0', 't1', 't2', 't3']->t6_rano''':'''1yr_t0:210->target_rano''',
        '''1yr_['t0', 't1', 't2', 't3', 't4']->t6_rano''':'''1yr_t0:270->target_rano''',
        '''1yr_['t0', 't1', 't2', 't3', 't4', 't5']->t6_rano''':'''1yr_t0:330->target_rano'''
    }
    models = {}
    # models['best_classic_cr'] = load_general(value_map=None, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/classic_experts_5fold_best_model_assignment_bugfix/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location']_selection=None/LGBM'''))
    # models['best_classic_resp'] = load_general(value_map={1:0,0:1}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/classic_experts_5fold_best_model_assignment_bugfix/classification/binary/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics', 'deep']_selection=LASSO/LogisticRegression'''))
    # models['best_gml_noisy_general_cr'] = load_general(value_map={'PD':1, 'CR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_general_noisy_5fold/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp2'''))
    # models['best_gml_noisy_general_resp'] = load_general(value_map={'PD':1, 'PR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_general_noisy_5fold/classification/binary/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp9'''))
    # models['best_gml_resamp_general_cr'] = load_general(value_map={'PD':1, 'CR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_general_5fold/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp2'''))
    # models['best_gml_resamp_general_resp'] = load_general(value_map={'PD':1, 'PR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_general_5fold/classification/binary/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp9'''))
    # models['best_gml_noisy_ts_cr'] = load_expert(value_map={'PD':1, 'CR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_experts_noisy_5fold_bugfix/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp2'''))
    # models['best_gml_noisy_ts_resp'] = load_expert(value_map={'PD':1, 'PR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_experts_noisy_5fold_bugfix/classification/binary/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp7'''))
    # models['best_gml_resamp_ts_cr'] = load_expert(value_map={'PD':1, 'CR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_experts_5fold/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp2'''))
    # models['best_gml_resamp_ts_resp'] = load_expert(value_map={'PD':1, 'PR':0}, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/graph_ml_experts_5fold/classification/binary/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/BigGAT_cross_entropy_exp2'''))

    models['clinic_vol'] = load_general(value_map=None, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/classic_experts_5fold_assignments_is_radio_value/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location']_selection=None/LGBM'''))
    models['clinic_vol_radio'] = load_general(value_map=None, source=pl.Path('''/home/lorenz/BMDataAnalysis/final_output/classic_experts_5fold_assignments_is_radio_value/classification/1v3/featuretypes=['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex', 'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics']_selection=None/LGBM'''))
    
    model_data = {}
    for k, v in models.items():
        model_data[k] = combine_folds(v)
        #print(model_data[k])

    ### Permutation test for a model across timepoints
    # printer = PrettyPrint.Printer(log_type="txt", log_prefix='permutations_crosstimepoints', location='/home/lorenz/BMDataAnalysis/final_output')
    # for model_name, tp_data in model_data.items():
    #     printer(f"Permutation test results for {model_name}")
    #     keys = list(tp_data.keys())
    #     #for i in range(0, len(tp_data)-1):
    #     for j in range(1, len(tp_data)):
    #         i = j-1
    #         df1 = tp_data[keys[i]]
    #         df2 = tp_data[keys[j]]
    #         auc1 = roc_auc_score(df1['target'], df1['confidence'])
    #         auc2 = roc_auc_score(df2['target'], df2['confidence'])
    #         base_auc_diff = abs(auc1-auc2)
    #         n_perm_diff_larger = 0
    #         for permutation in ProgressBar(range(N_PERMUTATIONS),100):
    #             # Create a boolean mask with 50% True (swap), 50% False (keep)
    #             swap_mask = np.random.rand(len(df1)) < 0.5

    #             # Apply row-wise swap using the mask
    #             df1_work = df1.copy()
    #             df2_work = df2.copy()

    #             df1_work[swap_mask] = df2.loc[swap_mask].values
    #             df2_work[swap_mask] = df1.loc[swap_mask].values

    #             work_auc1 = roc_auc_score(df1_work['target'], df1_work['confidence'])
    #             work_auc2 = roc_auc_score(df2_work['target'], df2_work['confidence'])
    #             perm_auc_diff = abs(work_auc1-work_auc2)

    #             if perm_auc_diff >= base_auc_diff:
    #                 n_perm_diff_larger += 1

                
    #         printer(f"Testing AUC significance between {keys[i]} and {keys[j]}, RESULT: p={n_perm_diff_larger/N_PERMUTATIONS}")
                
    ### Permutation test for a timepoint across models
    printer = PrettyPrint.Printer(log_type="txt", log_prefix='permutations_crossmodels', location='/home/lorenz/BMDataAnalysis/final_output/classic_experts_5fold_assignments_is_radio_value')
    for tp1, tp2 in tp_mapping.items():
        printer(f"Permutation test results for {tp1}")
        keys = list(model_data.keys())

        #for i in range(0, len(keys)-1):
        i=0
        for j in range(1, len(keys)):
            j = 1
            df1 = model_data[keys[i]][tp1] if tp1 in list( model_data[keys[i]].keys()) else model_data[keys[i]][tp2]
            df2 = model_data[keys[j]][tp1] if tp1 in list( model_data[keys[j]].keys()) else model_data[keys[j]][tp2]
            auc1 = roc_auc_score(df1['target'], df1['confidence'])
            auc2 = roc_auc_score(df2['target'], df2['confidence'])
            base_auc_diff = abs(auc1-auc2)
            n_perm_diff_larger = 0
            for permutation in ProgressBar(range(N_PERMUTATIONS),100):

                # Create a boolean mask with 50% True (swap), 50% False (keep)
                swap_mask = np.random.rand(len(df1)) < 0.5

                # Apply row-wise swap using the mask
                df1_work = df1.copy()
                df2_work = df2.copy()

                df1_work[swap_mask] = df2.loc[swap_mask].values
                df2_work[swap_mask] = df1.loc[swap_mask].values

                work_auc1 = roc_auc_score(df1_work['target'], df1_work['confidence'])
                work_auc2 = roc_auc_score(df2_work['target'], df2_work['confidence'])
                perm_auc_diff = abs(work_auc1-work_auc2)

                if perm_auc_diff >= base_auc_diff:
                    n_perm_diff_larger += 1

            
            printer(f"Testing AUC significance between {keys[i]} and {keys[j]}, RESULT: p={n_perm_diff_larger/N_PERMUTATIONS}")