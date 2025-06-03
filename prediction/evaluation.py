import sklearn
import numpy as np
import csv

def classification_evaluation(rano_gt, rano_pd, ids, rano_encoding=None, rano_proba=None, out=None):
    res = {}

    # compute class weights
    weights = {u: len(rano_gt)/np.sum(rano_gt==u) for u in np.unique(rano_gt)}
    #print(weights)
    # make weight vector
    weights = [weights[l] for l in rano_gt]
    # is equivalent to sklearn.utils.class_weight.compute_sample_weight
    if rano_encoding is not None:
        rano_encoding_reverse = {v: k for k, v in rano_encoding.items()}
        rano_gt = [rano_encoding_reverse[v] for v in rano_gt]
        rano_pd = [rano_encoding_reverse[v] for v in rano_pd]

    if out is not None:
        with open(out/'assignments.csv', 'w') as file:
            writer = csv.DictWriter(file, fieldnames=['id', 'prediction', 'target', 'confidence'])
            writer.writeheader()
            for g, p, c, i in zip(rano_gt, rano_pd, rano_proba, ids):
                writer.writerow({'id': i, 'prediction': p, 'confidence': c, 'target': g})


    res['balanced_accuracy'] = sklearn.metrics.accuracy_score(rano_gt, rano_pd, sample_weight=weights)
    res['accuracy'] = sklearn.metrics.accuracy_score(rano_gt, rano_pd)
    if rano_proba is not None: res['roc_auc'] = sklearn.metrics.roc_auc_score(rano_gt, rano_proba)
    res['f1'] = sklearn.metrics.f1_score(rano_gt, rano_pd, average='weighted')
    res['precision'] = sklearn.metrics.precision_score(rano_gt, rano_pd, average='weighted')
    res['recall'] = sklearn.metrics.recall_score(rano_gt, rano_pd, average='weighted')
    res['classification_report'] = sklearn.metrics.classification_report(rano_gt, rano_pd, digits=4)
    res['confusion_matrix'] = sklearn.metrics.confusion_matrix(rano_gt, rano_pd, sample_weight=weights, normalize='true', labels=list(rano_encoding.keys()))
    return res

def regression_evaluation(gt, pd, decoder):
    gt_dec = []
    pd_dec = []
    if decoder is not None:
        for i in range(len(gt)):
            gt_dec.append(eval(decoder[i].format(gt[i])))
            pd_dec.append(eval(decoder[i].format(pd[i])))
            print(gt[i], decoder[i].format(gt[i]), eval(decoder[i].format(gt[i])))
    else:
        gt_dec = gt
        pd_dec = pd

    res = {}
    res['r2'] = sklearn.metrics.r2_score(gt_dec, pd_dec)
    res['median_ae-sep'] = sklearn.metrics.median_absolute_error(gt_dec, pd_dec)
    res['mean_ae-sep'] = sklearn.metrics.mean_absolute_error(gt_dec, pd_dec)
    res['rmse-sep'] = sklearn.metrics.root_mean_squared_error(gt_dec, pd_dec)
    res['evs'] = sklearn.metrics.explained_variance_score(gt_dec, pd_dec)
    res['evs_encoded'] = sklearn.metrics.explained_variance_score(gt, pd)
    res['r2_encoded'] = sklearn.metrics.r2_score(gt, pd)
    res['r2_zeros'] = sklearn.metrics.r2_score(np.zeros_like(pd), pd)
    res['r2_dec_zeros'] =  sklearn.metrics.r2_score(np.zeros_like(pd), pd_dec)
    return res