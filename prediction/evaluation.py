import sklearn
import numpy as np

def classification_evaluation(rano_gt, rano_pd):
    res = {}

    # compute class weights
    weights = {u: len(rano_gt)/np.sum(rano_gt==u) for u in np.unique(rano_gt)}
    #print(weights)
    # make weight vector
    weights = [weights[l] for l in rano_gt]
    # is equivalent to sklearn.utils.class_weight.compute_sample_weight
    
    res['balanced_accuracy'] = sklearn.metrics.accuracy_score(rano_gt, rano_pd, sample_weight=weights)
    res['accuracy'] = sklearn.metrics.accuracy_score(rano_gt, rano_pd)
    #res['roc_auc'] = sklearn.metrics.roc_auc_score(rano_gt, rano_pd, multi_class='ovr')
    res['f1'] = sklearn.metrics.f1_score(rano_gt, rano_pd, average='weighted')
    res['precision'] = sklearn.metrics.precision_score(rano_gt, rano_pd, average='weighted')
    res['recall'] = sklearn.metrics.recall_score(rano_gt, rano_pd, average='weighted')
    res['classification_report'] = sklearn.metrics.classification_report(rano_gt, rano_pd, digits=4)
    res['confusion_matrix'] = sklearn.metrics.confusion_matrix(rano_gt, rano_pd, sample_weight=weights, normalize='all', labels=np.unique(rano_gt))
    return res