import numpy as np
import sklearn

gt = np.ones(9000)
gt = np.concatenate((gt, np.zeros(1000)))
np.random.shuffle(gt)

pd = np.random.choice([0, 1], size=10000)

print('accuracy = ', sklearn.metrics.accuracy_score(gt, pd))


weights = {u: len(gt)/np.sum(gt==u) for u in np.unique(gt)}
weights = [weights[l] for l in gt]
print('balanced_accuracy = ', sklearn.metrics.accuracy_score(gt, pd, sample_weight=weights))