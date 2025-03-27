import pandas as pd
import numpy as np

def aic(model, X, y): # adapted from stepmix source code
    return -2 * model.score(X, y) * X.shape[0] + 2 * model.n_parameters

def bic(model, X, y): # adapted from stepmix source code
    return 2 * model.score(X, y) * X.shape[0] + model.n_parameters * np.log(X.shape[0])