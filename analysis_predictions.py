from prediction import *
import pandas as pd
import pathlib as pl
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from visualization import *


if __name__ == '__main__':
    folder_name = 'csv_linear_multiclass_reseg_only_valid'
    data =  pd.read_csv(f'/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch/{folder_name}/volumes.csv', index_col=None)
    output = pl.Path(f'/home/lorenz/BMDataAnalysis/output/{folder_name}/{'SVR'}')
    
    data_cols = ["0", "60", "120", "180", "240", "300", "360"]

    data[data_cols[1:]] = data[data_cols[1:]].div(data["0"], axis=0) # normalize

    train, test = train_test_split(data[data_cols], test_size=0.2)
    _, res_qual, res_quant = train_model_sweep(SVR(), train, test, data_cols=data_cols)
    plot_regression_metrics(res_qual, output)
    plot_regression_metrics(res_quant, output)