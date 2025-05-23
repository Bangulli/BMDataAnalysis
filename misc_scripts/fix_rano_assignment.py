import pandas as pd

data = pd.read_csv('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn/all_features_broken_rano.csv', index_col = 'Lesion ID')
rano = pd.read_csv('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn/feature_fixed_rano.csv', index_col = 'Lesion ID')
rano_cols = ['t0_rano', 't1_rano', 't2_rano', 't3_rano', 't4_rano', 't5_rano', 't6_rano']

data[rano_cols]=rano[rano_cols]

data.to_csv('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn/features.csv')