import pandas as pd

data = pd.read_csv('/mnt/nas6/data/Target/BMPipeline_DEVELOPMENT_runs/task_524-504_PARSED_METS_mrct1000_nobatch/csv_nn_524-504_reseg_only_valid/rano.csv', index_col='Lesion ID')
renamer = {elem: elem+'_rano' for elem in data.columns}
rano_data = data.rename(columns=renamer)

data = pd.read_csv('/mnt/nas6/data/Target/BMPipeline_DEVELOPMENT_runs/task_524-504_PARSED_METS_mrct1000_nobatch/csv_nn_524-504_reseg_only_valid/volumes.csv', index_col='Lesion ID')
renamer = {elem: elem+'_volume' for elem in data.columns}
volume_data = data.rename(columns=renamer)
complete_data = pd.concat([volume_data, rano_data], axis=1)

cols = ["0", "60", "120", "180", "240", "300", "360"]

for i, c in enumerate(cols):
    complete_data[f"t{i}_timedelta_days"] = int(c)
    renamer = {elem: f"t{i}_{elem.split('_')[-1]}" for elem in complete_data.columns if c == elem.split('_')[0]}
    print(renamer)
    complete_data = complete_data.rename(columns=renamer)

complete_data.to_csv('/mnt/nas6/data/Target/BMPipeline_DEVELOPMENT_runs/task_524-504_PARSED_METS_mrct1000_nobatch/csv_nn_524-504_reseg_only_valid/patched_data.csv')