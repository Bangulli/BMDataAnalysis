import pandas as pd
import pathlib as pl

main_path = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn_experiment/features_vol_radiomics.csv")
supplement_path = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn_experiment/total_load.csv")
output_path = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn/features.csv")

assert main_path != output_path
assert supplement_path != output_path

main_df = pd.read_csv(main_path, index_col='Lesion ID')
supplement_df = pd.read_csv(supplement_path, index_col='Lesion ID')
main_df[supplement_df.columns]=supplement_df
output = pd.concat([main_df, supplement_df], axis=1)
main_df.to_csv(output_path)