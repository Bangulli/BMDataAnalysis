import pandas as pd
import pathlib as pl

main_path = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_uninterpolated/all_features_all_tps_vincent_foundation.csv")
supplement_path = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_uninterpolated/location_features_all_tps.csv")
output_path = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_uninterpolated/all_features_all_tps_vincent_foundation.csv")

assert main_path != output_path
assert supplement_path != output_path

main_df = pd.read_csv(main_path, index_col='Lesion ID')
supplement_df = pd.read_csv(supplement_path, index_col='Lesion ID')
#supplement_df = supplement_df[['lesion_location']]
main_df[supplement_df.columns]=supplement_df
main_df.to_csv(output_path)