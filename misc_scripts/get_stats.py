import pandas as pd
from collections import Counter

df = pd.read_csv('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/cleaned_dataset_description_complete.csv', index_col=None)

df = df[df['#Lesions']!=0]

m = df[df['Sex']=='M']
f = df[df['Sex']=='F']

print(f'#Patients: {len(f)} Female  {len(m)} Male')
for col in df.columns:
    if pd.api.types.is_numeric_dtype(f[col]):
        mean_f = f[col].mean()
        median_f = f[col].median()
        std_f = f[col].std()
        mean_m = m[col].mean()
        median_m = m[col].median()
        std_m = m[col].std()
        print(f"{col}:")
        print(f"Female:     mean={mean_f:.2f}, median={median_f:.2f}, std={std_f:.2f}")
        print(f"Male:       mean={mean_m:.2f}, median={median_m:.2f}, std={std_m:.2f}")
    else:
        res_f = Counter(f[col])
        res_m = Counter(m[col])
        print(f"{col}:")
        print("Female:")
        [print(f"   {n}: {x}") for n,x in res_f.items()]
        print("Male:")
        [print(f"   {n}: {x}") for n,x in res_m.items()]
