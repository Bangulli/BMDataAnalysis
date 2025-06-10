import pandas as pd
from collections import Counter

df = pd.read_csv('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/final_extraction/all_features_nn.csv', index_col=('Lesion ID'))

df.drop(index=['sub-PAT0122:1', 
                'sub-PAT0167:0', 
                'sub-PAT0182:2', 
                'sub-PAT0342:0', 
                'sub-PAT0411:0', 
                'sub-PAT0434:6', 
                'sub-PAT0434:9', 
                'sub-PAT0434:10', 
                'sub-PAT0434:11', 
                'sub-PAT0480:20', 
                'sub-PAT0484:4', 
                'sub-PAT0490:0', 
                'sub-PAT0612:2', 
                'sub-PAT0666:0', 
                'sub-PAT0756:0', 
                'sub-PAT1028:3',
                'sub-PAT0045:6',
                'sub-PAT0105:0',
                'sub-PAT0441:0', 
                'sub-PAT0686:1',
                'sub-PAT0807:3',
                ], inplace=True)

#df = df[df['#Lesions']!=0]
un = [id.split(':')[0] for id in df.index]
un = set(un)
print(len(un), un)
m = df[df['Sex']=='M']
f = df[df['Sex']=='F']

print(f'#Patients: {len(f)} Female  {len(m)} Male')
for col in ['t0_volume', 't6_rano']:
    #col = 't6_rano'
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
