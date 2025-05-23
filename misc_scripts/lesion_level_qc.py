import numpy as np
import os
import pathlib as pl

data = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502')

all_mets = []
pats = [p for p in os.listdir(data) if p.startswith('sub-PAT')]

for p in pats:
    mets = [p+'/'+m for m in os.listdir(data/p) if m.startswith('Metastasis')]
    all_mets += mets

z = 1.65 # == 80% confidence, 1.65=90% confidence
e = 0.1 # == margin of error 10%
N = len(all_mets) # popuöation  size
p = 0.5 # proportion
n = round((N*(z**2)*p*(1-p))/(e**2*(N-1)+(z**2)*p*(1-p)))
print(f"Minimum required sample size for confidence 80% and margin of error 10% and a population of {N} is {n}")

sample_set = np.random.choice(all_mets, replace=False, size=n)

print(sample_set)