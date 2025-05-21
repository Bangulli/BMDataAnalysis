import pandas as pd
import pathlib as pl

main_path = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_uninterpolated/all_features_all_tps.csv")
supplement_path = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/dataset_description_complete.csv")
output_path = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_uninterpolated/features_all_tps.csv")

import chardet

with open(supplement_path, 'rb') as f:
    result = chardet.detect(f.read(10000))
print(result)

assert main_path != output_path
assert supplement_path != output_path

main_df = pd.read_csv(main_path, index_col='Lesion ID')
supplement_df = pd.read_csv(supplement_path, index_col='Patient ID', encoding='utf-8')
supplement_df = supplement_df.replace("""['nan']""", None)
supplement_df = supplement_df.replace('''[]''', None)
supplement_df = supplement_df.replace('''[nan]''', None)
supplement_df = supplement_df.replace("""['""", '')
supplement_df = supplement_df.replace("""']""", '')

renamer_Primary_loc_1 = {
    "Rein": "Kidney",
    "Poumon": "Lung",
    "Mélanome": "Melanoma",
    "Cerveau": "Brain",
    "Sein": "Breast",
    "Prostate": "Prostate",
    "Ovaire": "Ovary",
    "Cerveau + Moelle": "Brain",
    "Glioblastome": "Glioblastoma",
    "Parotide": "Parotid",
    "Vessie": "Bladder",
    "ORL": "ENT",
    "Tumeur épithéliale": "Epithelial tumor",
    "Ampoule (Pancréas)": "Ampulla (Pancreas)",
    "Colon": "Colon",
    "Oesophage": "Esophagus",
    "ORL (Hypopharynx)": "ENT (Hypopharynx)",
    "Astrocytome": "Astrocytoma",
    "Pulmonaire": "Lung"
}
renamer_Primary_loc_2 = {
    "Oesophage": "Esophagus",
    "Sein": "Breast",
    "rétropharynx": "retropharynx",
    "Poumon": "Lung",
    "Leucémie lymphocytique chronique": "Chronic lymphocytic leukemia",
    "Prostate": "Prostate",
    "Jonction oesogastrique": "Esogastric junction"
}
renamer_Primary_hist_1 = {
    "Cellules claires": "Clear cell",
    "Adénocarcinome": "Adenocarcinoma",
    "Gangliogliome": "Ganglioglioma",
    "RH- HER2+": "RH- HER2+",
    "SCLC": "SCLC",
    "Épidermoïde": "Squamous cell",
    "NOS": "NOS",
    "Oligodendrogliome": "Oligodendroglioma",
    "adénocarcinome": "Adenocarcinoma",
    "Neuroendocrine Grandes Cellules": "Neuroendocrine Large Cell",
    "Glioblastome": "Glioblastoma",
    "Séreux de haut grade": "High-grade serous",
    "Hémangiopéricytome": "Hemangiopericytoma",
    "Triple -": "Triple -",
    "Carcinome épidermoïde": "Squamous cell carcinoma",
    "Carcinome NST": "NST carcinoma",
    "NSCLC": "NSCLC",
    "Carcinome à cellules claires": "Clear cell carcinoma",
    "Adénocarcinom": "Adenocarcinoma",
    "Ductal": "Ductal",
    "Carcinome canalaire": "Ductal carcinoma",
    "Oligendrogliome + Astrocytome": "Oligendroglioma + Astrocytoma",
    "Urothélial": "Urothelial",
    "Carcinome Adénoïde Kystique": "Adenoid cystic carcinoma",
    "RH+ HER2-": "RH+ HER2-",
    "Adénocarcinome des canaux salivaires": "Salivary duct adenocarcinoma",
    "Neuroendocrine": "Neuroendocrine",
    "Carcinome Basaloïde": "Basaloid carcinoma",
    "grande cellules peu différencié": "large cell carcinoma",
    "RH+ HER2+": "RH+ HER2+",
    "Schwannome vestibulaire": "Vestibular schwannoma",
    "Méningiome atypique": "Atypical meningioma",
    "Triple-": "Triple -",
    "Carcinome à Cellules claires": "Clear cell carcinoma",
    "Carcinoïde atypique": "Atypical carcinoid"
}

renamer_Primary_hist_2 = {
    "Épidermoïde": "Squamous cell",
    "Adénocarcinome": "Adenocarcinoma",
    "Carcinome NST": "NST carcinoma",
    "RH+ HER2 inconnu": "RH+ HER2",
    "Carcinome épidermoïde kératinisant": "Keratinizing squamous cell carcinoma",
    "RH+ HER2-": "RH+ HER2-",
    "Carcinome triple négatif": "Triple -"
}

supplement_df["Primary_hist_2"] = supplement_df["Primary_hist_2"].map(renamer_Primary_hist_2)
supplement_df["Primary_hist_1"] = supplement_df["Primary_hist_1"].map(renamer_Primary_hist_1)
supplement_df["Primary_loc_1"] = supplement_df["Primary_loc_1"].map(renamer_Primary_loc_1)
supplement_df["Primary_loc_2"] = supplement_df["Primary_loc_2"].map(renamer_Primary_loc_2)



supplement_df.to_csv(supplement_path.parent/('cleaned_'+supplement_path.name))

for i in main_df.index:
    patient_id = i.split(':')[0]
    if patient_id in supplement_df.index:
        for col in supplement_df.columns:
            main_df.loc[i, col] = supplement_df.loc[patient_id, col]


main_df.to_csv(output_path)

['Sex',	'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_loc_2', 'Primary_hist_1', 'Primary_hist_2']
