import os
import pathlib as pl
import pandas as pd
import csv
from datetime import datetime
import re
import pydicom

def sort_directories(dirs, pattern): # courtesy of chatgpt
        """
        Finds directories in the specified base_dir that match the pattern
        ses-yyyymmddhhmmss, parses the timestamp, and returns a list of directory
        names sorted in chronological order.
        """
        # List all directories in the base directory
        matching_dirs = []
        for d in dirs:
            match = re.match(pattern, d)
            if match:
                timestamp_str = match.group(1)
                # Convert the timestamp string to a datetime object
                dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                matching_dirs.append((d, dt))
        # Sort the directories based on the datetime objects
        matching_dirs.sort(key=lambda x: x[1])
  
        return matching_dirs

def safe_get(dicom, tag, cast_fn=str, default=None):
    try:
        value = dicom[tag].value
        if value == '' or value is None:
            return default
        if tag=='PatientAge': value=value.replace('Y','')
        return cast_fn(value)
    except KeyError:
        return default
    except Exception as e:
        print(f"Failed to read {tag}: {e}")
        return default


metastases = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502')
processed = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PROCESSED')
raw_dicom = pl.Path('/mnt/nas6/data/Target/mrct1000_nobatch')
output = metastases/'dataset_description_complete.csv'
primaries = pd.read_excel('/home/lorenz/BMDataAnalysis/logs/Patients_BM_DIAGNOSTIC_coded.xlsx')
primaries['patient_id'] = primaries['patient_id'].str.replace('PAT-', 'PAT')

patients = [p for p in os.listdir(metastases) if (metastases/p).is_dir() and p.startswith('sub-PAT')]

with open(output, 'w') as file:
    #          done           done   done        done           done   done     done       done        done                done       done
    columns = ['Patient ID', 'Sex', 'Age@Onset', 'Age@Offset', 'Age', 'Weight', 'Height', '#Lesions', 'Observation Span', 'Studies', 'Avg Study Interval', 'Primary_loc_1', 'Primary_loc_2', 'Primary_hist_1', 'Primary_hist_2']
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    for pat in patients:
        print(f"Working on patient {pat}")
        line = {}
        line['Patient ID'] = pat
        line['#Lesions'] = len([m for m in os.listdir(metastases/pat) if m.startswith('Metastasis')])

        prim = primaries[primaries['patient_id']==pat]
        line['Primary_loc_1'] = prim['Localisation'].values[0] if prim['Localisation'].values else None
        line['Primary_loc_2'] = prim['Localisation 2'].values[0] if prim['Localisation 2'].values else None
        line['Primary_hist_1'] = prim['Histologie'].values[0] if prim['Histologie'].values else None
        line['Primary_hist_2'] = prim['Histologie 2'].values[0] if prim['Histologie 2'].values else None

        dates = sort_directories([d for d in os.listdir(processed/pat) if os.path.isdir(os.path.join(processed/pat, d))], r"^ses-(\d{14})$")

        onset_date = dates[0]
        offset_date = dates[-1]

        line['Observation Span'] = (offset_date[1]-onset_date[1]).days
        line['Studies'] = len(dates)
        line['Avg Study Interval'] = line['Observation Span']/line['Studies']

        raw_patient_filename = pat.replace('PAT', 'PAT-')

        onset_file = raw_dicom/raw_patient_filename/dates[0][0]
        onset_studies = [f for f in os.listdir(onset_file) if (onset_file/f).is_dir()]
        for study in onset_studies:

            keys = list(line.keys())
            onset_series = onset_file/study/os.listdir(onset_file/study)[0]
            onset_dicom = pydicom.read_file(onset_series)
            sex = safe_get(onset_dicom, 'PatientSex', str)
            if 'Sex' not in keys and sex is not None: line['Sex']=sex    
            height = safe_get(onset_dicom, 'PatientSize', float)
            if 'Height' not in keys and height is not None: line['Height']=height   
            weight = safe_get(onset_dicom, 'PatientWeight', float)
            if 'Weight' not in keys and weight is not None: line['Weight']=weight 
            age = safe_get(onset_dicom, 'PatientAge', float)
            if 'Age@Onset' not in keys and age is not None: line['Age@Onset']=age 


        offset_file = raw_dicom/raw_patient_filename/dates[-1][0]
        offset_studies = [f for f in os.listdir(offset_file) if (offset_file/f).is_dir()]
        for study in offset_studies:

            keys = list(line.keys())
            offset_series = offset_file/study/os.listdir(offset_file/study)[0]
            offset_dicom = pydicom.read_file(offset_series)

            sex = safe_get(offset_dicom, 'PatientSex', str)
            if 'Sex' not in keys and sex is not None: line['Sex']=sex    
            height = safe_get(offset_dicom, 'PatientSize', float)
            if 'Height' not in keys and height is not None: line['Height']=height   
            weight = safe_get(offset_dicom, 'PatientWeight', float)
            if 'Weight' not in keys and weight is not None: line['Weight']=weight 
            age = safe_get(offset_dicom, 'PatientAge', float)
            if 'Age@Offset' not in keys and age is not None: line['Age@Offset']=age 


        if 'Age@Offset' in line.keys() and 'Age@Onset' in line.keys(): line['Age'] = (line['Age@Offset']+line['Age@Onset'])/2

        writer.writerow(line)


