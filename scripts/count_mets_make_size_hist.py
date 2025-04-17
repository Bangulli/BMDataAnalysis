import csv
import pathlib as pl
import os
import numpy as np
from PrettyPrint import *
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import label, generate_binary_structure, binary_closing, binary_opening

def make_size_histogram(data, filename, clip=500):
    data = np.asarray(data)
    data[data>clip] = clip
    data=data[data<clip]
    counts = np.bincount(data)
    x = np.arange(len(counts))

    plt.figure(figsize=(12, 6))
    plt.bar(x, counts, width=0.6, edgecolor='black') 
    plt.xlabel(f"Metastasis volume [voxel] clipped at {clip}")
    plt.ylabel("Frequency")
    plt.title(f"{filename} Dataset Met Size Histogram")
    plt.grid()
    plt.savefig(f'/home/lorenz/BMDataAnalysis/logs/{filename}.png')

if __name__ == '__main__':
    tasks = {
        '502': 'nnUNet_Datasets/all_singlemod_predictions',
        '504': 'nnUNet_Datasets/singlemod_predictions',
        '524': 'nnUNet_Datasets/multimod_predictions',
        'raw': 'temp_parse_to_gt_comparison'
    }
    task = '504'
    folder = tasks[task]
    labeled_gt_masks = pl.Path(f'/mnt/nas6/data/Target/{folder}')

    sizes_raw = [] # stores the sizes in voxel for the current basis apporach
    sizes_morph_op = [] # stores the sizes in voxel with morph opening (removes small foregorund holes)
    sizes_morph_co = []
    sizes_morph_both = []
    sizes_3con = [] # stores the sizes in voxel with connectivity=3 struct element
    sizes_both = [] # stores the sizes in vocel with both new transformations applied

    struct_2C = generate_binary_structure(3, 2)
    struct_3C = generate_binary_structure(3, 3)

    for mask_file in os.listdir(labeled_gt_masks):
        if not mask_file.endswith('.nii.gz'):
            continue
        mask = sitk.ReadImage(labeled_gt_masks/mask_file)
        # get raw info
        raw = sitk.GetArrayFromImage(mask)
        raw[raw != 0] = 1
        # compute raw labels
        raw_labels, raw_n = label(raw, struct_2C)
        if raw_n == 0: sizes_raw.append(0)
        else:
            for l in range(raw_n):
                sizes_raw.append(np.sum(raw_labels== l+1))
        # compute morph op labels
        morph_opened = binary_opening(raw, struct_3C)
        morph_opened_labels, morph_opened_n = label(morph_opened, struct_2C)
        if morph_opened_n == 0: sizes_morph_op.append(0)
        else:
            for l in range(morph_opened_n):
                sizes_morph_op.append(np.sum(morph_opened_labels==l+1))

        morph_closed = binary_closing(raw, struct_3C)
        morph_closed_labels, morph_closed_n = label(morph_closed, struct_2C)
        if morph_closed_n == 0: sizes_morph_co.append(0)
        else:
            for l in range(morph_closed_n):
                sizes_morph_co.append(np.sum(morph_closed_labels==l+1))

        morph_both = binary_closing(raw, struct_3C)
        morph_both = binary_opening(morph_both, struct_3C)
        morph_both_labels, morph_both_n = label(morph_both, struct_2C)
        if morph_both_n == 0: sizes_morph_both.append(0)
        else:
            for l in range(morph_both_n):
                sizes_morph_both.append(np.sum(morph_both_labels==l+1))
        
        # compute 3 connectivity labels
        connect3_labels, connect3_n = label(raw, struct_3C)
        if connect3_n == 0: sizes_3con.append(0)
        else:
            for l in range(connect3_n):
                sizes_3con.append(np.sum(connect3_labels== l+1))
        # compute labels with both changes
        morph3_labels, morph3_n = label(morph_both, struct_3C)
        if morph3_n == 0: sizes_both.append(0)
        else:
            for l in range(morph3_n):
                sizes_both.append(np.sum(morph3_labels== l+1))
        print(f"File {mask_file} has objects:\n    {raw_n}=default\n    {morph3_n}=with morph\n    {connect3_n}=with 3 connectivity labeling\n    {morph3_n}=with both changes")

        
        
    with open(f"/home/lorenz/BMDataAnalysis/logs/{task}.txt", "w") as f:
        f.write(f"Found {len([o for o in sizes_raw if o != 0])} objects in mode = default\n")
        f.write(f"Found {len([o for o in sizes_morph_op if o != 0])} objects in mode = only morphology open\n")
        f.write(f"Found {len([o for o in sizes_morph_co if o != 0])} objects in mode = only morphology close\n")
        f.write(f"Found {len([o for o in sizes_morph_both if o != 0])} objects in mode = only morphology both\n")
        f.write(f"Found {len([o for o in sizes_3con if o != 0])} objects in mode = only connect 3 labeling\n")
        f.write(f"Found {len([o for o in sizes_both if o != 0])} objects in mode = moprh both and connect 3 labeling\n")


    make_size_histogram(sizes_raw, f'{task}_Default')
    make_size_histogram(sizes_morph_op, f'{task}_Morph_open')
    make_size_histogram(sizes_morph_co, f'{task}_Morph_close')
    make_size_histogram(sizes_morph_both, f'{task}_Morph_both')
    make_size_histogram(sizes_3con, f'{task}_Connect3')
    make_size_histogram(sizes_both, f'{task}_MorprhCon3')



