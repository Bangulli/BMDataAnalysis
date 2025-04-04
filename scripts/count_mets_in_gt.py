import csv
import pathlib as pl
import os
import numpy as np
from PrettyPrint import *
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import label, generate_binary_structure, binary_closing

def make_size_histogram(data, filename, clip=2000):
    data[data>clip] = clip
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
    labeled_gt_masks = pl.Path('/mnt/nas6/data/Target/temp_parse_to_gt_comparison')

    sizes_raw = [] # stores the sizes in voxel for the current basis apporach
    sizes_morph = [] # stores the sizes in voxel with morph opening (removes small foregorund holes)
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
        morphed = binary_closing(raw, struct_3C)
        morphed_labels, morphed_n = label(morphed, struct_2C)
        if morphed_n == 0: sizes_morph.append(0)
        else:
            for l in range(morphed_n):
                sizes_morph.append(np.sum(morphed_labels==l+1))
        # compute 3 connectivity labels
        connect3_labels, connect3_n = label(raw, struct_3C)
        if connect3_n == 0: sizes_3con.append(0)
        else:
            for l in range(connect3_n):
                sizes_3con.append(np.sum(connect3_labels== l+1))
        # compute labels with both changes
        morph3_labels, morph3_n = label(morphed, struct_3C)
        if morph3_n == 0: sizes_both.append(0)
        else:
            for l in range(morph3_n):
                sizes_both.append(np.sum(morph3_labels== l+1))
        print(f"File {mask_file} has objects:\n    {raw_n}=default\n    {morph3_n}=with morph\n    {connect3_n}=with 3 connectivity labeling\n    {morph3_n}=with both changes")

        
        
    print(f"Found {len([o for o in sizes_raw if o != 0])} objects in mode = default")
    print(f"Found {len([o for o in sizes_morph if o != 0])} objects in mode = only morphology")
    print(f"Found {len([o for o in sizes_3con if o != 0])} objects in mode = only connect 3 labeling")
    print(f"Found {len([o for o in sizes_both if o != 0])} objects in mode = moprh and connect 3 labeling")

    sizes_raw = np.asarray(sizes_raw) # stores the sizes in voxel for the current basis apporach
    sizes_morph = np.asarray(sizes_morph) # stores the sizes in voxel with morph opening (removes small foregorund holes)
    sizes_3con = np.asarray(sizes_3con) # stores the sizes in voxel with connectivity=3 struct element
    sizes_both = np.asarray(sizes_both) # stores the sizes in vocel with both new transformations applied

    make_size_histogram(sizes_raw, 'Default')
    make_size_histogram(sizes_morph, 'Morpho')
    make_size_histogram(sizes_3con, 'Connect3')
    make_size_histogram(sizes_both, 'MorprhCon3')


    
