import warnings
warnings.filterwarnings('ignore')
import os
from os.path import join
import glob
import re
import numpy
import pandas as pd
import ants
import numpy as np
from scipy import ndimage
import pathlib as pl


#import metastasis

def find_closest_label(lesion_mask, labels): # local method because this code is a bit more complex and i dont want to split it into sub functions. Would be better to just create its own script for this
            """
            Find the closest label in the 'labels' array to the connected component represented by the boolean 'lesion_mask'.

            Parameters:
                lesion_mask (numpy array): Boolean numpy array representing a single connected component.
                labels (numpy array): Numpy array with different labels (different int values).

            Returns:
                closest_label (int): The label of the closest label in 'labels' to 'lesion_mask'.
            """
            # Calculate the distance to the nearest True pixel for each pixel in 'lesion_mask'
            distances = ndimage.distance_transform_cdt(~lesion_mask)

            # Find the label in 'labels' that is closest to 'lesion_mask'
            unique_labels = np.unique(labels)
            closest_label = int(unique_labels[np.argmin([np.mean(distances[labels == label]) for label in unique_labels])])

            return closest_label


def find_regions(imt1, lesion_mask, im_brain):
        ## read atlas and labels images from nas. i hope theyre still in that location
        path_atlas = "/mnt/nas4/datasets/ToReadme/USCLobes/"
        im_t1atlas = ants.image_read(join(path_atlas,"BCI-DNI_brain.bfc.nii.gz"))
        im_atlas = ants.image_read(join(path_atlas,"BCI-DNI_brain.label.nii.gz"))
        df_labels = pd.read_xml(join(path_atlas,"brainsuite_labeldescription.xml"))
        df_labels = df_labels[df_labels['id'].isin(np.unique(im_atlas.numpy()))]
        im_hemis = ants.image_read(join(path_atlas,"BCI-DNI_brain.hemi.label.nii.gz"))

        ## read current image and masks
        im_t1 = ants.image_read(str(imt1))
        #print('img', imt1)
        #print('brn', im_brain)
        im_brain = ants.image_read(str(im_brain))
        im_t1 = im_t1*im_brain
        im_t1 = ants.resample_image(im_t1,(1.,1.,1.),interp_type=4)
        lesions_mask = ants.image_read(str(lesion_mask))
        #print('lsn', lesion_mask)

        list_tgt_regions = ['R. Frontal Lobe','L. Frontal Lobe','R. Parietal Lobe','L. Parietal Lobe',
                    'R. Temporal Lobe','L. Temporal Lobe','R. Occipital Lobe','L. Occipital Lobe',
                    'R. Brainstem', 'L. Brainstem']
        list_tgt_labels = [100, 101, 200, 201, 300, 301, 400, 401, 800, 801]

        ## move 2 atlas
        df_labels, im_t1, lesions_mask, np_atlas = to_atlas(im_t1atlas, im_atlas, im_hemis, im_t1, lesions_mask, df_labels)

        ## idk why we reshape but it was in the original code
        lesions_mask = np.flip(np.transpose(lesions_mask,(2,1,0)))
        atlas = np.flip(np.transpose(np_atlas,(2,1,0)))

        orders = []
        regions = []
        
        
        np_l = lesions_mask==1  # Extract each lesion one by one
        order = 1
        
        # Create a mask to filter out non-target regions from the atlas
        mask = np.isin(atlas, list_tgt_labels)
        atlas_tgt_regions = atlas * mask

        # Get labels of regions overlapped by the current lesion
        labels_overlap = np.asarray(atlas[np_l], dtype=int)
        list_labels_overlap = list(labels_overlap)
        
        # Filter out non-target regions
        list_labels_overlap = [x for x in list_labels_overlap if x in list_tgt_labels]

        if len(list_labels_overlap) == 0:
            # If no overlap with target regions, find the closest one
            label = find_closest_label(np_l, atlas_tgt_regions)
            region = df_labels.loc[df_labels['id'] == label, 'fullname'].values[0]
            #print("(No overlap with target regions)")
        else:
            # Determine the most frequent label among the overlapping regions
            label = np.bincount(list_labels_overlap).argmax()
            region = df_labels.loc[df_labels['id'] == label, 'fullname'].values[0]

        print(f"{order:02d} {region} ({label}), overlaps with: {np.unique(labels_overlap)}")

        orders.append(order)
        regions.append(region)
            
        return orders, regions

def to_atlas(atlas_t1, atlas_lbl, atlas_hemis, mov_img, mov_lesion, df_labels, verbose=False):
    # this method joins the hemisphere labels and region labels for better seperation
    # it resamples and registers images and returns all the stuff to the find_regions function
    # it would be more efficient to only run this once and then just reading, but im not a systems engineer so not a prio

    # Fixed image for registration: atlas so that we get a common alignement. 
    # But the Target naming is made on the orientation of the MRI...
    # Set the origin to be closer in space.

    # resample an label
    mov_img.set_origin(ants.get_origin(atlas_t1))
    mov_lesion.set_origin(ants.get_origin(atlas_t1))
    
    # atlas_t1 = ants.resample_image_to_target(atlas_t1,mov_img)
    # atlas_lbl = ants.resample_image_to_target(atlas_lbl,mov_img,"nearestNeighbor")
    # im_hemis = ants.resample_image_to_target(atlas_hemis,mov_img)
    np_hemis = atlas_hemis.numpy()
    np_atlas = atlas_lbl.numpy()
    np_hemisL = np.where(np_hemis == 0, 0, np.where(np_hemis % 2 == 0, 1, 0))
    np_hemisR = np.where(np_hemis % 2 != 0, 1,  0)

    # joint hemispheric and region labels
    list_splitLR = [3,800,850,900]
    for v in list_splitLR:
        np_atlas = np.where(np_atlas == v, np.where(np_hemisR==1,v,v+1), np_atlas)
        new_row = pd.DataFrame({'id': [v + 1], 'fullname': ['L. ' + df_labels[df_labels['id'] == v]['fullname'].values[0]]})
        df_labels.loc[df_labels['id'] == v, 'fullname'] = 'R. ' + df_labels[df_labels['id'] == v]['fullname'].values[0]
        df_labels = pd.concat([df_labels, new_row], ignore_index=True)
    im_atlas = ants.from_numpy(np_atlas, origin=atlas_lbl.origin, spacing=atlas_lbl.spacing, direction=atlas_lbl.direction)
    df_labels = df_labels.sort_values(by=['id'])

    if verbose: mov_img.plot(overlay=atlas_t1, title='Before Registration', filename='/home/lorenz/BMDataAnalysis/before.png')
    # perofrm reg and trans
    reg = ants.registration(atlas_t1, mov_img, 'TRSAA') # Affine, SyNCC
    im_t1_reg = reg['warpedmovout']
    im_pred_reg = ants.apply_transforms(fixed = atlas_t1, moving = mov_lesion, transformlist = reg['fwdtransforms'], interpolator  = 'nearestNeighbor')
    
    #      labels     trans img  trans lesion the atlas. 
    if verbose: im_t1_reg.plot(overlay=atlas_t1, title='After Registration', filename='/home/lorenz/BMDataAnalysis/after.png')
    # ants.image_write(im_t1_reg, 't1_reg.nii.gz')
    # ants.image_write(im_pred_reg, 'lesion_reg.nii.gz')
    # ants.image_write(atlas_t1, 'atlas_t1.nii.gz')
    # ants.image_write(atlas_lbl, 'atlas_lbl.nii.gz')
    np_pred_reg = im_pred_reg.numpy()
    return df_labels, im_t1_reg, np_pred_reg, np_atlas

     
if __name__ == '__main__':
    met = metastasis.load_metastasis(pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/sub-PAT0741/Metastasis 0/t0 - 20170822113221'))
    print(met.get_location_in_brain())