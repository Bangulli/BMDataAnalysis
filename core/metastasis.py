import SimpleITK as sitk
import numpy as np
import copy
import os
import pathlib as pl

def generate_empty_met_from_met(met):
    met = copy.deepcopy(met)
    met.image = None
    met.sitk = None
    met.lesion_volume_voxel = 0
    met.lesion_volume = 0
    met.t1_path = None
    met.t2_path = None
    return met

def load_metastasis(path):
    files = os.listdir(path)
    if files: # check if any files are present
        t1 = None
        t2 = None
        if 't1.nii.gz' in files:
            t1 = path/'t1.nii.gz'
        if 't2.nii.gz' in files:
            t2 = path/'t2.nii.gz'
        bc_source = path/'metastasis_mask_binary.nii.gz'
        mask = sitk.ReadImage(bc_source)
        if 'metastasis_mask_multiclass.nii.gz' in files:
            mc_source = path/'metastasis_mask_multiclass.nii.gz'
        return Metastasis(mask, binary_source=bc_source, multiclass_source=mc_source, t1_path=t1, t2_path=t2)

    else:
        return EmptyMetastasis()

def generate_interpolated_met_from_met(met):
    i_met = InterpolatedMetastasis(met.lesion_volume)
    return i_met

class Metastasis():
    """
    Represents a single metastasis in time
    """
    def __init__(self, metastatis_mask: sitk.Image, binary_source: pl.Path = None, multiclass_source:pl.Path = None, t1_path: pl.Path=None, t2_path: pl.Path=None):
        """
        Instantiates a metastasis from an sitk.Image
        """
        self.sitk = metastatis_mask
        self.image = sitk.GetArrayFromImage(metastatis_mask)
        self.voxel_spacing = metastatis_mask.GetSpacing()
        self.voxel_volume = self.voxel_spacing[0] * self.voxel_spacing[1] * self.voxel_spacing[2]
        self.lesion_size_voxel = np.sum(self.image)
        self.lesion_volume = self.lesion_size_voxel*self.voxel_volume
        self.t1_path = t1_path
        self.t2_path = t2_path
        self.binary_source = binary_source
        self.multiclass_source = multiclass_source

    def same_space(self, met):
        """
        Checks whether a metastasis has the same spacing as self
        """
        return self.voxel_spacing == met.voxel_spacing
    
    def same_voxel_volume(self, met):
        """
        Checks whether two metastases have the same voxel volume
        """
        return self.voxel_volume == met.voxel_volume
    
    def __str__(self):
        return f"lesion volume [mm³] = {self.lesion_volume}"
    
    def print(self):
        print(self.__str__())

    def save(self, path, use_symlinks):
        """
        Saves the mask, t1 and t2 images of a lesion to a given directory
        """
        assert path.is_dir(), "Metastases can only be saved to directories"
        if self.sitk is not None:
            sitk.WriteImage(self.sitk, path/'metastasis_mask_binary.nii.gz')

            if self.multiclass_source is not None: # save three class as well
                msk = sitk.ReadImage(self.multiclass_source)
                msk = sitk.GetArrayFromImage(msk)
                msk[self.image==0]=0
                msk = sitk.GetImageFromArray(msk)
                msk.CopyInformation(self.sitk)
                sitk.WriteImage(msk, path/"metastasis_mask_multiclass.nii.gz")
        if use_symlinks:
            if self.t1_path is not None:
                (path/"t1.nii.gz").symlink_to(self.t1_path)
            if self.t2_path is not None:
                (path/"t2.nii.gz").symlink_to(self.t2_path)


class InterpolatedMetastasis(Metastasis):
    def __init__(self, lesion_volume: float):
        self.lesion_volume = lesion_volume
        
    def __str__(self):
        return f"interpolated lesion volume [mm³] = {self.lesion_volume}"
    
    def save(self, path, use_symlink):
        raise RuntimeError('InterpolatedMetastasis Objects can not be saved, they are missing the image data')
    
class EmptyMetastasis(Metastasis):
    def __init__(self):
        self.image = None
        self.sitk = None
        self.lesion_volume_voxel = 0
        self.lesion_volume = 0
        self.t1_path = None
        self.t2_path = None
        self.spacing = (1, 1, 1)