import SimpleITK as sitk
import numpy as np
import copy
import os
import pathlib as pl
from radiomics import featureextractor
import logging

radiomics_logger = logging.getLogger("radiomics")
radiomics_logger.setLevel(logging.CRITICAL)

def generate_empty_met_from_met(met):
    emet = EmptyMetastasis()
    emet.t1_path = met.t1_path
    emet.t2_path = met.t2_path
    emet.voxel_spacing = met.voxel_spacing
    return emet

def load_metastasis(path):
    files = os.listdir(path)
    if files: # check if any files are present
        t1 = None
        t2 = None
        if 't1.nii.gz' in files:
            t1 = path/'t1.nii.gz'
        if 't2.nii.gz' in files:
            t2 = path/'t2.nii.gz'

        if 'metastasis_mask_binary.nii.gz' in files:
            bc_source = path/'metastasis_mask_binary.nii.gz'
            mask = sitk.ReadImage(bc_source)
            mc_source=None
            if 'metastasis_mask_multiclass.nii.gz' in files:
                mc_source = path/'metastasis_mask_multiclass.nii.gz'
            return Metastasis(mask, binary_source=bc_source, multiclass_source=mc_source, t1_path=t1, t2_path=t2)

        else:
            return EmptyMetastasis(t1, t2)
    else: return EmptyMetastasis()

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
                sitk.WriteImage(msk, path/"metastasis_mask_multiclass.nii.gz")

        if use_symlinks:
            if self.t1_path is not None:
                (path/"t1.nii.gz").symlink_to(self.t1_path)
            if self.t2_path is not None:
                (path/"t2.nii.gz").symlink_to(self.t2_path)

    def rano(self, baseline, nadir, mode='3d'):
        """
        Returns the RANO-BM classification for the Metastasis, given the basline and nadir values from the series
        """
        if baseline<nadir:
            print("Values for autoread RANO incorrect: baseline < nadir")

        if self.lesion_volume == 0:
            return 'CR'
            
        ratio_baseline = self.lesion_volume/baseline
        ratio_nadir = self.lesion_volume/nadir
        if mode == '1d':
            th1 = 0.7
            th2 = 1.2
        elif mode == '3d':
            th1 = 0.343
            th2 = 1.728

        if ratio_baseline<=th1:
            response='PR'
        elif ratio_nadir<th2:
            response='SD'
        else:
            response='PD'
        return response
    
    def resample(self, ref):
        """
        Resamples the underlying images
        for now unused but could be useful later
        """
        if not isinstance(ref, sitk.Image):
            self.sitk = sitk.Resample(self.sitk, referenceImage=ref.sitk)
            self.image = sitk.GetArrayFromImage(self.sitk)
            self.voxel_spacing = self.sitk.GetSpacing()
            self.voxel_volume = self.voxel_spacing[0] * self.voxel_spacing[1] * self.voxel_spacing[2]
            self.lesion_size_voxel = np.sum(self.image)
            self.lesion_volume = self.lesion_size_voxel*self.voxel_volume
        else:
            self.sitk = sitk.Resample(self.sitk, referenceImage=ref)
            self.image = sitk.GetArrayFromImage(self.sitk)
            self.voxel_spacing = self.sitk.GetSpacing()
            self.voxel_volume = self.voxel_spacing[0] * self.voxel_spacing[1] * self.voxel_spacing[2]
            self.lesion_size_voxel = np.sum(self.image)
            self.lesion_volume = self.lesion_size_voxel*self.voxel_volume

    def get_t1_radiomics(self):
        if self.t1_path is not None and self.sitk is not None and sitk.GetArrayFromImage(self.sitk).any():
            extractor = featureextractor.RadiomicsFeatureExtractor()
            extractor.enableAllFeatures()
            extractor.enableAllImageTypes()
            #extractor.settings['binwidth'] = 100
            #print(type(self.sitk), self.t1_path)
            arr = sitk.GetArrayFromImage(self.sitk)
            if arr.max == np.iinfo(arr.dtype).max: arr[arr==arr.max]=0
            arr = sitk.GetImageFromArray(arr)
            arr.CopyInformation(self.sitk)
            self.sitk = arr
            mask = self.sitk if self.sitk is not None else self.binary_source
            t1_radiomics = extractor.execute(str(self.t1_path), sitk.BinaryDilate(mask, 
                            kernelRadius=(1,1,1),
                            kernelType=sitk.sitkBall,  # or Cross, Box, Annulus
                            foregroundValue=1))
            return t1_radiomics
        else: return False

    def set_total_lesion_load(self, count, load):
        self.count=count
        self.load=load
        


class InterpolatedMetastasis(Metastasis):
    def __init__(self, lesion_volume: float):
        self.lesion_volume = lesion_volume
        
    def __str__(self):
        return f"interpolated lesion volume [mm³] = {self.lesion_volume}"
    
    def save(self, path, use_symlink):
        raise RuntimeError('InterpolatedMetastasis Objects can not be saved, they are missing the image data')
    
class EmptyMetastasis(Metastasis):
    def __init__(self, t1_path=None, t2_path=None):
        self.image = None
        self.sitk = None
        self.lesion_volume_voxel = 0
        self.lesion_volume = 0
        self.t1_path = t1_path
        self.t2_path = t2_path
        self.voxel_spacing = (1, 1, 1)