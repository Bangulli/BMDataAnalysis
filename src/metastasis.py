import SimpleITK as sitk
import numpy as np
import copy
import pathlib as pl

def generate_empty_met_from_met(met):
    met = copy.deepcopy(met)
    met.image[met.image!=0]=0
    met.lesion_volume_voxel = 0
    met.lesion_volume = 0
    return met

class Metastasis():
    """
    Represents a single metastasis in time
    """
    def __init__(self, metastatis_mask = sitk.Image):
        self.image = sitk.GetArrayFromImage(metastatis_mask)
        self.voxel_spacing = metastatis_mask.GetSpacing()
        self.voxel_volume = self.voxel_spacing[0] * self.voxel_spacing[1] * self.voxel_spacing[2]
        self.lesion_volume_voxel = np.sum(self.image)
        self.lesion_volume = self.lesion_volume_voxel*self.voxel_volume

    def same_space(self, met):
        return self.voxel_spacing == met.voxel_spacing
    
    def __str__(self):
        return f"lesion volume [mm³] = {self.lesion_volume}"
    
    def print(self):
        print(self.__str__())