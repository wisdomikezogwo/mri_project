import nibabel as nib
import numpy as np
import os

modalities = ['flair', 'seg', 't1', 't1ce', 't2']

ex_img = nib.load(f'~./MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/'
                  f'BraTS20_Training_001_{modalities[0]}.nii.gz')

