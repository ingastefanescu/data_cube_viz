# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 00:42:59 2024

@author: Andrei
"""

from PIL import Image
import os
import scipy
import mat73

path = os.getcwd()
im = Image.open(path + '\MSI_SITS_GIS.tif')

msi = mat73.loadmat('msi.mat')