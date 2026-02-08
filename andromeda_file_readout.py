from Cepheid_apertures import AperturePhotometry

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

photomin_path = r"/storage/teaching/TelescopeGroupProject/Andromeda_LT_Data/GaiaPhotomIn.Dat"
photomout_path = r"/storage/teaching/TelescopeGroupProject/Andromeda_LT_Data/GaiaPhotomOut.Dat"

#read in .Dat files (ASCII text)
photomin = pd.read_csv(photomin_path, delimiter = " ", header=None)
photomout = pd.read_csv(photomout_path, delimiter = " ", header=None)

example_fits_path = r"/storage/teaching/TelescopeGroupProject/Andromeda_LT_Data/h_e_20170613_128_1_1_1.fits"

example_fits_path0 = r"/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/2025-10-06/cepheid_07_stacked.fits"

andro_ex = AperturePhotometry(example_fits_path)
coords = andro_ex.get_pixel_coords(10.36375, 41.1695556)
#print(f"Pixel coordinates: {coords}")
data=  andro_ex.mask_data_and_plot(coords[0], coords[1],2000, plot = True )
#centroid, fwhm = andro_ex.get_centroid_and_fwhm(data, 1000)
#print(f"Centroid: {centroid}, FWHM: {fwhm}")"""
#print(andro_ex.header["FILTER1"])