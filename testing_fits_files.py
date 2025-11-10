import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits
import glob
import os

import astropy.units as u
import astropy.wcs as WCS
from astropy.coordinates import SkyCoord

from astropy.visualization import ZScaleInterval

from trim import Trim
from test_dark_subtract import Bias
from test_flat_divide import Flat

def find_ceph(ceph_folder, ceph_nums):
    """
    Using glob to filter through a folder which contains the observations from one night
    this will Return all of the cepheid images taken (we only included cepheids if there
    was 5 images taken of them)
    """
    ceph_files = glob.glob(os.path.join(ceph_folder, "*.fits"))

    filtered_files = []
    for file in ceph_files:
        filename = os.path.basename(file)
        if any(f"Cepheids_{num}_" in filename for num in ceph_nums):
            filtered_files.append(file)

    return filtered_files

if __name__ == "__main__":
    
    #read in the bias frames NOTE: this is the 1x1 binning bias frames for week 1
    bias_frame1 = fits.open("/home/s2407710/Documents/tgp/bias_2025_09_20/PIRATE_161197_Bias11_0_2025_09_20_18_25_20.fits")[0].data
    bias_frame2 = fits.open("/home/s2407710/Documents/tgp/bias_2025_09_20/PIRATE_161208_Bias11_1_2025_09_20_18_36_04.fits")[0].data
    bias_frame3 = fits.open("/home/s2407710/Documents/tgp/bias_2025_09_20/PIRATE_161209_Bias11_2_2025_09_20_18_36_09.fits")[0].data
    bias_frame4 = fits.open("/home/s2407710/Documents/tgp/bias_2025_09_20/PIRATE_161210_Bias11_3_2025_09_20_18_36_15.fits")[0].data

    #read in the flat frames NOTE: this is the 1x1 binning flat frames for week 1
    flat_frame1 = fits.open("/home/s2407710/Documents/tgp/flat_V_2025_09_04/PIRATE_159244_flats_V_01_2025_09_04_06_28_52.fits")[0].data
    flat_frame2 = fits.open("/home/s2407710/Documents/tgp/flat_V_2025_09_04/PIRATE_159243_flats_V_02_2025_09_04_06_28_05.fits")[0].data
    flat_frame3 = fits.open("/home/s2407710/Documents/tgp/flat_V_2025_09_04/PIRATE_159242_flats_V_03_2025_09_04_06_27_20.fits")[0].data
    flat_frame4 = fits.open("/home/s2407710/Documents/tgp/flat_V_2025_09_04/PIRATE_159241_flats_V_04_2025_09_04_06_26_35.fits")[0].data

    #the path to the folder where the cepheid data is stored (in this case this is the path to week 1's data)
    ceph_folder_path = "/home/s2407710/Documents/tgp/22_09_cepheid1"

    #numbers of all the cepheids observed on night 1 (there were not enough observations for cepheid 1 on night 1)
    ceph_nums = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]

    #a folder to store the folders of the different cepheids
    all_corrected = {}

    #filter through the different cepheids, this will allow a stacked cepheid image
    #for each cepheid to be made
    for num in ceph_nums:
        ceph_files = find_ceph(ceph_folder_path, [num])
        corrected_images = []

        #correct each image of a specific number, then stack the images and take the mean
        for ceph_image in ceph_files:
            ceph_data = fits.open(ceph_image)[0].data
            bias_correction = Bias(bias_frame1, bias_frame2, bias_frame3, bias_frame4, ceph_data)
            bias_image = bias_correction.subtraction()
            flat_correction = Flat(flat_frame1, flat_frame2, flat_frame3, flat_frame4, bias_image)
            correct_image = flat_correction.flat_divide()
            corrected_images.append(correct_image)
        stacked_cepheid = np.stack(corrected_images, axis=0)
        mean_cepheid = np.mean(stacked_cepheid, axis=0)
        all_corrected[num] = corrected_images
        
    #this plots the stacked images along the Z-scale
    zscale = ZScaleInterval()

    fig, axes = plt.subplots(1, len(all_corrected), figsize=(5 * len(all_corrected), 5))

    if len(all_corrected) == 1:
        axes = [axes]

    for ax, (ceph_num, images) in zip(axes, all_corrected.items()):
        mean_img = np.mean(images, axis=0)
        vmin, vmax = zscale.get_limits(mean_img)
        ax.imshow(mean_img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"Cepheid {ceph_num}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
