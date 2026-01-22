import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits
import glob
import os
import re
from collections import defaultdict
import astropy.units as u
import astropy.wcs as WCS
from astropy.coordinates import SkyCoord

from datetime import datetime, timedelta

from astropy.visualization import ZScaleInterval

from trim import Trim
from test_dark_subtract import Bias
from test_flat_divide import Flat
from opening_files import CepheidFileGrouper


def select_time_coherent_stack(filenames, max_dt_minutes=2, max_stack_size=5):
    """
    Selects the largest group of images per Cepheid number
    that lie within ±max_dt_minutes of each other.
    """

    cep_pattern = re.compile(
        r"Cepheids_(\d+)_00_.*?"
        r"_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.fits$"
    )

    cep_groups = defaultdict(list)

    # --- Parse filenames ---
    for f in filenames:
        name = os.path.basename(f)
        m = cep_pattern.search(name)
        if not m:
            continue

        cep_id = m.group(1)
        year, month, day, hour, minute, second = map(int, m.groups()[1:])
        timestamp = datetime(year, month, day, hour, minute, second)

        cep_groups[cep_id].append((f, timestamp))

    TIME_WINDOW = timedelta(minutes=max_dt_minutes)
    final_stacks = {}

    # --- Find best stack per Cepheid ---
    for cep_id, files in cep_groups.items():
        files.sort(key=lambda x: x[1])

        best_stack = []

        for i in range(len(files)):
            start_time = files[i][1]
            #current_stack = [files[i][0]]
            current_stack = [files[i]]

            for j in range(i + 1, len(files)):
                if files[j][1] - start_time <= TIME_WINDOW:
                    current_stack.append(files[j])
                else:
                    break

            if len(current_stack) > len(best_stack):
                best_stack = current_stack


        # Ignore Cepheids with no coherent images
        if len(best_stack) < 2:
            continue

        # Find median time
        times = [t for _, t in best_stack]
        median_time = sorted(times)[len(times) // 2]

        # Keep images closest in time
        best_stack.sort(key=lambda x: abs(x[1] - median_time))
        best_stack = best_stack[:max_stack_size]

        final_stacks[cep_id] = [f for f, _ in best_stack]

        print(f"Cepheid {cep_id} → selected {len(best_stack)} files:")
        for f, _ in best_stack:
            print(f"  {os.path.basename(f)}")

    return final_stacks


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

    base_path = "/storage/teaching/TelescopeGroupProject/2025-26/Cepheids/2025-09-22/"

    # Gather all FITS files
    all_fits = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(".fits"):
                all_fits.append(os.path.join(root, f))

    grouper = CepheidFileGrouper(base_path)
    final_stacks = select_time_coherent_stack(all_fits, max_dt_minutes=2)
    all_corrected = {}

    for number, files in final_stacks.items():
        print(f"Cepheid {number}: {len(files)} images")

    #filter through the different cepheids, this will allow a stacked cepheid image
    #for each cepheid to be made
    for num, ceph_files in final_stacks.items():
        corrected_images = []
        total_exposure = 0
        total_gain = []
        total_read_noise = []


        with fits.open(ceph_files[0]) as hdul0:
            base_header = hdul0[0].header.copy()

        #correct each image of a specific number, then stack the images and take the mean
        for ceph_image in ceph_files:
            with fits.open(ceph_image) as hdul:

                #opening the fits file and getting the data and header
                ceph_data = hdul[0].data 
                header = hdul[0].header

                #getting the exposure time for each image of the cepheid, adding this to total_exposure
                exposure = header.get("EXPOSURE", 0)
                total_exposure += exposure

                #getting the gain for each cepheid image, adding to list total_gain
                gain = header.get("GAIN", 0)
                total_gain.append(gain)

                #getting the read noise for each cepheid image, adding to list total_read_noise
                read_noise = header.get("RDNOISE", 0)
                total_read_noise.append(read_noise)

            #ceph_data = fits.open(ceph_image)[0].data
            bias_correction = Bias(bias_frame1, bias_frame2, bias_frame3, bias_frame4, ceph_data)
            bias_image = bias_correction.subtraction()
            flat_correction = Flat(flat_frame1, flat_frame2, flat_frame3, flat_frame4, bias_image)
            correct_image = flat_correction.flat_divide()
            corrected_images.append(correct_image)

        stacked_cepheid = np.stack(corrected_images, axis=0)
        mean_cepheid = np.mean(stacked_cepheid, axis=0)
        #check this line
        all_corrected[num] = mean_cepheid

        base_header["TOTEXP"]  = total_exposure
        base_header["TOTGAIN"] = float(np.mean(total_gain))
        base_header["TOTRN"]   = float(np.sqrt(np.sum(np.array(total_read_noise)**2)))
        base_header["NSTACK"]  = len(ceph_files)
        base_header["COMMENT"] = f"Stacked {len(ceph_files)} images for Cepheid {num}"

        """
        #this part saves the stacked cepheid images as a fits file to a folder
        out_dir = "/home/s2407710/Documents/tgp/stacked_cepheids_22_09"
        os.makedirs(out_dir, exist_ok=True)

        save_path = os.path.join(out_dir, f"cepheid_{num}_stacked.fits")
       # fits.PrimaryHDU(mean_cepheid).writeto(save_path, overwrite=True)

        hdu = fits.PrimaryHDU(mean_cepheid, header=base_header)
        hdu.writeto(save_path, overwrite=True)

        print(f"Saved stacked Cepheid {num} to {save_path}")
        """
     
        #this plots the stacked images along the Z-scale
    zscale = ZScaleInterval()

    fig, axes = plt.subplots(1, len(all_corrected), figsize=(5 * len(all_corrected), 5))

    if len(all_corrected) == 1:
        axes = [axes]

    for ax, (ceph_num, images) in zip(axes, all_corrected.items()):
        vmin, vmax = zscale.get_limits(images)
        ax.imshow(images, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"Cepheid {ceph_num}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    

    """
        hdu = fits.PrimaryHDU(mean_cepheid)
        # Add a header keyword for total exposure time, making the title for the total exposure time
        hdu.header["TOTEXP"] = total_exposure
        #taking average of the gain values, adding a header for this
        avg_gain = np.mean(total_gain)
        hdu.header["TOTGAIN"] = avg_gain
        #adding the read noises in quadrature, adding this to fits file
        read_noise_vals = np.array(total_read_noise)
        read_noise_add = np.sqrt(np.sum(read_noise_vals**2))
        hdu.header["TOTRN"] = read_noise_add
        """
    

    """
def find_ceph(ceph_folder, ceph_nums):
    """
    #Using glob to filter through a folder which contains the observations from one night
    #this will Return all of the cepheid images taken (we only included cepheids if there
    #was 5 images taken of them)
    """
    ceph_files = glob.glob(os.path.join(ceph_folder, "*.fits"))

    filtered_files = []
    for file in ceph_files:
        filename = os.path.basename(file)
        if any(f"Cepheids_{num}_" in filename for num in ceph_nums):
            filtered_files.append(file)

    return filtered_files
"""
"""
def select_last_stack(filenames, max_last=5):
    """
    #filenames: list of all FITS filenames (full paths or just names)
    #max_last: maximum number of images to keep per stack
    #Returns: dict mapping Cepheid number -> list of selected files
"""

    cep_pattern = re.compile(r"Cepheids_(\d+)_00")
    cep_groups = defaultdict(list)

    # Step 1: group by numeric Cepheid number
    for f in filenames:
        m = cep_pattern.search(f)
        if m:
            cep_id = m.group(1)
            cep_groups[cep_id].append(f)

    selected = {}

    # Step 2: sort each group by timestamp (filename order works)
    for cep_id, files in cep_groups.items():
        files_sorted = sorted(files)  # timestamp is at the end; lex sort works
        # Step 3: take only last stack (assuming last contiguous group is the last stack)
        # We'll assume consecutive filenames belong to a stack
        # Find contiguous blocks by index gaps
        # For simplicity, we just take the **last n files**
        if len(files_sorted) > max_last:
            selected_files = files_sorted[-max_last:]
        else:
            selected_files = files_sorted
        selected[cep_id] = selected_files

        print(f"Cepheid {cep_id} → selecting {len(selected_files)} files:")
        for f in selected_files:
            print(f"  {os.path.basename(f)}")

    return selected
"""