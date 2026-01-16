import numpy as np
from astropy.io import fits
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re
import glob

class Calibration_Set:
    """
    Returns a master flat and bias for a given week+day configuration.
    """
    def __init__(self, name, bias_dir, flat_dir):
        self.name = name
        self.bias_dir = Path(bias_dir) if bias_dir else None
        self.flat_dir = Path(flat_dir) if flat_dir else None

        self.bias_files = []
        self.flat_files = []

        self.master_bias = None
        self.master_flat = None

    @staticmethod
    def img_trim(img):
        """
        Trims image by desired number of pixels.
        """
        trim = 50 # decided on 50 pixels
        return img[trim:-trim, trim:-trim]
    
    def create_master_bias(self):
        """
        Create master bias image from the stored bias file list.
        """
        if self.master_bias is not None:  # if this already exists
            return self.master_bias
        
        # Use the stored bias_files list instead of globbing
        if self.bias_files is None or len(self.bias_files) == 0:
            raise ValueError(f"No bias files specified for {self.name}")
        
        bias_files = sorted(self.bias_files)
        print(f"{len(bias_files)} bias files found in {self.bias_dir}.")

        bias_frames = [self.img_trim(fits.getdata(f)) for f in bias_files]
        bias_stack = np.stack(bias_frames, axis=0)
        self.master_bias = np.median(bias_stack, axis=0)
        print(f"Master bias shape: {self.master_bias.shape}")

        return self.master_bias

    def create_master_flat(self):
        """
        Create master flat image from the stored flat file list.
        """
        if self.master_flat is not None:  # if this already exists
            return self.master_flat

        if self.master_bias is None:
            self.create_master_bias() # since the flat-fielded images are bias-corrected we check whether master bias actually exists

        # Use the stored flat_files list instead of globbing
        if self.flat_files is None or len(self.flat_files) == 0:
            raise ValueError(f"No flat files specified for {self.name}")

        flat_files = sorted(self.flat_files)
        print(f"{len(flat_files)} flat files found in {self.flat_dir}.")

        flat_frames = [
        self.img_trim(fits.getdata(f)) - self.master_bias
        for f in flat_files]

        flat_stack = np.median(flat_frames, axis=0)
        normalisation = np.median(flat_stack)
        self.master_flat = flat_stack / normalisation
        
        print(f"Master flat shape: {self.master_flat.shape}")

        return self.master_flat

    def prepare(self):
        """
        Prepare master bias and flat frames.
        """
        self.create_master_bias()
        self.create_master_flat()
        return self

class Calibration_Manager:
    """
    Organises calibration frames; maps weekly calibration frames to observation nights. 
    """
    def __init__(self, calibrations_dir):
        self.calib_dir = Path(calibrations_dir)
        self.calibration_sets = {}
        self.night_to_calib_map = {}

    def week_calibrations(self, week_name, binning="binning1x1", filter="V"):
        """
        Locate calibration file directories for a specific week.
        Binning is always 1x1 and filter is always V (for PIRATE telescope data)
        """
        week_dir = self.calib_dir / week_name / binning

        if not week_dir.exists():
            print(f"Warning: Calibration directory not found: {week_dir}")
            return None

        print(f"Located directory {week_dir}")

        bias_tag = "Bias"
        flat_tag = "flats"

        bias_files = list(week_dir.glob(f"*{bias_tag}*.fits"))
        flat_files = list(week_dir.glob(f"*{flat_tag}*{filter}*.fits"))

        if len(bias_files) == 0:
            print(f"Warning: No bias files found in {week_dir}")
            return None
    
        if len(flat_files) == 0:
            print(f"Warning: No flat files found for filter {filter} in {week_dir}")
            return None
        
        print(f"Found {len(bias_files)} bias files")
        print(f"Found {len(flat_files)} flat files for filter {filter}")

        # now create the set of calibrations for the given week
        calib_name = f"{week_name}_{filter}_{binning}"
        calib_set = Calibration_Set(calib_name, week_dir, week_dir)

        calib_set.bias_files = sorted(bias_files)
        calib_set.flat_files = sorted(flat_files)
    
        self.calibration_sets[week_name] = calib_set
    
        return calib_set
    
    def map_night_to_week(self, night_name, calib_name):
        """
        Map an observation night to its corresponding calibration set.
        Night name will be a file such as "2025_09_22" (first night)
        Calibration name will be the full calibration identifier (week, filter, binning)""
        """
        self.night_to_calib_map[night_name] = calib_name # assign the nights/weeks to a place in the dictionary in __init__

    def get_calibration_for_night(self, night_name):
        """
        Use map function to acquire the appropriate calibration set for a given night.
        Returns the index in the dictionary containing the correct calibration.
        """
        if night_name not in self.night_to_calib_map:
            raise KeyError(f"No calibration mapping found for night {night_name}")
        
        calib_name = self.night_to_calib_map[night_name]
        
        if calib_name not in self.calibration_sets:
            raise KeyError(f"Calibration set '{calib_name}' not found")
        
        return self.calibration_sets[calib_name]
    
    def prepare_all(self):
        """
        Prepares all master bias and flat images at once.
        """
        # diagnostic prints
        print("\n" + "="*60)
        print("PREPARING CALIBRATION FRAMES")
        print("="*60)
        
        # prepare all calibration frames
        for name, calib_set in self.calibration_sets.items():
            print(f"\n{name}:")
            calib_set.prepare()
 
class Cepheid_Data_Organiser:
    """
    Organises Cepheid files by number and night.
    """
    def __init__(self, cepheids_directory):
        """
        Create path to Cepheid directory and find patterns in files of "Cepheid_(#)".
        """
        self.cepheids_directory = Path(cepheids_directory)
        self.cepheid_pattern = re.compile(r'Cepheids?_(\d+)', re.IGNORECASE) 
    
    def list_observation_nights(self):
        """
        Sort all directories for nights in the Cepheids directory
        in alphabetical order
        """
        nights = sorted([night for night in self.cepheids_directory.iterdir() if night.is_dir()])
        return nights

    def organise_night(self, night_directory):
        """
        Organise each night's files based on Cepheid number
        """
        cepheid_files = defaultdict(list)
        
        for file in sorted(Path(night_directory).glob("*.fits")):
            pattern_presence = self.cepheid_pattern.search(file.name)
            if pattern_presence:
                ceph_num = int(pattern_presence.group(1))
                cepheid_files[ceph_num].append(file)
        
        return dict(cepheid_files)
    
    def organise_all_nights(self):
        """
        Organise all Cepheid files firstly by night, and then by Cepheid number. Returns
        a list of files that correspond to each night+Cepheid.
        """
        all_data = {}
        nights = self.list_observation_nights()
        
        for night in nights:
            night_name = night.name
            cepheid_files = self.organise_night(night)
            if cepheid_files:
                all_data[night_name] = cepheid_files
        
        return all_data
    
    @staticmethod
    def filter_useful_images(file_list):
        """
        Find the burst of images by grouping images by exposure time.
        """
        # extract exposure times
        exp_times = []
        for fits_file in file_list:
            with fits.open(fits_file) as hdul:
                exp_time = hdul[0].header.get('EXPTIME', None)
                exp_times.append((fits_file, exp_time))
        
        # group images by exposure time
        exp_groups = defaultdict(list)
        for fits_file, exp_time in exp_times:
            if exp_time is not None:
                exp_time_rounded = round(exp_time, 2)
                exp_groups[exp_time_rounded].append(fits_file)
        
        # now find the largest group (this is the burst)
        largest_group = max(exp_groups.values(), key=len, default=[])
        print(f"{len(largest_group)} images in largest group.")

        return largest_group

            
    
class CepheidImageReducer:

    """Perform bias and flat-field corrections on raw Cepheid images"""

    def __init__(self, master_bias, master_flat_field, border=50):

        """Load in calibration frames and border to trim images by."""
        self.mstr_bias = master_bias
        self.mstr_ff = master_flat_field
        self.border = border

    def perform_reduction(self, filename, trim = True):

        """Perform data reduction on a single image"""

        with fits.open(filename) as hdul:
            data = hdul[0].data

        if trim is True:
            data = data[self.border:-self.border, self.border, -self.border]

        bias_corrected_frame = data - self.mstr_bias
        flat_fielded_frame = bias_corrected_frame / self.mstr_ff
        science_frame = flat_fielded_frame

        return science_frame
    
class CepheidImageStacker:

    """Stacks all science frames together to produce a single final image for each 
    night + Cepheid."""

    def stack_images(image_list, headers_list, method='mean'):

        """Stacks images and combines headers"""

        image_stack = np.stack(image_list, axis=0)
        if method == 'mean':
            final_image = np.mean(image_stack, axis=0) 
        else:
            final_image = np.median(image_stack, axis=0)

        base_header = headers_list[0].copy()
        base_header['TOTEXP'] = sum(h.get('EXPOSURE', 0) for h in headers_list)
        base_header['MEANEXP'] = base_header['TOTEXP'] / len(headers_list)

        gains = [h.get('GAIN', 0) for h in headers_list if 'GAIN' in h]
        if gains:
            base_header['MEANGAIN'] = float(np.mean(gains))

        read_noises = [h.get('RDNOISE', 0) for h in headers_list if 'RDNOISE' in h]
        if read_noises:
            base_header['TOTRN'] = float(np.sqrt(np.sum(np.array(read_noises)**2)))

        base_header['NSTACK'] = len(image_list)
        base_header['COMMENT'] = f'Stacked {len(image_list)} images using {method}'

        return final_image, base_header




        
            







