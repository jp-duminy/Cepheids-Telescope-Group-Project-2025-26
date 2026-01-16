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
    
    def img_stacker(self, imgs):
        """
        Stack images and return stack + median of stack.
        """
        stack = np.stack(imgs, axis=0)
        median = np.median(stack, axis=0)
        return stack, median
    
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
        self._master_bias = np.median(bias_stack, axis=0)
        print(f"Master bias shape: {self.master_bias.shape}")

        return self.master_bias

    def create_master_flat(self):
        """
        Create master flat image from the stored flat file list.
        """
        if self.master_flat is not None:  # if this already exists
            return self.master_flat
        
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
        self._master_flat = flat_stack / normalisation
        
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
    
        self.calibration_sets[calib_name] = calib_set
    
        return calib_set
    
    def map_night_to_week(self, night_name, calib_name):
        """
        Map an observation night to its corresponding calibration set.
        Night name will be a file such as "2025_09_22" (first night)
        Calibration name will be the week that night occured such as "Week1""
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


    

