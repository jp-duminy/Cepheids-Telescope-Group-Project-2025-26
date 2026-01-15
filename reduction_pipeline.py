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
        Create master bias image in a specified directory.
        """
        bias_tag = "*.fits"

        if self.master_bias is not None: # if this already exists
            return self.master_bias
        
        bias_files = sorted(self.bias_dir.glob(bias_tag))
        if len(bias_files) == 0:
            raise FileNotFoundError(f"No bias files in {self.bias_dir}") # diagnostic
        print(f"{len(bias_files)} found in d{self.bias_dir}.")

        bias_frames = [self.img_trim(fits.getdata(f)) for f in bias_files]
        _, self.master_bias = self.img_stacker(bias_frames)
        print(f"Master bias shape: {self.master_bias.shape}")

        return self.master_bias

    def create_master_flat(self):
        """
        Create master bias image in a specified directory.
        """
        flat_tag = "*.fits"

        if self.master_flat is not None: # if this already exists
            return self.master_flat
        
        flat_files = sorted(self.bias_dir.glob(flat_tag))
        if len(flat_files) == 0:
            raise FileNotFoundError(f"No flat files in {self.flat_dir}") # diagnostic
        print(f"{len(flat_files)} found in d{self.flat_dir}.")

        flat_frames = [self.img_trim(fits.getdata(f)) for f in flat_files]
        stack, normalisation = self.img_stacker(flat_frames)
        self.master_flat = stack / normalisation
        print(f"Master flat shape: {self.master_flat.shape}")

        return self.master_flat
        
    def prepare(self):
        """
        Prepare master bias and flat frames.
        """
        self.create_master_bias()
        self.create_master_flat()
        return self
    
