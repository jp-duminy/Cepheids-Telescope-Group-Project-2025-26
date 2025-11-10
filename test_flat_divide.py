import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits

import astropy.units as u
import astropy.wcs as WCS
from astropy.coordinates import SkyCoord

from astropy.visualization import ZScaleInterval

from trim import Trim
from test_dark_subtract import Bias

class Flat:

    def __init__(self, flat1, flat2, flat3, flat4, science_img):
        """
        Store the flat images and the science image
        """
        self.flat1 = flat1
        self.flat2 = flat2
        self.flat3 = flat3
        self.flat4 = flat4

        self.science_img = science_img

    def trim(self):
        """
        Need to trim the flat field image, the science image has already been trimmed during the bias subtraction
        """
        trim_flat1 = Trim(self.flat1).cutting()
        trim_flat2 = Trim(self.flat2).cutting()
        trim_flat3 = Trim(self.flat3).cutting()
        trim_flat4 = Trim(self.flat4).cutting()

        self.trimmed_flats = [trim_flat1, trim_flat2, trim_flat3, trim_flat4]

        return self.trimmed_flats

    def stack_flat(self):
        """
        Stacking the flat field images
        """
        trimmed = self.trim()
        stack = np.stack(trimmed, axis=0)
        #taking the median of the flats to create the master flat
        master_flat = np.median(stack, axis=0)
        return master_flat
    
    def normalise(self):
        """
        Normalise the stacked flat field images!
        """
        master_flat = self.stack_flat()

        norm_factor = np.median(master_flat)
        flat_norm = master_flat / norm_factor
        return flat_norm
    
    def flat_divide(self):
        """
        Dividing by the flat field image, returns the flat-fielded image
        """
        norm_flat = self.normalise()
        #b = Bias()
        #corrected_img = b.subtraction(self.science_img)/ norm_flat
        corrected_img = self.science_img / norm_flat
        return corrected_img
    