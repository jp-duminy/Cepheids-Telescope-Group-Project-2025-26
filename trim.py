import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits

import astropy.units as u
import astropy.wcs as WCS
from astropy.coordinates import SkyCoord

from astropy.visualization import ZScaleInterval

class Trim:

    def __init__(self, science_img):
        """
        Store the image that will be trimmed.
        NOTE: this says science image but this trim function is used for 
        the bias and flat-fielding frames as well as the science images.
        """
        self.science_img = science_img

    def cutting(self):
        """
        Cutting 50 pixels off of each side of the image
        This returns the trimmed image
        """
        #read in the data
        data = self.science_img
        trimmed = data[50:-50, 50:-50]
        return trimmed