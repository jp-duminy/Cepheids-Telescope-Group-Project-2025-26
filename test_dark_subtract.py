import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits

import astropy.units as u
import astropy.wcs as WCS
from astropy.coordinates import SkyCoord

from astropy.visualization import ZScaleInterval

from trim import Trim

class Bias:

    def __init__(self, bias1, bias2, bias3, bias4, science_img):
        """
        Store the bias frames and the science image
        """
        self.bias1 = bias1
        self.bias2 = bias2
        self.bias3 = bias3
        self.bias4 = bias4

        self.science_img = science_img

    def trim(self):
        """
        Trim the bias frames and the science image
        """
        trim_bias1 = Trim(self.bias1).cutting()
        trim_bias2 = Trim(self.bias2).cutting()
        trim_bias3 = Trim(self.bias3).cutting()
        trim_bias4 = Trim(self.bias4).cutting()
        trim_science_img = Trim(self.science_img).cutting()

        self.trimmed_biases = [trim_bias1, trim_bias2, trim_bias3, trim_bias4]
        self.trimmed_science = trim_science_img

        return self.trimmed_biases, self.trimmed_science

    def stack(self):
        """
        Stack the bias frames and take median to make a master bias
        """
        trimmed, _ = self.trim()
        stack = np.stack(trimmed, axis=0)
        master_bias = np.median(stack, axis=0)
        return master_bias
    
    def subtraction(self):
        """
        Subtract master bias, return the bias subtracted image
        """
        master_bias = self.stack()
        corrected_img = self.trimmed_science - master_bias
        return corrected_img
