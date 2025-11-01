"""
Cepheids TGP 25/26 Error Analysis Code
@author: jp
"""

import numpy as np
import scipy
import emcee as mc
from matplotlib import pyplot as plt

class Cepheid_Chi_Error_Analysis:
    
    def __init__(self, name, period, magnitude, snr, distance):
        """
        Store Cepheid properties.
        """
        self.name = name
        self.period = np.array(period)
        self.magnitude = np.array(magnitude)
        self.snr = np.array(snr)
        self.distance = np.array(distance) # pc
        
    def apparent_to_absolute(self):
        """
        Convert apparent magnitudes to absolute magnitudes.
        """
        return self.magnitude - (5 * np.log10(self.distance / 10))
        
    def magnitude_error(self):
        """
        Compute error on magnitudes.
        """
        return (2.5 / np.log(10)) * (1 / self.snr)
    
    def model(self, a, b):
        """
        Model period-luminosity function for use with scipy.optimize.
        a: Gradient
        b: Intercept
        """
        return a*(np.log10(self.period)-1) + b # luminosity-period relationship
    
    def chi_squared_model(self, params):
        """
        Chi-squared function to be minimised with scipy.optimize.
        Parameters retain values defined in model().
        """
        a, b = params
        M_err = self.magnitude_error() # assume distance has negligible uncertainty
        M_obs = self.apparent_to_absolute()
        M_model = self.model(a, b)
        
        chisq = np.sum(((M_obs - M_model) / M_err)**2)
        
        return chisq
    
    def minimise_chi_square(self):
        """
        Minimises chi-square & generates contour plots.
        """
        initial_guess = [2.43, 4.05] # from Benedict et al. (2007)
        
        result = scipy.optimize.minimize(self.chi_squared_model, initial_guess)
        a, b = result.x # minimised parameter values
        minimised_chi2 = result.fun # chi-squared value for parameters
        
        return a, b, minimised_chi2
    
    def chi_square_contour_plot(self):
        """
        Generate contour plots of chi-square fit; displays minimised values.
        """
        a, b, minimised_chi2 = self.minimise_chi_square()
        
        a_vals = np.linspace(a-0.5, a+0.5, 100)
        b_vals = np.linspace(b-0.5, b+0.5, 100)
        
        chi2_grid = np.array([[self.chi_squared_model([a, b]) for b in b_vals] for a in a_vals])
        
        plt.contour(a_vals, b_vals, chi2_grid)
        plt.title('Chi-squares contour plot for modelled magnitudes.')
        plt.xlabel('a')
        plt.ylabel('b')
