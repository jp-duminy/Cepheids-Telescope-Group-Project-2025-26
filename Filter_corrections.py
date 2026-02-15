import numpy as np
class AndromedaFilterCorrection:

    """Photometric data from Liverpool comes in the ugriz filter
    system. This method will convert data from the ugriz system to
    the UBVRI system of PIRATE.
    
    args: u, g
    __call__: Returns B-V colour with uncertainty."""

    def __init__(self, u = None, g = None):
        """Initialise with magnitudes from ugriz filters"""
        self.u = u
        self.g = g

    def mags(self, filter):
        """Convert u and g photometric data to B and V data"""
        B = self.u - 0.8116*(self.u - self.g) + 0.1313
        B_uncertainty = 0.0095
        V = self.g - 0.2906*(self.u - self.g) + 0.0885
        V_uncertainty = 0.0129
        if filter == "B":
            return B, B_uncertainty
        return V, V_uncertainty

    def colours(self): 
        """To convert the u and g photometric data to its B-V colour
        using Lupton (2005) calculations."""
        B = self.u - 0.8116*(self.u - self.g) + 0.1313
        B_uncertainty = 0.0095
        V = self.g - 0.2906*(self.u - self.g) + 0.0885
        V_uncertainty = 0.0129
        colour = B-V
        colour_uncert = np.sqrt(B_uncertainty**2 + V_uncertainty**2)
        return colour, colour_uncert

    

def filter_correction(g, method = "Jordi"):

    """Convert g-band magnitude to V-band magnitude using one of two methods. The first method uses
    a formula combined from two equations in Jordi et al. (2006), returning the V-band magnitude with
    an error. The second method uses a formula combined from equations in Jester et al. (2005), and
    Bilir et al. *(2005), returning the V-band magnitude without an error. 

    Inputs: g-band magnitude, method of choice (either "Jordi" or "Jester-Bilir")
    Output: V-band magnitude, uncertainty (if method is "Jordi")
    
    For more information, visit https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php"""

    if method == "Jordi":
        V = g - 0.3485 
        V_err = 0.4506

    elif method == "Jester-Bilir":
        V = g + 0.3298
        V_err = None

    return V, V_err
