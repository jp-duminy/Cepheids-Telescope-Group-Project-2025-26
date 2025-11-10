import numpy as np
import scipy
import astropy

class Astro_Functions:

    def magnitude_error(snr: int):
        """
        Compute error on magnitudes.
        """
        return (2.5 / np.log(10)) * (1 / snr)

    def apparent_to_absolute(magnitude: int, distance: int):
        """
        Convert apparent magnitudes to absolute magnitudes.
        """
        return magnitude - (5 * np.log10(distance / 10))
    
    def modified_julian_date_converter(time):
        """
        Converts ISO times to MJD for ease of use.
        """
        t = astropy.time.Time(time, scale='utc', format='iso')
        return t.mjd # astronomers use modified julian date
    
        