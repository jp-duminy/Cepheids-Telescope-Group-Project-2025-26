import numpy as np
from astropy.time import Time

class Astro_Functions:

    def magnitude_error(snr: float):
        """
        Compute error on magnitudes.
        """
        return (2.5 / np.log(10)) * (1 / snr)

    def apparent_to_absolute(magnitude: float, distance: float):
        """
        Convert apparent magnitudes to absolute magnitudes.
        """
        return magnitude - (5 * np.log10(distance / 10))
    
    def modified_julian_date_converter(time):
        """
        Converts ISO times to MJD for ease of use.
        """
        time_array = np.array(time)
        t = Time(time_array, scale='utc', format='isot')
        return t.mjd # astronomers use modified julian date