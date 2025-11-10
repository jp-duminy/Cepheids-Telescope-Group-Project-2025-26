"""Unfinished as of 10/11/25"""

import numpy as np
from photutils.aperture import CircularAperture as circ_ap, CircularAnnulus as circ_ann, \
    aperture_photometry as ap, ApertureMask as mask, SkyAperture as sky_ap, \
    RectangularAperture as rect_ap
from photutils.centroids import centroid_2dg
import photutils.psf as psf
import astropy.io.fits as fits
import astropy.wcs as wcs
import matplotlib.pyplot as plt


class Aperture_Photometry: 

    """A class to perform basic aperture photometry on a dataset, up to and
    including determining the instrumental magnitude of a target."""

    """AS OF 10/11/25, NO MECHANISM TO NORMALISE FLUXES TO CTS PER SECOND"""

    def __init__(self, filename): 

        """Initialise with dataset in form of 2D Numpy array derived from
        .fits file."""

        #Load data and header from FITS file as an HDU list 
        #Remember HDU list may have several indices
        #Perhaps ask Kenneth about HDU index

        with fits.open(filename) as hdul:
            data = hdul[0]._data
            header = hdul[0]._data

        if data.ndim != 2:
            raise ValueError(f"The image is {data.ndim}D, not 2D")
        
        data = data.astype(float)
        data = np.nan_to_num(data)

        self.data = data
        self.header = header

    def get_pixel_coords(self, WCS, RA, Dec, origin = 0):
        """Converts world coordinates of targets (RA, Dec) to pixel coordinates (x, y). 
        Requires argument of a WCS object and RA & Dec as seperate arrays
        (see wcs.WCS documentation)."""
        if isinstance(RA, float) == False or isinstance(Dec, float) == False:
            raise TypeError(f"These coordinates must be floats. If in arrays, try \
                            doing them one at a time.")
        
        x, y = WCS.wcs_world2pix(RA, Dec, origin)
        #origin is 0 for numpy array, 1 for FITS file
        return x, y
        #NB: FITS file might by upside down by the time this is used, could cause issues. 
    
    def mask_data_and_plot(self, x, y, width, plot = False):
        """Set boolean mask to cut out a square shape around the target, to remove
        other sources. Plot masked data as heatmap if plot == True, don't otherwise"""
        aperture = rect_ap((x,y), width, width)
        mask = aperture.to_mask(method = "center")
        masked_data = mask.multiply(self.data)
    
        if plot == True:
            plt.imshow(masked_data)
            plt.show()
        
        return masked_data


    def get_centroid_and_fwhm(self, data):
        """Get Gaussian centroid of target source around which to
        centre the aperture, and the FWHM of the target source centred around
        the centroid."""
        centroid = centroid_2dg(data) #Should be masked
        fwhm = psf.fit_fwhm(data = data, xypos = centroid)
        #Function expects data to be bkgd subtracted
        #Nan/inf values automatically masked
        return centroid, fwhm
    
    def aperture_photometry(self, data, centroid, ap_rad, inner, outer):
        """Main method: Using the determined centroids and FWHM of the source, Sum the fluxes
        through the circular apertures and annuli."""

        if inner < 1 or outer < 1:
            raise ValueError(f"inner and outer constants must both be > 1")

        target_aperture = circ_ap(centroid, ap_rad) 
        sky_annulus = circ_ann(centroid, r_in = inner*ap_rad, r_out = outer*ap_rad)
        #inner/outer are multiplicative constants to scale the size of the aperture.
        #Inner should be ~1.5, outer ~2.
        #Might ask Kenneth what the ideal constants are

        #Sum flux through apertures
        total_flux = ap(data, target_aperture)["aperture_sum"].value
        annulus_flux = ap(data, sky_annulus)["aperture_sum"].value
        #Get sky background
        mean_sky_bckgnd_per_pixel = annulus_flux / sky_annulus.area
        total_sky_bckgnd = mean_sky_bckgnd_per_pixel * target_aperture.area

        target_flux = total_flux - total_sky_bckgnd

        return target_flux #Sky subtracted
    
    def curve_of_growth(self, data):
        """To calculate and plot the sky-subtracted flux obtained in a series
        of increasingly large apertures."""

        aperture_radius = np.zeros(16)
        sky_sub_ap_flux = np.zeros(16)
        inner = 1.4
        outer = 2.0
        centroid, fwhm = self.get_centroid_and_fwhm(data)

        for index, factor in enumerate(np.linspace(0, 4, 16)):
            aperture_radius[index] = factor*fwhm
            flux = self.aperture_photometry(data, centroid, ap_rad = factor*fwhm, inner=inner, outer=outer)
            sky_sub_ap_flux[index] = flux

        plt.plot(aperture_radius, sky_sub_ap_flux, color = "red", linestyle = "-", marker = "o")
        plt.xlabel("Radius of aperture (pixels)")
        plt.ylabel("Sky subtracted flux through aperture (arb units)")
        plt.show()