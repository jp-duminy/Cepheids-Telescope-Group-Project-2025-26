import numpy as np
from photutils.aperture import CircularAperture as circ_ap, CircularAnnulus as circ_ann, \
    aperture_photometry as ap, RectangularAperture as rect_ap
from photutils.centroids import centroid_2dg
import photutils.psf as psf
import astropy.io.fits as fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class AperturePhotometry: 

    """A class to perform basic aperture photometry on a dataset, up to and
    including determining the instrumental magnitude of a target."""

    def __init__(self, filename): 

        """Initialise with dataset in form of 2D Numpy array derived from
        .fits file."""

        #Load data and header from FITS file as an HDU list 
        #Remember HDU list may have several indices

        with fits.open(filename) as hdul:
            data = hdul[0].data
            header = hdul[0].header 

        if data.ndim != 2:
            raise ValueError(f"The image is {data.ndim}D, not 2D")
        
        data = data.astype(float)
        data = np.nan_to_num(data)

        self.header = header
        #Normalise to cts per second
        self.data = data / float(header["EXPTIME"])

        self.wcs = WCS(header)

    def get_pixel_coords(self, RA, Dec):
        """Converts world coordinates of targets (RA, Dec) to pixel coordinates (x, y). 
        Uses WCS created from astropy.wcs.WCS in conjunction with fits header keywords."""
        
        x, y = self.wcs.world_to_pixel_values(RA, Dec)

        return x, y
        #NB: FITS file might by upside down by the time this is used, could cause issues. 
    
    def mask_data_and_plot(self, x, y, width, plot = False):
        """Set boolean mask to cut out a square shape around the target, to remove
        other sources. Plot masked data as heatmap if plot == True, don't otherwise"""
        aperture = rect_ap((x,y), width, width)
        mask = aperture.to_mask(method = "center")
        masked_data = mask.cutout(self.data)
    
        if plot == True:
            plt.imshow(masked_data, norm=LogNorm())
            plt.show()
        
        return masked_data

    def get_centroid_and_fwhm(self, data, plot = False):
        """Get Gaussian centroid of target source around which to
        centre the aperture, and the FWHM of the target source centred around
        the centroid."""
        #Crude bkgd subtraction
        crude_sub_data = data - np.mean(data)
        centroid = centroid_2dg(crude_sub_data) #Should be masked
        fwhm = psf.fit_fwhm(data = crude_sub_data, xypos = centroid).item()
        #Function expects data to be bkgd subtracted
        #Nan/inf values automatically masked
        if plot == True:
            plt.imshow(data, norm = LogNorm())
            plt.plot(centroid[0], centroid[1], marker = "+", color = "r")
            plt.show()

        return centroid, fwhm
    
    def aperture__photometry(self, data, centroid, ap_rad, ceph_name = None,
                            date = None, inner=1.5, outer=2, plot = True, savefig = False):
        """Main method: Using the determined centroids and FWHM of the source, Sum the fluxes
        through the circular apertures and annuli."""

        if inner < 1 or outer < 1:
            raise ValueError(f"inner and outer constants must both be > 1")

        target_aperture = circ_ap(centroid, ap_rad) 
        sky_annulus = circ_ann(centroid, r_in = inner*ap_rad, r_out = outer*ap_rad)
        #inner/outer are multiplicative constants to scale the size of the aperture.
        #Inner should be ~1.5, outer ~2.

        #Plot apertures
        if plot == True:
            fig, ax = plt.subplots()
            ax.imshow(data, origin='lower', interpolation='nearest', cmap='viridis' , norm=LogNorm) # Display the image
            target_aperture.plot(ax=ax, color='red')
            sky_annulus.plot(ax=ax, color = "white")
            plt.title(f"{ceph_name} taken on {date}")
            if savefig == True:
                plt.savefig(f"{ceph_name}_{date}")
            plt.show()

        #Sum flux through apertures
        total_flux = ap(data, target_aperture)["aperture_sum"].value
        annulus_flux = ap(data, sky_annulus)["aperture_sum"].value
        #Get sky background
        mean_sky_bckgnd_per_pixel = annulus_flux / sky_annulus.area
        total_sky_bckgnd = mean_sky_bckgnd_per_pixel * target_aperture.area

        target_flux = total_flux - total_sky_bckgnd 

        return target_flux, target_aperture.area, mean_sky_bckgnd_per_pixel, sky_annulus.area
    
    def curve_of_growth(self, data, ceph_name, date, savefig = False):
        """To calculate and plot the sky-subtracted flux obtained in a series
        of increasingly large apertures."""

        aperture_radius = np.zeros(16)
        sky_sub_ap_flux = np.zeros(16)
        inner = 1.4
        outer = 2.0
        centroid, fwhm = self.get_centroid_and_fwhm(data)

        for index, factor in enumerate(np.linspace(0.1, 4, 16)):
            aperture_radius[index] = factor*fwhm
            flux = self.aperture__photometry(data, centroid, ap_rad = factor*fwhm, inner=inner, outer=outer, plot = False)[0]
            sky_sub_ap_flux[index] = flux

        normalised_ssaf = sky_sub_ap_flux / np.max(sky_sub_ap_flux)

        plt.plot(aperture_radius, normalised_ssaf, color = "red", linestyle = "-", marker = "o")
        plt.xlabel("Radius of aperture (pixels)")
        plt.ylabel("Sky subtracted flux through aperture (arb units)")
        plt.title("Normalised curve of growth")
        if savefig == True:
            plt.savefig(f"CoG_{ceph_name}_{date}")
        plt.show()

    def instrumental_magnitude(self, counts):
        """Compute instrumental magnitude of targets from sky-subtracted flux"""
        inst_mag = -2.5 * np.log10(counts).item()
        return inst_mag 
    
    def get_inst_mag_error(self, target_counts, aperture_area, sky_counts, sky_ann_area, 
                gain = None, exp_time = None, read_noise = None, stack_size = None):

        """Compute uncertainty of instrumental magnitude using a revised CCD equation 
        for the SNR taken from Handbook of CCD Astronomy by Howell.

        NB: Assumes that sky background uncertainty follows Poisson stats,
        and that the dark current is negligable.
        Remember that electrons are the subject of the CCD eqn, not the 
        digital numbers. Include the gain factor
        sky_counts units of e- per s per pixel"""

        signal = gain * target_counts * exp_time
        
        source_variance = signal
        sky_variance = gain * exp_time * aperture_area * sky_counts * (1 + (aperture_area / sky_ann_area))
        rn_term = aperture_area  * (1 + (aperture_area / sky_ann_area))* (read_noise ** 2) * stack_size

        noise = np.sqrt(source_variance + sky_variance + rn_term)

        snr = signal / noise
        snr = snr.item()
        error = 1.086 / snr
        return error
    

    

    


        

        
