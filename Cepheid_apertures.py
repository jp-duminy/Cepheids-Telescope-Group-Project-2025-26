import numpy as np
from photutils.aperture import CircularAperture as circ_ap, CircularAnnulus as circ_ann, \
    aperture_photometry as ap, RectangularAperture as rect_ap
from photutils.centroids import centroid_2dg
import photutils.psf as psf
import astropy.io.fits as fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import csv
import statsmodels.api as sm
from matplotlib.lines import Line2D
import AirmassInfo
from astropy.stats import sigma_clipped_stats

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
            plt.imshow(masked_data)
            plt.show()
        
        return masked_data

    def get_centroid_and_fwhm(self, data, plot = False):
        """
        Get Gaussian centroid of target source around which to
        centre the aperture, and the FWHM of the target source centred around
        the centroid. data should be masked_data
        """
        # sigma-clipped background subtraction
        _, median, _ = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
        subtracted = np.maximum(data - median, 0)

        centroid = centroid_2dg(subtracted) #Should be masked
        fwhm = psf.fit_fwhm(data = subtracted, xypos = centroid).item()
        #Function expects data to be bkgd subtracted
        #Nan/inf values automatically masked
        if plot == True:
            plt.imshow(data)
            plt.plot(centroid[0], centroid[1], marker = "+", color = "r")
            plt.show()

        return centroid, fwhm
    
    def aperture_photometry(self, data, centroid, ap_rad, ceph_name = None,
                            date = None, inner=1.5, outer=2, plot = True, savefig = False):
        """
        Main method: Using the determined centroids and FWHM of the source, Sum the fluxes
        through the circular apertures and annuli.
        """

        if inner < 1 or outer < 1:
            raise ValueError(f"inner and outer constants must both be > 1")

        target_aperture = circ_ap(centroid, ap_rad) 
        sky_annulus = circ_ann(centroid, r_in = inner*ap_rad, r_out = outer*ap_rad)
        #inner/outer are multiplicative constants to scale the size of the aperture.
        #Inner should be ~1.5, outer ~2.

        # plot apertures
        if plot == True:
            fig, ax = plt.subplots()
            ax.imshow(data, origin='lower', interpolation='nearest', cmap='viridis') # Display the image
            target_aperture.plot(ax=ax, color='red')
            sky_annulus.plot(ax=ax, color = "white")
            plt.title(f"{ceph_name} taken on {date}")
            if savefig == True:
                plt.savefig(f"{ceph_name}_{date}")
            plt.show()

        total_flux = ap(data, target_aperture)["aperture_sum"].value

        # sigma clipping to produce an even background subtraction
        annulus_mask = sky_annulus.to_mask(method='center')
        annulus_data = annulus_mask.multiply(data)
        annulus_data_1d = annulus_data[annulus_mask.data > 0]

        mean_sky, median_sky, std_sky = sigma_clipped_stats(
        annulus_data_1d, 
        sigma=3.0, 
        maxiters=5
        ) # keep other variables for completeness

        mean_sky_bckgnd_per_pixel = median_sky # after sigma clipping still take median in case outliers survive
        total_sky_bckgnd = mean_sky_bckgnd_per_pixel * target_aperture.area

        target_flux = total_flux - total_sky_bckgnd 

        return target_flux, target_aperture.area, mean_sky_bckgnd_per_pixel, sky_annulus.area
    
    def curve_of_growth(self, data, ceph_name, date, savefig = False):
        """
        To calculate and plot the sky-subtracted flux obtained in a series
        of increasingly large apertures.
        """

        aperture_radius = np.zeros(16)
        sky_sub_ap_flux = np.zeros(16)
        inner = 1.4
        outer = 2.0
        centroid, fwhm = self.get_centroid_and_fwhm(data)

        for index, factor in enumerate(np.linspace(0.1, 4, 16)):
            aperture_radius[index] = factor*fwhm
            flux = self.aperture_photometry(data, centroid, ap_rad = factor*fwhm, inner=inner, outer=outer, plot = False)[0]
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
    
class DustExtinction:

    """This class calculates the dust extinction magnitude """

    def __init__(self, filter, colour_excess):

        """Initialise with R_V and colour_excess E(B-V), band
        wavelengths, and filter of choice."""

        self.R_V = 3.1
        self.lambda_B = 0.43 #micrometres
        self.lambda_V = 0.55 
        self.colour_excess = colour_excess
        self.filter = filter
        if self.filter != "B" and self.filter != "V":
            raise ValueError(f"The band filter must be either B or V")

    def compute_extinction_mag(self):

        """Compute dust extinction magnitude in B & V filters"""

        #V-band extinction
        A_V = self.R_V * self.colour_excess

        if self.filter == "V":
            return A_V
        
        #B-band extinction
        x = 1 / self.lambda_B

        y = x - 1.82
  
        a = 1 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + \
        0.72085*(y**4) +  0.01979*(y**5) - 0.77530*(y**6) + \
        0.32999*(y**7)

        b =  1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - \
        5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - \
        2.09002*(y**7)

        A_B = A_V * (a + b / self.R_V)
        
        return A_B

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

class Airmass:

    """Extract airmass data from fits header"""

    def __init__(self, fits_path, Vmag, m_inst, m_err):
        """Initialise filename, airmass, and other values"""
        self.filename = fits_path
        self.block = AirmassInfo().process_fits(fits_path)
        '''block = (
                f"Object;        {obj}\n"
                f"RA (h:m:s);    {ra_str}\n"
                f"DEC (d:m:s);   {dec_str}\n"
                f"Date Obs;      {t.iso.split('.')[0]}\n"
                f"Altitude (°);  {alt:.2f}\n"
                f"Airmass;       {airmass:.2f}\n"
            )'''
        self.airmass = float(self.block.split("Airmass;")[1].strip())
        self.Vmag = Vmag
        self.m_inst = m_inst
        self.m_err = m_err

        
    def fit_extinction_weighted(self):
        """
        Weighted fit to determine atmospheric extinction coefficient (k)
        and photometric zero-point (Z).

        Fits:
            V - m_inst = Z + kX

        where:
            m_inst = -2.5 log10(counts / exptime)

        Parameters
        ----------
        airmass : array-like
            Airmass values
        Vmag : array-like
            Catalog V magnitudes
        counts : array-like
            Measured counts
        count_err : array-like
            Uncertainty in counts
        exptime : array-like
            Exposure times (seconds)

        Returns
        -------
        k : float
            Extinction coefficient (mag/airmass)
        Z : float
            Zero-point at airmass = 0
        Z_airmass1 : float
            Zero-point at airmass = 1
        k_err : float
            Uncertainty in k
        Z_err : float
            Uncertainty in Z
        """

        # Convert to numpy arrays
        airmass = np.asarray(self.airmass)
        Vmag = np.asarray(Vmag) #Magnitudes for standard stars

        # Dependent variable
        y = self.Vmag - self.m_inst
        X = airmass

        # Weights
        w = 1 / self.m_err**2
        #use statsmodels for weighted least squares to get uncertainties (for more details see https://www.geeksforgeeks.org/machine-learning/weighted-least-squares-regression-in-python/)
        X_sm = sm.add_constant(X)
        wls_model = sm.WLS(y, X_sm, weights=w)
        results = wls_model.fit()
        Z, k = results.params
        Z_err, k_err = results.bse
        #Comment

        Z_airmass1 = Z + k * 1.0
        Z_airmass1_err = np.sqrt(Z_err ** 2 + k_err ** 2)
        return k, Z_airmass1, k_err, Z_airmass1_err
        
    def plot_atmospheric_extinction(self):
        """
        Plot atmospheric extinction fit.

        Parameters
        ----------
        airmass : array-like
            Airmass values
        Vmag : array-like
            Catalog V magnitudes
        counts : array-like
            Measured counts
        count_err : array-like
            Uncertainty in counts
        exptime : array-like
            Exposure times (seconds)
        """
        import matplotlib.pyplot as plt

        k, Z, Z1, k_err, Z_err = self.fit_extinction_weighted()

        y = self.Vmag - self.m_inst

        plt.errorbar(self.airmass, y, yerr=self.m_err, fmt='o', label='Data')
        x_fit = np.linspace(min(self.airmass), max(self.airmass), 100)
        y_fit = Z + k * x_fit
        plt.plot(x_fit, y_fit, 'r-', label=f'Fit: k={k:.3f}±{k_err:.3f}, Z(airmass=1)={Z1:.2f}')
        plt.xlabel('Airmass')
        plt.ylabel('V - m_inst')
        plt.title('Atmospheric Extinction Fit')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_parameter_space(self):
        '''
        Plot parameter space for extinction coefficient (k) and zero-point (Z)
        as rings of standard deviations (1σ, 2σ, 3σ).
        Parameters
        ----------
        airmass : array-like
            Airmass values
        Vmag : array-like
            Catalog V magnitudes
        counts : array-like
            Measured counts
        count_err : array-like
            Uncertainty in counts
        exptime : array-like
            Exposure times (seconds)
        
        Returns
        -------
        None
        '''

        k_best, Z_best, Z1_best, k_err, Z_err = self.fit_extinction_weighted()

        k_vals = np.linspace(k_best - 3*k_err, k_best + 3*k_err, 100)
        Z_vals = np.linspace(Z_best - 3*Z_err, Z_best + 3*Z_err, 100)
        K, ZM = np.meshgrid(k_vals, Z_vals)

        chi2_map = np.zeros(K.shape)

        y = self.Vmag - self.m_inst

        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                model = ZM[i,j] + K[i,j] * self.airmass
                chi2_map[i,j] = np.sum(((y - model) / self.m_err) ** 2)

        chi2_min = np.min(chi2_map)
        delta_chi2 = chi2_map - chi2_min

        plt.contour(K, ZM, delta_chi2, levels=[2.30, 6.17, 11.8], colors=['blue', 'orange', 'red'])
        custom_lines = [Line2D([0], [0], color='blue', lw=2),
                        Line2D([0], [0], color='orange', lw=2),
                        Line2D([0], [0], color='red', lw=2)]
        plt.legend(custom_lines, ['1σ', '2σ', '3σ'])
        plt.plot(k_best, Z_best, '.', markersize=10, label='Best Fit')
        plt.xlabel('Extinction Coefficient k')
        plt.ylabel('Zero-point Z')
        plt.title('Parameter Space for k and Z')
        plt.grid()
        plt.show()