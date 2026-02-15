import numpy as np
from photutils.aperture import CircularAperture as circ_ap, CircularAnnulus as circ_ann, \
    aperture_photometry as ap, RectangularAperture as rect_ap
from photutils.centroids import centroid_2dg
import photutils.psf as psf
import astropy.io.fits as fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import csv
import statsmodels.api as sm
from matplotlib.lines import Line2D
import AirmassInfo
from astropy.stats import sigma_clipped_stats

from astropy.visualization import PowerStretch, ImageNormalize, ZScaleInterval

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

        #self.wcs = WCS(header) (defunct)

    def get_pixel_coords(self, RA, Dec):
        """Converts world coordinates of targets (RA, Dec) to pixel coordinates (x, y). 
        Uses WCS created from astropy.wcs.WCS in conjunction with fits header keywords."""
        
        x, y = self.wcs.world_to_pixel_values(RA, Dec)

        return x, y
        #NB: FITS file might by upside down by the time this is used, could cause issues. 
    
    def mask_data_and_plot(self, x, y, width, name, date, plot = False):
        """
        Set boolean mask to cut out a square shape around the target, to remove
        other sources. Plot masked data as heatmap if plot == True, don't otherwise
        """
        # set rectangular aperture
        aperture = rect_ap((x,y), width, width)
        # mask image
        mask = aperture.to_mask(method = "center")
        # cut out a box of that image
        masked_data = mask.cutout(self.data)
        # extract the offsets
        x_offset, y_offset = mask.bbox.ixmin, mask.bbox.iymin
    
        if plot == True:
            fig, ax = plt.subplots()
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(masked_data)
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=PowerStretch(a=2))
            ax.imshow(masked_data, origin='lower', interpolation='nearest',
                    cmap='viridis', norm=norm)
            plt.title(f"{width}pix cutout of {name} on {date}")
            plt.show()
        
        return masked_data, x_offset, y_offset

    def get_centroid_and_fwhm(self, data, name, plot = False):
        """
        Get Gaussian centroid of target source around which to
        centre the aperture, and the FWHM of the target source centred around
        the centroid. data should be masked_data
        """

        subtracted = data - np.mean(data) # this does not need to be precise
        _, median_bg, _= sigma_clipped_stats(data, sigma=6.0)
        subtracted = data - median_bg
        centroid = centroid_2dg(subtracted) #Should be masked
        fwhm = psf.fit_fwhm(data = subtracted, xypos = centroid).item()
        #Function expects data to be bkgd subtracted
        #Nan/inf values automatically masked
        if plot == True:
            fig, ax = plt.subplots()
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(subtracted)
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=PowerStretch(a=2))
            ax.imshow(subtracted, origin='lower', interpolation='nearest',
                    cmap='viridis', vmin=0, vmax=np.percentile(subtracted[subtracted > 0], 99))
            plt.plot(centroid[0], centroid[1], marker = "+", color = "r")
            plt.title(f"Centroid for {name}")
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

        _, median_bg, _= sigma_clipped_stats(data, sigma=4.0)
        subtracted = data - median_bg

        # plot apertures
        if plot == True:
            fig, ax = plt.subplots()
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(subtracted)
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=PowerStretch(a=2))
            ax.imshow(subtracted, origin='lower', interpolation='nearest',
                    cmap='viridis', norm=norm)
            target_aperture.plot(ax=ax, color='red')
            sky_annulus.plot(ax=ax, color = "white")
            plt.title(f"Sky-subtracted {ceph_name} taken on {date}")
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
        NOW DEFUNCT!!!
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

    #NEEDS TO BE EDITED. AirmassInfo returns a single value of airmass whereas it needs to be an array of airmass values for each standard star.
    def __init__(self, airmass, Vmag, m_inst, m_err):
        """Initialise filename, airmass, and other values"""
        self.airmass = np.asarray(airmass)
        self.Vmag = np.asarray(Vmag)
        self.m_inst = np.asarray(m_inst)
        self.m_err = np.asarray(m_err)

        
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

        # Dependent variable
        y = self.Vmag - self.m_inst
        X = self.airmass

        # Weights
        w = 1 / self.m_err**2
        #use statsmodels for weighted least squares to get uncertainties (for more details see https://www.geeksforgeeks.org/machine-learning/weighted-least-squares-regression-in-python/)
        X_sm = sm.add_constant(X)
        print(type(X_sm), X_sm)
        print(type(w), w)
        wls_model = sm.WLS(y, X_sm, weights=w)
        results = wls_model.fit()
        Z, k = results.params
        cov_zk = results.cov_params()

        Z_airmass1 = Z + k * 1.0

        # Transform covariance from (Z, k) to (Z1, k)
        transform = np.array([[1.0, 1.0], [0.0, 1.0]])
        cov_z1k = transform @ cov_zk @ transform.T
        self.param_cov = cov_z1k

        k_err = np.sqrt(cov_z1k[1, 1])
        Z_airmass1_err = np.sqrt(cov_z1k[0, 0])
        return k, Z_airmass1, k_err, Z_airmass1_err
        
    def plot_atmospheric_extinction(self, k, Z1, k_err, Z1_err):
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

        y = self.Vmag - self.m_inst

        plt.errorbar(self.airmass, y, yerr=self.m_err, fmt='o', label='Data')
        airmass_min = min(self.airmass)
        airmass_max = max(self.airmass)
        airmass_range = airmass_max - airmass_min
        x_fit = np.linspace(airmass_min - 0.1*airmass_range, airmass_max + 0.1*airmass_range, 100)
        y_fit = Z1 + k * (x_fit - 1.0)
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: k={k:.3f}±{k_err:.3f}, Z(airmass=1)={Z1:.2f}')
        #Add shaded region for 1σ uncertainty
        cov = getattr(self, "param_cov", None)
        if cov is not None:
            dx = x_fit - 1.0
            var_y = cov[0, 0] + 2.0 * cov[0, 1] * dx + cov[1, 1] * dx ** 2
            sigma_y = np.sqrt(np.maximum(var_y, 0.0))
            y_fit_upper = y_fit + sigma_y
            y_fit_lower = y_fit - sigma_y
        else:
            y_fit_upper = (Z1 + Z1_err) + (k + k_err) * (x_fit - 1.0)
            y_fit_lower = (Z1 - Z1_err) + (k - k_err) * (x_fit - 1.0)
        plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='r', alpha=0.2)
        plt.xlabel('Airmass')
        plt.ylabel('V - m_inst')
        plt.title('Atmospheric Extinction Fit')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_parameter_space(self, k_best, Z1_best, k_err, Z1_err):
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

        cov = getattr(self, "param_cov", None)
        if cov is not None:
            k_sigma = np.sqrt(cov[1, 1])
            Z1_sigma = np.sqrt(cov[0, 0])
        else:
            k_sigma = k_err
            Z1_sigma = Z1_err

        k_values = np.linspace(k_best - 5 * k_sigma, k_best + 5 * k_sigma, 100)
        Z1_values = np.linspace(Z1_best - 5 * Z1_sigma, Z1_best + 5 * Z1_sigma, 100)
        K, Z1 = np.meshgrid(k_values, Z1_values)

        if cov is not None:
            inv_cov = np.linalg.inv(cov)
            dZ1 = Z1 - Z1_best
            dk = K - k_best
            chi2 = (
                inv_cov[0, 0] * dZ1 ** 2
                + inv_cov[1, 1] * dk ** 2
                + 2 * inv_cov[0, 1] * dZ1 * dk
            )
        else:
            chi2 = ((k_best - K) / k_err) ** 2 + ((Z1_best - Z1) / Z1_err) ** 2

        #create contour plot of background (background contour fill completely), with coloured rings for 1σ, 2σ, 3σ confidence intervals
        plt.contourf(K, Z1, chi2, levels=50, cmap='viridis', alpha=0.5)
        plt.contour(K, Z1, chi2, levels=[2.28, 6.17, 11.8], colors=['white'], linestyles=['-', '--', ':'], linewidths=2)
        plt.plot(k_best, Z1_best, 'bo', label='Best Fit')
        plt.xlabel('Extinction Coefficient (k)')
        plt.ylabel('Zero-point at Airmass=1 (Z)')
        plt.title('Parameter Space for Extinction Fit')
        plt.legend()
        #add legend for confidence intervals and best fit
        custom_lines = [
            Line2D([0], [0], color='white', lw=2),
            Line2D([0], [0], color='white', lw=2, linestyle='--'),
            Line2D([0], [0], color='white', lw=2, linestyle=':'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=6)
        ]
        plt.legend(custom_lines, ['1σ (68.3%)', '2σ (95.4%)', '3σ (99.7%)', 'Best Fit'], loc='upper right')
        plt.grid()
        plt.show()


        
        '''plt.contourf(K, Z1, chi2, levels=[0, 2.28, 6.17, 11.8], colors=['lightblue', 'blue', 'darkblue'], alpha=0.5) 
        plt.plot(k_best, Z1_best, 'ro', label='Best Fit') 
        plt.xlabel('Extinction Coefficient (k)') 
        plt.ylabel('Zero-point at Airmass=1 (Z)') 
        plt.title('Parameter Space for Extinction Fit') 
        plt.legend() 
        plt.grid() 
        plt.show()'''
    
    def plot_residuals(self, k, Z1):
        """ Plot residuals of atmospheric extinction fit. 
        
        Parameters 
        ---------- 
        airmass : array-like Airmass values 
        Vmag: array-like 
        Catalog V magnitudes counts : array-like 
        Measured counts count_err : array-like 
        Uncertainty in counts exptime : array-like 
        Exposure times (seconds) 
        
        Returns 
        ------- 
        None 
        """
        y = self.Vmag - self.m_inst 
        y_fit = Z1 + k* (self.airmass - 1.0) 
        residuals = y - y_fit 
        plt.errorbar(self.airmass, residuals, yerr=self.m_err, fmt='o') 
        plt.axhline(0, color='r', linestyle='--') 
        plt.xlabel('Airmass')
        plt.ylabel('Residuals (V - m_inst - Fit)')
        plt.title('Residuals of Atmospheric Extinction Fit') 
        plt.grid() 
        plt.show()

    def remove_outliers(self, k, Z1, threshold=3): 
        """Remove outliers from the dataset based on residuals of the fit. 
        Parameters 
        ---------- 
        airmass : array-like Airmass values 
        Vmag : array-like 
        Catalog V magnitudes counts : array-like 
        Measured counts count_err : array-like 
        Uncertainty in counts exptime : array-like 
        Exposure times (seconds) k : float 
        Extinction coefficient from fit 
        Z1 : float Zero-point at airmass=1 from fit 
        threshold : float Number of standard deviations to use as cutoff for outliers (default: 3) 
        
        Returns 
        ------- 
        mask : array-like Boolean mask indicating which data points are not outliers 
        """ 
        y = self.Vmag - self.m_inst 
        y_fit = Z1 + k * (self.airmass - 1.0) 
        residuals = y - y_fit 
        std_residuals = np.std(residuals) 
        mask = np.abs(residuals) > threshold * std_residuals 
        return mask    