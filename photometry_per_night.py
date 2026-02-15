"""
Working photometry pipeline for TGP Cepheids 25-26.
author: @jp
"""

# default packages
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
import re
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import warnings
import json

# our classes
from Cepheid_apertures import AperturePhotometry
from Cepheid_apertures import Airmass
from Cepheid_apertures import DustExtinction
from AirmassInfo import AirmassInfo

# catalogues
from catalogues import ALL_CATALOGUES, get_catalogues_for_night, get_pixel_guess
from reference_star_catalogues import reference_catalogue
from orientation_catalogue import cepheid_orientation_catalogue

# updated DAOStarFinder
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture as circ_ap
from matplotlib.colors import LogNorm
from photutils.background import Background2D, MedianBackground

# stylistic plots
import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False

# directories
input_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/2025-10-06" # change to the name of the night
output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Photometry" # where we store the results
base_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids"

# because standards were somewhat inconveniently named
ALL_STANDARD_IDS = {"114176", "SA111775", "F_108", "SA112_595", "GD_246", "G93_48", "G156_31"}

#
# helper functions
#

# find directory (TGP directories are messy)
def input_dir_for(night):
    """
    Finds the directory for a night from the base directory (for looping over standards).
    """
    return f"{base_dir}/{night}"

# store calibrations to avoid rerunning pipeline
CALIBRATION_FILE = f"{output_dir}/calibration.json"

def save_calibration(calibration, path=CALIBRATION_FILE):
    with open(path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"Calibration saved to {path}")

def load_calibration(path=CALIBRATION_FILE):
    with open(path, "r") as f:
        calibration = json.load(f)
    print(f"Calibration loaded from {path}")
    return calibration

class PhotometryDataManager:

    def __init__(self, input_dir, output_dir):
        """
        Store parameters for I/O.
        """
        self.base_dir = Path(base_dir)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.date = Path(input_dir).name

    def find_cepheid_files(self):
        """
        Locates reduced cepheid files and returns a list.
        """
        return sorted(self.input_dir.glob("cepheid_*_stacked.fits"))
    
    def find_standard_files(self):
        """
        Locates reduced standard files and returns a list.
        """
        return sorted(self.input_dir.glob("standard_*_stacked.fits"))
    
    def extract_cepheid_id(self, filename):
        """
        Extracts the number from the cepheid file name so they may be distinguished.
        """
        match = re.search(r'cepheid[_-]?(\d+)', filename.lower())
        return f"{int(match.group(1)):02d}" if match else None
    
    def extract_standard_id(self, filename):
        """
        Extracts the name of the standard file so they may be distinguished (slightly harder because standards are grouped by name).
        """
        filename_lower = filename.lower()
        for std_id in ALL_STANDARD_IDS:
            std_id_pattern = std_id.lower().replace("_", "[_-]?")
            if re.search(std_id_pattern, filename_lower):
                return std_id
        return None
    
    def save_results(self, df, filename):
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        return path
    
class SinglePhotometry:
    """
    Performs photometry on one .fits file.
    """
    def __init__(self, fits_path, x_rough, y_rough, name, ebv):
        # data management
        self.fits_path = Path(fits_path)
        self.name = name
        self.ebv = ebv
        self.date = self.fits_path.parent.name

        # coordinates & apertures
        self.x_rough = x_rough
        self.y_rough = y_rough
        self.ap = AperturePhotometry(str(fits_path))

        # ccd parameters
        self.ccd_params()

    def isot_time(self):
        """
        Extract the ISOT time from the .fits header (needed for period fitting).
        """
        return self.ap.header.get("DATE-OBS")

    def diagnose_wcs(self):
        """
        Check if WCS is reasonable.
        
        NOW DEFUNCT!!! (Using DAOStarFinder)
        """
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = WCS(self.ap.header)
        
        print(f"\nWCS Diagnosis for {self.name}:")
        print(f"  CRVAL1 (RA): {self.ap.header.get('CRVAL1', 'MISSING')}")
        print(f"  CRVAL2 (Dec): {self.ap.header.get('CRVAL2', 'MISSING')}")
        print(f"  CD1_1: {self.ap.header.get('CD1_1', 'MISSING')}")
        print(f"  CD2_2: {self.ap.header.get('CD2_2', 'MISSING')}")
        
        # Try converting back
        x_test, y_test = self.ap.get_pixel_coords(self.ra, self.dec)
        ra_check, dec_check = self.ap.wcs.pixel_to_world_values(x_test, y_test)
        
        print(f"  Coord roundtrip error: ΔRA={abs(self.ra - ra_check):.4f}°, ΔDec={abs(self.dec - dec_check):.4f}°")

    def ccd_params(self):
        """
        Extract parameters relevant to CCD equation.
        """
        with fits.open(self.fits_path) as hdul:
            hdr = hdul[0].header  # assuming primary HDU stores relevant keywords
    
            gain = hdr.get('MEANGAIN', None) # e-/ADU
            read_noise = hdr.get('TOTRN', None)  # e-
            exp_time = hdr.get('TOTEXP', None) # seconds
            stack_size = hdr.get('NSTACK', 1) # default to 1 if single image
            
        self.gain = gain
        self.read_noise = read_noise
        self.exp_time = exp_time
        self.stack_size = stack_size

    def get_airmass(self):
        """
        Use .fits file headers to compute the airmass.
        """
        airmass_info = AirmassInfo(str(self.fits_path))
        airmass = airmass_info.process_fits(str(self.fits_path))
        airmass = airmass.value # extract value from astropy object
        
        return airmass

    def locate_star(self, x_guess, y_guess, fwhm=4.0, sigma=6.0, plot=False):
        """
        Uses DAOStarFinder with an initial guess to locate the target star in the image.
        Most stars seem to have fwhm ~ 4.0 so we'll run with that.

        Also creates a plot of all stars in the image and highlights the target star for completeness.
        """
        raw_data = self.ap.data * self.exp_time # DAOStarFinder prefers the raw data rather than normalised counts.

        mean, median, std = sigma_clipped_stats(self.ap.data, sigma=sigma)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma * std)
        sources = daofind(self.ap.data - median)

        # find the distances from stars in the image to where the initial guess was
        distances = np.sqrt((sources['xcentroid'] - x_guess)**2 + 
                        (sources['ycentroid'] - y_guess)**2)
        min_distance_index = np.argmin(distances) # locate the star closest to the star in the image

        if distances[min_distance_index] > 30:  # sanity check - nearest star shouldn't be far
            print(f"Nearest source is {distances[min_distance_index]:.1f}px away, using raw guess instead")
            nearest_x = float(sources['xcentroid'][min_distance_index])
            nearest_y = float(sources['ycentroid'][min_distance_index])
            print(f"Nearest source is at x: {nearest_x:.2f}, y: {nearest_y:.2f}")
            return x_guess, y_guess
        
        x_coord, y_coord = float(sources['xcentroid'][min_distance_index]), float(sources['ycentroid'][min_distance_index])

        if plot:
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
            all_aps = circ_ap(positions, r=6)
            target_ap = circ_ap((x_coord, y_coord), r=6)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(self.ap.data, origin='lower',
                    norm=LogNorm(vmin=np.median(self.ap.data), vmax=np.percentile(self.ap.data, 99)),
                    cmap='gray')
            all_aps.plot(ax=ax, color='blue', lw=0.8, alpha=0.4)
            target_ap.plot(ax=ax, color='red', lw=1.5, label=f"{self.name}")
            ax.set_title(f"{self.date} {self.name} Full Image: ({len(sources)} Sources Detected)")
            ax.legend(facecolor='white', edgecolor='black', framealpha=0.5)
            plt.show()

        return x_coord, y_coord

    def curve_of_growth(self, data, centroid, fwhm, inner=1.5, outer=2.0, plot=False):
        """
        Determine the optimal aperture size from a curve-of-growth plot.
        """
        ap_radii = np.arange(0.1, 4.1, 0.05) * fwhm # do a broad range of apertures
        fluxes = []
        bckgnds = []

        for radius in ap_radii:
            flux, _, bckgnd, _ = self.ap.aperture_photometry(
                data, centroid, ap_rad=radius, 
                inner=inner, outer=outer, plot=False
            )
            fluxes.append(flux if flux is not None else 0)
            bckgnds.append(bckgnd)

        bckgnds = np.array(bckgnds)
        normalised_ssaf = bckgnds / np.max(bckgnds)

        fluxes = np.array(fluxes)

        valid_mask = ~np.isnan(fluxes).flatten() # the ~ sign flips boolean values

        fluxes = fluxes[valid_mask]
        ap_radii = ap_radii[valid_mask]

        max_flux = np.max(fluxes)
        target_flux = 0.98 * max_flux

        print(f"Max flux: {max_flux:.3f}, target flux: {target_flux:.3f}")

        idx = np.argmax(fluxes >= target_flux) # locate the first aperture that contains 90% of maximum flux
        optimal_radius = ap_radii[idx]

        if plot:
            plt.plot(ap_radii, fluxes, color='red', marker='o')
            plt.xlabel('Aperture Radius [pix]')
            plt.ylabel('Sky-Subtracted Flux through Aperture [ADU/s]')
            plt.title(f"Curve of Growth for {self.name} on {self.date}")
            plt.show()

        return optimal_radius

    def raw_photometry(self, width, plot):
        """
        Raw photometry (computes instrumental magnitude and associated error.)
        """
        # locate approximate pixel coordinates of star
        x_guess, y_guess = self.locate_star(self.x_rough, self.y_rough, plot=plot)
        # cut out a 100x100 rectangle containing the star
        masked_data, x_offset, y_offset = self.ap.mask_data_and_plot(x_guess, y_guess, width=width, 
                                                                     name=self.name, date=self.date, plot=plot)

        # use centroiding to find the exact sub-pixel centre of the star
        centroid_local, fwhm = self.ap.get_centroid_and_fwhm(masked_data, self.name, plot=plot) # centroid for cutout
        centroid_global = (centroid_local[0] + x_offset, centroid_local[1] + y_offset) # centroid for full image

        # compute optimal aperture size from curve-of-growth analysis
        ap_rad = self.curve_of_growth(masked_data, centroid_local, fwhm, inner=1.5, outer=2.0, plot=plot)

        print(f"FWHM for {self.name}: {fwhm:.3f}\n")
        print(f"Aperture size for {self.name}: {ap_rad:.3f}")

        if plot:
            self.plot_psf_profile(masked_data, centroid_local, fwhm, self.name)

        # extract flux via full aperture photometry
        flux, ap_area, sky_bckgnd, annulus_area = self.ap.aperture_photometry(
            masked_data, centroid_local, ap_rad, ceph_name=self.name, date=self.date,
            inner=1.5, outer=2.0, plot=plot, savefig=False
        )

        # compute instrumental magnitudes and errors
        instrumental_mag = self.ap.instrumental_magnitude(flux)
        instrumental_mag_error = self.ap.get_inst_mag_error(flux, ap_area, sky_bckgnd, annulus_area,
                                                            self.gain, self.exp_time, self.read_noise, self.stack_size)
        
        return instrumental_mag, instrumental_mag_error

    def dust_correction(self):
        """
        Uses The Don's dust class to compute Av.
        """
        dust = DustExtinction(filter="V", colour_excess=float(self.ebv))
        A_V = dust.compute_extinction_mag()

        return A_V

    def standard_magnitudes(self, calibrations, plot):
        """
        Compute a standard magnitude, appropriately corrected.
        """
        m_inst, m_inst_err = self.raw_photometry(width=150, plot=plot)
        airmass = self.get_airmass()
        A_V = self.dust_correction()

        k = calibrations["k"]
        Z1 = calibrations["Z1"]
        k_err = calibrations["k_err"]
        Z1_err = calibrations["Z1_err"]

        m_true = (m_inst - A_V) + Z1 + (k * airmass)
        m_true_err = np.sqrt(m_inst_err**2 + (airmass * k_err)**2 + Z1_err**2)

        return m_true, m_true_err

    def plot_psf_profile(self, data, centroid, fwhm, name):
        """
        Radial flux profile centred on the star, with FWHM marked.
        """
        y_grid, x_grid = np.mgrid[:data.shape[0], :data.shape[1]]
        radii = np.sqrt((x_grid - centroid[0])**2 + (y_grid - centroid[1])**2)

        radii_flat = radii.flatten()
        flux_flat = data.flatten()

        # bin into annular rings
        max_radius = 5 * fwhm
        bin_edges = np.arange(0, max_radius, 0.5)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        binned_flux = np.zeros(len(bin_centres))

        for i in range(len(bin_centres)):
            mask = (radii_flat >= bin_edges[i]) & (radii_flat < bin_edges[i + 1])
            if np.any(mask):
                binned_flux[i] = np.mean(flux_flat[mask])

        # normalise to peak
        binned_flux /= np.max(binned_flux)

        plt.plot(bin_centres, binned_flux, 'k-', linewidth=1.5)
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        plt.axvline(fwhm / 2, color='blue', linestyle=':', alpha=0.5, label=f'FWHM/2 = {fwhm/2:.1f} px')
        plt.axvline(-fwhm / 2 + fwhm, color='blue', linestyle=':', alpha=0.5)
        plt.xlabel('Radius from centroid [pixels]')
        plt.ylabel('Normalised Flux')
        plt.title(f'Radial PSF profile — {name}')
        plt.legend()
        plt.show()
    
class Corrections:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.all_standards = []
        self.calibration = None

    def process_standards(self, plot):
        """
        Compute instrumental magnitude and its error for standard stars on the night.
        """
        # collect all standard star data across all nights to generate a global fit
        for night in ALL_CATALOGUES:
            cep_cat, std_cat = get_catalogues_for_night(night)
            if not std_cat:
                continue
            data_manager = PhotometryDataManager(input_dir_for(night), output_dir) # initialise data manager
            std_files = data_manager.find_standard_files() # locate standard files

            # loop over standard files
            for std_file in std_files: 
                std_id = data_manager.extract_standard_id(std_file.name) # extract their IDs
                if std_id is None or std_id not in std_cat:
                    print(f"Skipping unrecognised standard: {std_file.name}")
                    continue

                std_data = std_cat[std_id] # extract data
                x_g, y_g = get_pixel_guess(std_cat, std_id) # extract handwritten pixel guesses

                # photometry object
                phot = SinglePhotometry(
                fits_path=std_file,
                x_rough=x_g,
                y_rough=y_g,
                name=std_id,
                ebv=std_data["e(b-v)"], # might want to make this zero
                )
                
                m_inst, m_err = phot.raw_photometry(width=150, plot=plot)
                airmass = phot.get_airmass()

                result = {
                    "ID": std_id,
                    "V_true": float(std_data["mag"]),
                    "m_inst": m_inst,
                    "m_inst_err": m_err,
                    "airmass": airmass,
                    "ISOT": phot.isot_time()
                }

                self.all_standards.append(result)
            print(self.all_standards)

        # global extinction fit across all nights
        self.standards_df = pd.DataFrame(self.all_standards)
        #remove standard with id {'ID': 'G93_48', 'V_true': 12.74, 'm_inst': -7.1586712021845, 'm_inst_err': 0.005702656977785718, 'airmass': np.float64(1.2303238241120058), 'ISOT': '2025-10-14T19:42:57.4186109'}, since outlier in data
        #self.standards_df = self.standards_df[self.standards_df['ID'] != 'G93_48']

        print(f"\nTotal standards collected for airmass fit: {len(self.standards_df)}")

    def fit_extinction(self):
        """
        Use Harsha's airmass class to fit extinction.
        """
        airmass_fitter = Airmass(
            airmass = self.standards_df['airmass'].values,
            Vmag=self.standards_df["V_true"].values,
            m_inst=self.standards_df["m_inst"].values,
            m_err=self.standards_df["m_inst_err"].values
        )

        k, Z1, k_err, Z1_err = airmass_fitter.fit_extinction_weighted()
        airmass_fitter.plot_atmospheric_extinction(k, Z1, k_err, Z1_err)
        airmass_fitter.plot_parameter_space(k, Z1, k_err, Z1_err)
        airmass_fitter.plot_residuals(k, Z1)
        outlier_mask = airmass_fitter.remove_outliers(k, Z1)
        #if there are any outliers, plot the fit again without them to show the improvement.
        
        if np.any(outlier_mask): 
            print(f"Outliers detected: {self.standards_df[outlier_mask]['ID'].values}") 
            self.standards_df = self.standards_df[~outlier_mask]
            # refit without outliers
            airmass_fitter_update = Airmass(
                airmass = self.standards_df['airmass'].values,
                Vmag=self.standards_df["V_true"].values,
                m_inst=self.standards_df["m_inst"].values,
                m_err=self.standards_df["m_inst_err"].values
            )

            k, Z1, k_err, Z1_err = airmass_fitter_update.fit_extinction_weighted()
            airmass_fitter_update.plot_atmospheric_extinction(k, Z1, k_err, Z1_err)
            airmass_fitter_update.plot_parameter_space(k, Z1, k_err, Z1_err)
            airmass_fitter_update.plot_residuals(k, Z1)   

        calibration = {
        "k": k,
        "Z1": Z1,
        "k_err": k_err,
        "Z1_err": Z1_err
        }

        return calibration
    
class DifferentialCorrections:
    """
    Differential photometry: analysing reference stars for each cepheid in order to correct for night-by-night variations.
    """
    def __init__(self, cepheid_id, fits_path, reference_catalogue, calibration, reference_flipstat):
        self.cepheid_id = cepheid_id
        self.fits_path = fits_path
        self.refs = reference_catalogue.get(cepheid_id, {})
        self.calibration = calibration # standard star airmass correction

        self.reference_flipstat = reference_flipstat
        self.flipped = self.check_flip()

    @staticmethod
    def flip_coords(x, y):
        """
        Flips y-coordinate of the image (images can only be flipped 180 degrees).
        """
        img_size = 3995 # size of image after reduction (3995 because image goes from -0.5->3995.5)
        return img_size - x, img_size - y

    def get_reference_star_coords(self, ref_data):
        """
        Extracts coordinates for five bright reference standard stars.
        """
        x = float(ref_data["x-coord"])
        y = float(ref_data["y-coord"])
        if self.flipped:
            print(f"Original coords are x: {x}, y: {y}")
            x, y = self.flip_coords(x, y)
            print(f"Flipped coords are x: {x}, y: {y}")
        return x, y
    
    def check_flip(self):
        """
        Checks whether the image is flipped relative to the reference direction from 2025-10-06.
        So, if this is detecting flips on the testing night, something is very wrong.
        """
        with fits.open(self.fits_path) as hdul:
            current_flipstat = hdul[0].header.get("FLIPSTAT", None)
        
        if current_flipstat is None:
            warnings.warn(f"No FLIPSTAT in {self.fits_path.name}")
            return False
        
        if current_flipstat != self.reference_flipstat:
            print(f"Image is flipped!")
        else:
            print(f"Image is not flipped.")
        
        return current_flipstat != self.reference_flipstat
    
    def count_matches(self, use_flip):
        """
        Determines how many bright reference stars are found for a given image orientation.
        Reuses locate_star code but not too much.

        DEFUNCT!!!

        """
        ap = AperturePhotometry(str(self.fits_path))
        mean, median, std = sigma_clipped_stats(ap.data, sigma=6.0)
        daofind = DAOStarFinder(fwhm=4.0, threshold=6.0 * std)
        sources = daofind(ap.data - median)

        if sources is None:
            return 0

        n_found = 0
        for _, ref_data in self.refs.items():
            try:
                x_g = float(ref_data["x-coord"])
                y_g = float(ref_data["y-coord"])
                if use_flip:
                    x_g, y_g = self.flip_coords(x_g, y_g)

                distances = np.sqrt((sources['xcentroid'] - x_g)**2 +
                                    (sources['ycentroid'] - y_g)**2)
                if np.min(distances) < 80: # 80 pix threshold (from workers)
                    n_found += 1
            except Exception:
                continue
        return n_found
    
    def detect_and_correct_flip(self):
        """
        Detects whether the image is flipped by checking how many stars are found according to their assigned coordinates.

        DEFUNCT!!!

        """
        n_normal = self.count_matches(use_flip=False)
        if n_normal >= 3: # this threshold should work
            self.flipped = False
            return

        n_flipped = self.count_matches(use_flip=True)
        if n_flipped > n_normal:
            self.flipped = True
            print(f"Image flip detected for {self.fits_path.name}")
        else:
            self.flipped = False
            print(f"Reference stars poorly-matched for {self.fits_path.name}")

    def measure_references(self, plot=False):
        """
        Compute standard magnitudes of reference stars and compare offset from true value.
        Provides estimate of empirical error.
        """
        # standard calibrations
        k = self.calibration["k"]
        Z1 = self.calibration["Z1"]

        offsets = []

        if plot:
            ap = AperturePhotometry(str(self.fits_path))
            fig, ax = plt.subplots(figsize=(8, 8))
            night = Path(self.fits_path).parent.name
            ax.set_title(f"Cepheid {self.cepheid_id} reference stars — {night}")
            ax.imshow(ap.data, origin='lower',
                    norm=LogNorm(vmin=np.median(ap.data),
                                vmax=np.percentile(ap.data, 99)),
                    cmap='gray')

        # need extensive error statements in these pipelines to prevent hanging...
        for ref_id, ref_data in self.refs.items():
            try:
                x_g, y_g = self.get_reference_star_coords(ref_data)
                V_known = float(ref_data["V_true"])

                phot = SinglePhotometry(
                    fits_path=self.fits_path,
                    x_rough=x_g,
                    y_rough=y_g,
                    name=ref_id,
                    ebv="0.0", # no dust correction
                )

                m_inst, m_inst_err = phot.raw_photometry(width=150, plot=False)
                airmass = phot.get_airmass()
                m_cal = m_inst + Z1 + k * airmass

                offsets.append(m_cal - V_known)

                if plot:
                    x_found, y_found = phot.locate_star(x_g, y_g, plot=False)
                    ref_ap = circ_ap((x_found, y_found), r=8)
                    ref_ap.plot(ax=ax, color='blue', lw=1.5,
                    label='Reference Stars' if ref_id == list(self.refs.keys())[0] else None)

            except Exception as e:
                print(f"  Reference {ref_id} failed: {e}")
                continue

        if plot:
            ax.legend(facecolor='white', edgecolor='black', framealpha=0.5)
            plt.show()

        if len(offsets) < 3:
            raise ValueError(
                f"Only {len(offsets)} refs succeeded for Cepheid "
                f"{self.cepheid_id}, need >= 3"
            )

        self.offsets = np.array(offsets)
        return self.offsets
    
    def compute_offset(self):
        """
        Compute the offset of the measured V magnitudes from their true values.
        """
        delta = np.median(self.offsets) # remove any wacky outliers
        delta_err = np.std(self.offsets) / np.sqrt(len(self.offsets))
        self.delta = delta
        self.delta_err = delta_err
        return delta, delta_err
    
    def apply(self, ceph_m_calibrated, ceph_m_calibrated_err, plot=False):
        """
        Apply the empirical correction to the cepheid magnitude. This is the normalisation.
        """
        self.measure_references(plot=plot)
        self.compute_offset()

        m_corrected = ceph_m_calibrated - self.delta
        m_corrected_err = np.sqrt(ceph_m_calibrated_err**2 + self.delta_err**2)
        return m_corrected, m_corrected_err
    
def plot_full_sky_subtracted(fits_path, name, date):
    """
    Bins the image and sky-subtracts each part individually.
    """
    ap = AperturePhotometry(str(fits_path))
    bkg = Background2D(ap.data, box_size=128, bkg_estimator=MedianBackground())
    subtracted = ap.data - bkg.background

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(subtracted, origin='lower', cmap='gray',
            vmin=0, vmax=np.percentile(subtracted[subtracted > 0], 99))
    ax.set_xlabel('x [pixels]')
    ax.set_ylabel('y [pixels]')
    ax.set_title(f"Sky-Subtracted Full Image: {name}, ({date})")
    plt.show()

def main(night, diagnostic_plot=False, refit_calibration=False):
    """
    Runs the full photometry pipeline for a given night.
    """

    # check if the calibration is saved 
    if refit_calibration or not Path(CALIBRATION_FILE).exists():
        # run the pipeline
        corrections = Corrections(output_dir)
        corrections.process_standards(plot=diagnostic_plot)
        calibration = corrections.fit_extinction()
        save_calibration(calibration) # save it as a .json file (readable format, see if the messy standard is included).
    else:
        calibration = load_calibration()

    # standard calibrations
    k = calibration["k"]
    Z1 = calibration["Z1"]


    cep_cat, _ = get_catalogues_for_night(night)
    if not cep_cat:
        print(f"No cepheid catalogue exists for {night}")
        return
    
    data_manager = PhotometryDataManager(input_dir_for(night), output_dir)
    results = []

    cep_files = data_manager.find_cepheid_files()

    # pull out background subtracted MW Cyg from every night
    if diagnostic_plot and len(cep_files) > 0:
        plot_full_sky_subtracted(cep_files[0], "MW Cyg", night)

    for cep_file in cep_files:
        cep_id = data_manager.extract_cepheid_id(cep_file.name)
        if cep_id is None or cep_id not in cep_cat:
            continue
        cep_data = cep_cat[cep_id]
        x_g, y_g = get_pixel_guess(cep_cat, cep_id)

        phot = SinglePhotometry(
            fits_path=cep_file,
            x_rough=x_g, 
            y_rough=y_g,
            name=cep_data["name"], 
            ebv=cep_data["e(b-v)"],
        )

        # this needs to be done in a somewhat roundabout way
        m_standard, m_standard_err = phot.standard_magnitudes(calibration, plot=diagnostic_plot)

        # we then basically type out the calibrated/standard magnitude for the differential magnitude
        m_inst, m_inst_err = phot.raw_photometry(width=150, plot=False)
        airmass = phot.get_airmass()
        k, Z1 = calibration["k"], calibration["Z1"]
        k_err, Z1_err = calibration["k_err"], calibration["Z1_err"]

        m_calibrated = m_inst + Z1 + k * airmass

        diff = DifferentialCorrections(cep_id, cep_file, reference_catalogue, calibration, cepheid_orientation_catalogue[cep_id])
        m_corrected, m_corrected_err = diff.apply(m_calibrated, m_inst_err, plot=diagnostic_plot)

        A_V = phot.dust_correction()
        m_diff = m_corrected - A_V
        m_diff_err = np.sqrt(m_corrected_err**2 + (airmass * k_err)**2 + Z1_err**2)

        results.append({
            "ID": cep_id,
            "Name": cep_data["name"],
            "Night": night,
            "ISOT": phot.isot_time(),
            "m_standard": m_standard,
            "m_standard_err": m_standard_err,
            "m_differential": m_diff,
            "m_differential_err": m_diff_err,
        })

    df = pd.DataFrame(results)
    filename = f"photometry_{night}.csv"
    df.to_csv(f"{output_dir}/{filename}", index=False)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main("2025-09-22", diagnostic_plot=True)  # put in night syntax as needed
