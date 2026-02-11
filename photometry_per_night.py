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

# our classes
from Cepheid_apertures import AperturePhotometry
from Cepheid_apertures import Airmass
from Cepheid_apertures import DustExtinction
from AirmassInfo import AirmassInfo
from catalogues import ALL_CATALOGUES, get_catalogues_for_night, get_pixel_guess

# updated DAOStarFinder
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture as circ_ap
from matplotlib.colors import LogNorm

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

# quick helper function
def input_dir_for(night):
    """
    Finds the directory for a night from the base directory (for looping over standards).
    """
    return f"{base_dir}/{night}"

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

    def locate_star(self, x_guess, y_guess, fwhm=4.0, sigma=3.0, plot=True):
        """
        Uses DAOStarFinder with an initial guess to locate the target star in the image.
        Most stars seem to have fwhm ~ 4.0 so we'll run with that.

        Also creates a plot of all stars in the image and highlights the target star for completeness.
        """
        raw_data = self.ap.data * self.exp_time # DAOStarFinder prefers the raw data rather than normalised counts.

        mean, median, std = sigma_clipped_stats(raw_data, sigma=sigma)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma * std)
        sources = daofind(raw_data - median)

        # find the distances from stars in the image to where the initial guess was
        distances = np.sqrt((sources['xcentroid'] - x_guess)**2 + 
                        (sources['ycentroid'] - y_guess)**2)
        min_distance_index = np.argmin(distances) # locate the star closest to the star in the image

        if distances[min_distance_index] > 30:  # sanity check - nearest star shouldn't be far
            print(f"Nearest source is {distances[min_distance_index]:.1f}px away, using raw guess instead")
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
            ax.set_title(f"{self.name} Full Image: ({len(sources)} Sources Detected)")
            plt.show()

        return x_coord, y_coord

    def curve_of_growth(self, data, centroid, fwhm, inner=1.5, outer=2.0, plot=False):
        """
        Determine the optimal aperture size from a curve-of-growth plot.
        """
        ap_radii = np.arange(0.1, 5.1, 0.05) * fwhm # do a broad range of apertures
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
        target_flux = 0.9 * max_flux

        print(f"Max flux: {max_flux}, target flux: {target_flux}")

        idx = np.argmax(fluxes >= target_flux) # locate the first aperture that contains 90% of maximum flux
        optimal_radius = ap_radii[idx]

        if plot:
            plt.plot(ap_radii, fluxes, color='red', marker='o')
            plt.xlabel('Aperture Radius [pix]')
            plt.ylabel('Sky-Subtracted Flux through Aperture [Arbitrary Units]')
            plt.title("Normalised curve of growth")
            plt.show()

        return optimal_radius

    def raw_photometry(self, width):
        """
        Raw photometry (computes instrumental magnitude and associated error.)
        """
        # locate approximate pixel coordinates of star
        x_guess, y_guess = self.locate_star(self.x_rough, self.y_rough)
        print(f"X-guess: {x_guess}, Y-guess: {y_guess}.")
        # cut out a 100x100 rectangle containing the star
        masked_data, x_offset, y_offset = self.ap.mask_data_and_plot(x_guess, y_guess, width=width, plot=True)

        # use centroiding to find the exact sub-pixel centre of the star
        centroid_local, fwhm = self.ap.get_centroid_and_fwhm(masked_data, self.name, plot=True) # centroid for cutout
        centroid_global = (centroid_local[0] + x_offset, centroid_local[1] + y_offset) # centroid for full image

        # compute optimal aperture size from curve-of-growth analysis
        ap_rad = self.curve_of_growth(masked_data, centroid_local, fwhm, inner=1.5, outer=2.0, plot=True)

        # extract flux via full aperture photometry
        flux, ap_area, sky_bckgnd, annulus_area = self.ap.aperture_photometry(
            masked_data, centroid_local, ap_rad, ceph_name=self.name, date=self.date,
            inner=1.5, outer=2.0, plot=True, savefig=False
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

    def standard_magnitudes(self, calibrations):
        """
        Compute a standard magnitude, appropriately corrected.
        """
        m_inst, m_inst_err = self.raw_photometry(width=50)
        airmass = self.get_airmass()
        A_V = self.dust_correction()

        k = calibrations["k"]
        Z1 = calibrations["Z1"]
        k_err = calibrations["k_err"]
        Z1_err = calibrations["Z1_err"]

        m_true = (m_inst - A_V) + Z1 + (k * airmass)
        m_true_err = np.sqrt(m_inst_err**2 + (airmass * k_err)**2 + Z1_err**2)

        return m_true, m_true_err
    
class Corrections:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.all_standards = []
        self.calibration = None

    def process_standards(self):
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
                
                m_inst, m_err = phot.raw_photometry(width=50)
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

        # global extinction fit across all nights
        self.standards_df = pd.DataFrame(self.all_standards)
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
        airmass_fitter.plot_atmospheric_extinction()
        airmass_fitter.plot_parameter_space()

        calibration = {
        "k": k,
        "Z1": Z1,
        "k_err": k_err,
        "Z1_err": Z1_err
        }

        return calibration
    
def main(night):
    """
    Runs the full photometry pipeline for a given night.
    """
    corrections = Corrections(output_dir)
    corrections.process_standards()
    calibration = corrections.fit_extinction()

    cep_cat, _ = get_catalogues_for_night(night)
    if not cep_cat:
        print(f"No cepheid catalogue exists for {night}")
        return
    
    data_manager = PhotometryDataManager(input_dir_for(night), output_dir)
    results = []

    for cep_file in data_manager.find_cepheid_files():
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

        m_true, m_err = phot.standard_magnitudes(calibration)
        results.append({
            "ID": cep_id,
            "Name": cep_data["name"],
            "Night": night,
            "ISOT": phot.isot_time(),
            "m_true": m_true,
            "m_err": m_err
        })

    df = pd.DataFrame(results)
    filename = f"photometry_{night}.csv"
    df.to_csv(f"{output_dir}/{filename}", index=False)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main("2025-10-06")  # put in night syntax as needed
