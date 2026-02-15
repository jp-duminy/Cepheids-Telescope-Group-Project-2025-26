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

from Cepheid_apertures import AperturePhotometry
from Cepheid_apertures import Airmass
from Cepheid_apertures import DustExtinction
from AirmassInfo import AirmassInfo
from catalogues import get_catalogues_for_night, get_pixel_guess

from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture as circ_ap
from matplotlib.colors import LogNorm

import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False

# Directories
input_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Andromeda/reduced"
output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Andromeda/Photometry"
#base_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids"
base_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_standard_stars"

# Standard star catalogue with BOTH B and V magnitudes
standard_catalogue = {
    "F_108": {
        "ra": "+23 16 12.0",
        "dec": "-01 50 35.0",
        "mag_V": "12.96",
        "mag_B": "12.73",
        "e(b-v)": "0.0016",
        "x_guess_V": "2070.9942",
        "y_guess_V": "1950.8339",
        "x_guess_B": "2008.7393",
        "y_guess_B": "1956.1373"
    },
    "SA112_595": {
        "ra": "+20 41 19.0",
        "dec": "+00 16 11.0",
        "mag_V": "11.35",
        "mag_B": "12.95",
        "e(b-v)": "0.0016",
        "x_guess_V": "2034.8409",
        "y_guess_V": "1992.7936",
        "x_guess_B": "2050.9755",
        "y_guess_B": "1971.9336"
    },
    "GD_246": {
        "ra":  "+23 12 21.6",
        "dec": "+10 47 04.0",
        "mag_V": "13.09",
        "mag_B": "12.77",
        "e(b-v)": "0.0015",
        "x_guess_V": "2525.3814",
        "y_guess_V": "1569.7755",
        "x_guess_B": "1577.5708",
        "y_guess_B": "2387.9009"
    }
}

# Andromeda catalogue - ADD YOUR PIXEL GUESSES HERE
andromeda_catalogue = {
    "CV1": {
        "ra": "+00 41 27.30",
        "dec": "+41 10 10.4",
        "e(b-v)": 0.06,
        "name": "Andromeda CV1",
        "x_guess_V": 2095.1935,
        "y_guess_V": 1996.0093,
        "x_guess_B": 2088.5409,
        "y_guess_B": 1987.3803
    }
}

ALL_STANDARD_IDS = {"114176", "SA111775", "F_108", "SA112_595", "GD_246", "G93_48", "G156_31"}

def input_dir_for(night = "2025-10-06"):
    """Finds the directory for a night from the base directory."""
    return f"{base_dir}/{night}"

class PhotometryDataManager:
    def __init__(self, input_dir, output_dir, standards_dir=None, standards_night="2025-10-06"):
        """Store parameters for I/O."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        # FIX: Look in the date subdirectory
        if standards_dir:
            self.standards_dir = Path(standards_dir) / standards_night
        else:
            self.standards_dir = self.input_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_andromeda_files(self):
        """Locates the two Andromeda files and returns them organized by filter."""
        files = {
            'B': self.input_dir / "Andromeda_B_reduced.fits",
            'V': self.input_dir / "Andromeda_V_reduced.fits"
        }
        
        for filter_name, filepath in files.items():
            if not filepath.exists():
                print(f"Warning: {filepath} not found!")
        
        return files
    
    def find_standard_files(self, filter_type):
        """
        Locates reduced standard files for a specific filter (B or V).
        UPDATED to match actual file naming convention
        """
        if filter_type == 'B':
            # Files like: standard_01_F_108_Filter_B_stacked.fits
            pattern = "standard_*_Filter_B_stacked.fits"
        else:  # V filter
            # Files like: standard_01_F_108_Filter_V_stacked.fits
            pattern = "standard_*_Filter_V_stacked.fits"
        
        files = sorted(self.standards_dir.glob(pattern))
        
        print(f"Found {len(files)} standard star files for {filter_type} filter in {self.standards_dir}:")
        for f in files:
            print(f"  - {f.name}")
        
        return files
    
    def extract_standard_id(self, filename):
        """Extracts the ID of a standard star."""
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
    """Performs photometry on one .fits file."""
    
    def __init__(self, fits_path, x_rough, y_rough, name, ebv, filter_type):
        self.fits_path = Path(fits_path)
        self.name = name
        self.ebv = ebv
        self.filter_type = filter_type
        self.x_rough = x_rough
        self.y_rough = y_rough
        self.date = "Andromeda"  # Fixed: set date attribute
        
        self.ap = AperturePhotometry(str(fits_path))
        self.ccd_params()

    def isot_time(self):
        """Extract the ISOT time from the .fits header."""
        return self.ap.header.get("DATE-OBS")

    def ccd_params(self):
        """Extract parameters relevant to CCD equation."""
        with fits.open(self.fits_path) as hdul:
            hdr = hdul[0].header
            self.gain = hdr.get('GAIN', None)
            self.read_noise = hdr.get('RDNOISE', None)
            self.exp_time = hdr.get('EXPTIME', None)
            self.stack_size = hdr.get('NSTACK', 1)

    def get_airmass(self):
        """Use .fits file headers to compute the airmass."""
        airmass_info = AirmassInfo(str(self.fits_path))
        airmass = airmass_info.process_fits(str(self.fits_path))
        return airmass.value

    def locate_star(self, x_guess, y_guess, fwhm=4.0, sigma=6.0, plot=True):
        """Uses DAOStarFinder to locate the target star."""
        mean, median, std = sigma_clipped_stats(self.ap.data, sigma=sigma)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma * std)
        sources = daofind(self.ap.data - median)

        distances = np.sqrt((sources['xcentroid'] - x_guess)**2 + 
                          (sources['ycentroid'] - y_guess)**2)
        min_distance_index = np.argmin(distances)

        if distances[min_distance_index] > 30:
            print(f"Nearest source is {distances[min_distance_index]:.1f}px away, using raw guess instead")
            return x_guess, y_guess
        
        x_coord = float(sources['xcentroid'][min_distance_index])
        y_coord = float(sources['ycentroid'][min_distance_index])

        if plot:
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
            all_aps = circ_ap(positions, r=6)
            target_ap = circ_ap((x_coord, y_coord), r=6)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(self.ap.data, origin='lower',
                    norm=LogNorm(vmin=np.median(self.ap.data), 
                               vmax=np.percentile(self.ap.data, 99)),
                    cmap='gray')
            all_aps.plot(ax=ax, color='blue', lw=0.8, alpha=0.4)
            target_ap.plot(ax=ax, color='red', lw=1.5, label=f"{self.name}")
            ax.set_title(f"{self.filter_type}-filter {self.name}: {len(sources)} Sources Detected")
            ax.legend(facecolor='white', edgecolor='black', framealpha=0.5)
            plt.show()

        return x_coord, y_coord

    def curve_of_growth(self, data, centroid, fwhm, inner=1.5, outer=2.0, plot=False):
        """Determine the optimal aperture size from a curve-of-growth plot."""
        ap_radii = np.arange(0.1, 5.1, 0.1) * fwhm
        fluxes = []

        for radius in ap_radii:
            flux, _, _, _ = self.ap.aperture_photometry(
                data, centroid, ap_rad=radius, 
                inner=inner, outer=outer, plot=False
            )
            fluxes.append(flux if flux is not None else 0)

        fluxes = np.array(fluxes)
        valid_mask = ~np.isnan(fluxes).flatten()
        fluxes = fluxes[valid_mask]
        ap_radii = ap_radii[valid_mask]

        max_flux = np.max(fluxes)
        target_flux = 0.9 * max_flux
        idx = np.argmax(fluxes >= target_flux)
        optimal_radius = ap_radii[idx]

        if plot:
            plt.plot(ap_radii, fluxes, color='red', marker='o')
            plt.xlabel('Aperture Radius [pix]')
            plt.ylabel('Sky-Subtracted Flux [Arbitrary Units]')
            plt.title("Curve of Growth")
            plt.show()

        return optimal_radius

    def raw_photometry(self, width):
        """Raw photometry (computes instrumental magnitude and error)."""
        x_guess, y_guess = self.locate_star(self.x_rough, self.y_rough, plot=False)
        print(f"Located star at: X={x_guess:.2f}, Y={y_guess:.2f}")
        
        masked_data, x_offset, y_offset = self.ap.mask_data_and_plot(
            x_guess, y_guess, width=width, 
            name=self.name, date=self.date, plot=False
        )

        centroid_local, fwhm = self.ap.get_centroid_and_fwhm(
            masked_data, self.name, plot=False
        )
        
        ap_rad = self.curve_of_growth(
            masked_data, centroid_local, fwhm, 
            inner=1.5, outer=2.0, plot=False
        )

        print(f"FWHM: {fwhm:.3f}, Aperture radius: {ap_rad:.3f}")

        flux, ap_area, sky_bckgnd, annulus_area = self.ap.aperture_photometry(
            masked_data, centroid_local, ap_rad,
            ceph_name=self.name, date=self.date,
            inner=1.5, outer=2.0, plot=False, savefig=False
        )

        instrumental_mag = self.ap.instrumental_magnitude(flux)
        instrumental_mag_error = self.ap.get_inst_mag_error(
            flux, ap_area, sky_bckgnd, annulus_area,
            self.gain, self.exp_time, self.read_noise, self.stack_size
        )
        
        return instrumental_mag, instrumental_mag_error

    def dust_correction(self):
        """Compute extinction magnitude for the appropriate filter."""
        dust = DustExtinction(filter=self.filter_type, colour_excess=float(self.ebv))
        A_filter = dust.compute_extinction_mag()
        return A_filter

    def standard_magnitudes(self, calibrations):
        """Compute a standard magnitude, appropriately corrected."""
        m_inst, m_inst_err = self.raw_photometry(width=200)
        airmass = self.get_airmass()
        A_filter = self.dust_correction()

        k = calibrations["k"]
        Z1 = calibrations["Z1"]
        k_err = calibrations["k_err"]
        Z1_err = calibrations["Z1_err"]

        m_true = (m_inst - A_filter) + Z1 + (k * airmass)
        m_true_err = np.sqrt(m_inst_err**2 + (airmass * k_err)**2 + Z1_err**2)

        return m_true, m_true_err

class Corrections:
    def __init__(self, output_dir, filter_type):
        self.output_dir = Path(output_dir)
        self.filter_type = filter_type
        self.standards_results = []

    def process_standards(self, data_manager):
        """
        Process standard stars for calibration using the standard_catalogue.
        """
        print(f"\nProcessing {self.filter_type} filter standards...")
        
        # Find all standard files for this filter in the single night
        std_files = data_manager.find_standard_files(self.filter_type)
        
        # Use the standard_catalogue defined at the module level
        for std_id, std_data in standard_catalogue.items():
            # Check if this standard has the required magnitude for this filter
            mag_key = f"mag_{self.filter_type}"
            if mag_key not in std_data:
                print(f"Skipping {std_id}: no {mag_key} in catalogue")
                continue
            
            # Get filter-specific pixel coordinates
            x_guess_key = f"x_guess_{self.filter_type}"
            y_guess_key = f"y_guess_{self.filter_type}"
            
            if x_guess_key not in std_data or y_guess_key not in std_data:
                print(f"Skipping {std_id}: no pixel coordinates for {self.filter_type} filter")
                continue
            
            # Look for files matching this standard
            matching_files = [f for f in std_files if std_id.lower().replace('_', '') in f.name.lower().replace('_', '').replace('-', '')]
            
            for std_file in matching_files:
                print(f"Processing {std_id} ({self.filter_type} filter)...")
                
                try:
                    x_g = float(std_data[x_guess_key])
                    y_g = float(std_data[y_guess_key])
                    
                    phot = SinglePhotometry(
                        fits_path=std_file,
                        x_rough=x_g,
                        y_rough=y_g,
                        name=std_id,
                        ebv=float(std_data["e(b-v)"]),
                        filter_type=self.filter_type
                    )
                    
                    m_inst, m_err = phot.raw_photometry(width=150)
                    airmass = phot.get_airmass()
                    
                    result = {
                        "ID": std_id,
                        f"{self.filter_type}_true": float(std_data[mag_key]),
                        "m_inst": m_inst,
                        "m_inst_err": m_err,
                        "airmass": airmass
                    }
                    
                    self.standards_results.append(result)
                    print(f"  ✓ Success: m_inst = {m_inst:.3f} ± {m_err:.3f}, airmass = {airmass:.3f}")
                    
                except Exception as e:
                    print(f"  ✗ ERROR processing {std_id}: {e}")
                    continue
        
        if len(self.standards_results) == 0:
            print(f"\n⚠ WARNING: No standards processed for {self.filter_type} filter!")
            print("This will cause calibration to fail.")
            self.standards_df = pd.DataFrame()
            return False
        
        self.standards_df = pd.DataFrame(self.standards_results)
        print(f"\n✓ Total {self.filter_type} standards collected: {len(self.standards_df)}")
        
        # Save standards results
        standards_file = self.output_dir / f"standards_{self.filter_type}_filter.csv"
        self.standards_df.to_csv(standards_file, index=False)
        print(f"Standards saved to: {standards_file}")
        return True

    def fit_extinction(self):
        """Fit extinction for the specific filter."""
        if not hasattr(self, 'standards_df') or len(self.standards_df) == 0:
            print(f"\nERROR: No standards data available for {self.filter_type} filter!")
            print("Cannot perform calibration.")
            return None
        
        print(f"\nFitting atmospheric extinction for {self.filter_type} filter...")
        
        airmass_fitter = Airmass(
            airmass=self.standards_df['airmass'].values,
            Vmag=self.standards_df[f"{self.filter_type}_true"].values,
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
                Vmag=self.standards_df[f"{self.filter_type}_true"].values,
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
        
        print(f"\nCalibration for {self.filter_type} filter:")
        print(f"  k = {k:.4f} ± {k_err:.4f}")
        print(f"  Z1 = {Z1:.4f} ± {Z1_err:.4f}")

        return calibration

def process_filter(filter_type, andromeda_data_manager, output_dir):
    """Process Andromeda photometry for a single filter (B or V)."""
    print(f"\n{'='*60}")
    print(f"Processing {filter_type} filter")
    print(f"{'='*60}\n")
    
    # Process standard stars for calibration
    corrections = Corrections(output_dir, filter_type=filter_type)
    corrections.process_standards(andromeda_data_manager)
    calibration = corrections.fit_extinction()
    
    # Process Andromeda file
    andromeda_files = andromeda_data_manager.find_andromeda_files()
    andromeda_file = andromeda_files[filter_type]
    
    if not andromeda_file.exists():
        print(f"Error: {andromeda_file} not found!")
        return None
    
    print(f"\nProcessing Andromeda CV1 in {filter_type} filter...")
    
    andromeda_data = andromeda_catalogue["CV1"]
    
    # Get pixel guess for this filter
    x_guess_key = f"x_guess_{filter_type}"
    y_guess_key = f"y_guess_{filter_type}"
    
    if andromeda_data[x_guess_key] is None or andromeda_data[y_guess_key] is None:
        print(f"ERROR: You need to add pixel guesses for CV1 in {filter_type} filter!")
        print(f"Please update andromeda_catalogue with {x_guess_key} and {y_guess_key}")
        return None
    
    phot = SinglePhotometry(
        fits_path=andromeda_file,
        x_rough= float(andromeda_data[x_guess_key]),
        y_rough=float(andromeda_data[y_guess_key]),
        name=andromeda_data["name"],
        ebv=andromeda_data["e(b-v)"],
        filter_type=filter_type
    )
    
    m_true, m_err = phot.standard_magnitudes(calibration)
    
    result = {
        "Name": andromeda_data["name"],
        "Filter": filter_type,
        f"m_{filter_type}": m_true,
        f"m_{filter_type}_err": m_err,
        "RA": andromeda_data["ra"],
        "Dec": andromeda_data["dec"],
        "E(B-V)": andromeda_data["e(b-v)"],
        "Calibration_k": calibration["k"],
        "Calibration_Z1": calibration["Z1"]
    }
    
    df = pd.DataFrame([result])
    filename = f"andromeda_CV1_{filter_type}.csv"
    saved_path = andromeda_data_manager.save_results(df, filename)
    
    print(f"\n{filter_type} filter photometry complete!")
    print(f"Results saved to: {saved_path}")
    print(f"m_{filter_type} = {m_true:.3f} ± {m_err:.3f}")
    
    return result

def main():
    """Main function to process both B and V filters for Andromeda."""

    print("="*60)
    print("ANDROMEDA CV1 PHOTOMETRY PIPELINE")
    print("="*60)
    
    try:
        # FIX: Pass the standards directory and night
        andromeda_manager = PhotometryDataManager(
            input_dir=input_dir,
            output_dir=output_dir,
            standards_dir=base_dir,  # Base directory
            standards_night="2025-10-06"  # Subdirectory with the standard files
        )
        print(f"✓ Data manager initialized")
        print(f"  Andromeda input dir: {input_dir}")
        print(f"  Standards dir: {andromeda_manager.standards_dir}")
        print(f"  Output dir: {output_dir}")
        
        print(f"✓ Data manager initialized")
        print(f"  Andromeda input dir: {input_dir}")
        print(f"  Standards dir: {base_dir}")
        print(f"  Output dir: {output_dir}")
        
        results = {}
        for filter_type in ['V', 'B']:
            print(f"\n{'='*60}")
            print(f"Starting {filter_type} filter processing...")
            print(f"{'='*60}")
            result = process_filter(filter_type, andromeda_manager, output_dir)
            if result is not None:
                results[filter_type] = result
            else:
                print(f"WARNING: {filter_type} filter processing returned None!")
        
        # Create combined summary if both filters succeeded
        if len(results) == 2:
            print("\n" + "="*60)
            print("Creating combined results...")
            print("="*60)
            
            andromeda_data = andromeda_catalogue["CV1"]
            
            combined_df = pd.DataFrame([{
                "Name": andromeda_data["name"],
                "RA": andromeda_data["ra"],
                "Dec": andromeda_data["dec"],
                "E(B-V)": andromeda_data["e(b-v)"],
                "m_B": results['B']['m_B'],
                "m_B_err": results['B']['m_B_err'],
                "m_V": results['V']['m_V'],
                "m_V_err": results['V']['m_V_err'],
                "B-V": results['B']['m_B'] - results['V']['m_V'],
                "B-V_err": np.sqrt(results['B']['m_B_err']**2 + results['V']['m_V_err']**2)
            }])
            
            combined_path = andromeda_manager.save_results(
                combined_df, "andromeda_CV1_combined.csv"
            )
            
            print(f"\nCombined results saved to: {combined_path}")
            print(f"\nFinal Results:")
            print(f"  m_V = {results['V']['m_V']:.3f} ± {results['V']['m_V_err']:.3f}")
            print(f"  m_B = {results['B']['m_B']:.3f} ± {results['B']['m_B_err']:.3f}")
            print(f"  B-V = {combined_df['B-V'].values[0]:.3f} ± {combined_df['B-V_err'].values[0]:.3f}")
        
        print("\n" + "="*60)
        print("All processing complete!")
        print("="*60)

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()