"""
diagnostic_photometry.py

Run this to check which files need manual intervention.
Generates a CSV report of successes and failures.
"""

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
from AirmassInfo import AirmassInfo

# ============================================================================
# CATALOGUES
# ============================================================================

cepheid_catalogue = {
    "01": {"ra": "20 12 22.83", "dec": "+32 52 17.8", "e(b-v)": "0.68", "name": "MW Cyg"},
    "02": {"ra": "20 54 57.53", "dec": "+47 32 01.7", "e(b-v)": "0.80", "name": "V520 Cyg"},
    "03": {"ra": "20 57 20.83", "dec": "+40 10 39.1", "e(b-v)": "0.79", "name": "VX Cyg"},
    "04": {"ra": "21 04 16.63", "dec": "+39 58 20.1", "e(b-v)": "0.65", "name": "VY Cyg"},
    "05": {"ra": "21 57 52.69", "dec": "+56 09 50.0", "e(b-v)": "0.68", "name": "CP Cep"},
    "06": {"ra": "22 40 52.15", "dec": "+56 49 46.1", "e(b-v)": "0.40", "name": "Z Lac"},
    "07": {"ra": "22 48 38.00", "dec": "+56 19 17.5", "e(b-v)": "0.36", "name": "V Lac"},
    "08": {"ra": "23 07 10.08", "dec": "+58 33 15.1", "e(b-v)": "0.49", "name": "SW Cas"},
    "09": {"ra": "00 26 19.45", "dec": "+51 16 49.3", "e(b-v)": "0.12", "name": "TU Cas"},
    "10": {"ra": "00 29 58.59", "dec": "+60 12 43.1", "e(b-v)": "0.53", "name": "DL Cas"},
    "11": {"ra": "01 32 43.22", "dec": "+63 35 37.7", "e(b-v)": "0.70", "name": "V636 Cas"},
}

standard_catalogue = {
    "114176": {"ra": "+22 43 11.0", "dec": "+00 21 16.0", "mag": "9.239", "e(b-v)": "0.0013"},
    "SA111775": {"ra": "+19 37 17.0", "dec": "+00 11 14.0", "mag": "10.74", "e(b-v)": "0.0009"},
    "F_108": {"ra": "+23 16 12.0", "dec": "-01 50 35.0", "mag": "12.96", "e(b-v)": "0.0016"},
    "SA112_595": {"ra": "+20 41 19.0", "dec": "+00 16 11.0", "mag": "11.35", "e(b-v)": "0.0016"},
    "GD_246": {"ra": "+23 12 21.6", "dec": "+10 47 04.0", "mag": "13.09", "e(b-v)": "0.0015"},
    "G93_48": {"ra": "+21 52 25.4", "dec": "+02 23 23.0", "mag": "12.74", "e(b-v)": "0.0012"},
    "G156_31": {"ra": "+22 38 28.0", "dec": "-15 19 17.0", "mag": "12.36", "e(b-v)": "0.0049"}
}

# ============================================================================
# DIAGNOSTIC PHOTOMETRY CLASS
# ============================================================================

class DiagnosticPhotometry:
    """
    Performs photometry with detailed diagnostics.
    """
    def __init__(self, fits_path, ra, dec, name, ebv):
        self.fits_path = Path(fits_path)
        self.name = name
        self.ebv = ebv

        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
        self.ra = coord.ra.deg
        self.dec = coord.dec.deg

        self.ap = AperturePhotometry(str(fits_path))
        self.ccd_params()

    def ccd_params(self):
        """Extract CCD parameters from header."""
        with fits.open(self.fits_path) as hdul:
            hdr = hdul[0].header
            self.gain = hdr.get('MEANGAIN', None)
            self.read_noise = hdr.get('TOTRN', None)
            self.exp_time = hdr.get('TOTEXP', None)
            self.stack_size = hdr.get('NSTACK', 1)

    def raw_photometry(self, width=200, plot_diagnostics=False):
        """
        Raw photometry with automatic failure detection.
        
        Returns dict with:
            - 'status': 'SUCCESS' or 'NEEDS_MANUAL'
            - 'm_inst': instrumental magnitude (if successful)
            - 'm_err': magnitude error (if successful)
            - 'flux': sky-subtracted flux
            - 'fwhm': measured FWHM
            - Error details if failed
        """
        print(f"\n{'='*60}")
        print(f"Processing: {self.name}")
        print(f"{'='*60}")
        
        # Step 1: Try WCS to get initial position
        try:
            x_guess, y_guess = self.ap.get_pixel_coords(self.ra, self.dec)
            print(f"✓ WCS position: ({x_guess:.1f}, {y_guess:.1f})")
            
            # Check if position is within image bounds
            ny, nx = self.ap.data.shape
            if not (0 <= x_guess < nx and 0 <= y_guess < ny):
                raise ValueError(f"WCS position ({x_guess:.1f}, {y_guess:.1f}) outside image ({nx}x{ny})")
            
            wcs_success = True
            
        except Exception as e:
            print(f"❌ WCS failed: {e}")
            return {
                'status': 'NEEDS_MANUAL',
                'filename': str(self.fits_path),
                'name': self.name,
                'error': f'WCS failed: {e}'
            }
        
        # Step 2: Cut out region around star
        try:
            masked_data = self.ap.mask_data_and_plot(x_guess, y_guess, width=width, plot=False)
            print(f"✓ Cutout created: {masked_data.shape}")
            
            # Quick statistics
            print(f"  Data range: {np.min(masked_data):.1f} to {np.max(masked_data):.1f} ADU/s")
            print(f"  Median: {np.median(masked_data):.2f} ADU/s")
            
        except Exception as e:
            print(f"❌ Cutout failed: {e}")
            return {
                'status': 'NEEDS_MANUAL',
                'filename': str(self.fits_path),
                'name': self.name,
                'error': f'Cutout failed: {e}'
            }
        
        # Step 3: Find centroid with simple background subtraction
        try:
            # Use simple mean subtraction (you found this works)
            crude_sub = masked_data - np.mean(masked_data)
            crude_sub = np.maximum(crude_sub, 0)
            
            centroid, fwhm = self.ap.get_centroid_and_fwhm(crude_sub, plot=False)
            print(f"✓ Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
            print(f"✓ FWHM: {fwhm:.2f} pixels")
            
            # Diagnostic: Check centroid position relative to cutout center
            cutout_center = np.array(masked_data.shape[::-1]) / 2  # (x, y)
            centroid_offset = np.linalg.norm(np.array(centroid) - cutout_center)
            print(f"  Centroid offset from cutout center: {centroid_offset:.1f} pixels")
            
            # Validate centroid position
            if centroid_offset > width * 0.35:
                raise ValueError(f"Centroid offset too large ({centroid_offset:.1f} > {width*0.35:.1f})")
            
            # Validate FWHM
            if not (1.5 < fwhm < 20):
                raise ValueError(f"FWHM out of range: {fwhm:.2f} pixels")
            
        except Exception as e:
            print(f"❌ Centroid finding failed: {e}")
            return {
                'status': 'NEEDS_MANUAL',
                'filename': str(self.fits_path),
                'name': self.name,
                'error': f'Centroid failed: {e}'
            }
        
        # Step 4: Curve of growth to find optimal aperture
        try:
            ap_rad = self.curve_of_growth(masked_data, centroid, fwhm, inner=1.5, outer=2.0, plot=False)
            print(f"✓ Optimal aperture radius: {ap_rad:.2f} pixels ({ap_rad/fwhm:.2f} × FWHM)")
            
        except Exception as e:
            print(f"⚠️  Curve of growth failed, using 2.5×FWHM: {e}")
            ap_rad = 2.5 * fwhm
        
        # Step 5: Aperture photometry
        try:
            result = self.ap.aperture_photometry(
                masked_data, centroid, ap_rad, 
                ceph_name=self.name, date=self.fits_path.parent.name,
                inner=1.5, outer=2.0, 
                plot=plot_diagnostics, 
                savefig=False
            )
            
            flux, ap_area, sky_bckgnd, annulus_area = result
            
            # Check for None returns (indicates failure)
            if flux is None:
                raise ValueError("Aperture photometry returned None (likely negative flux or boundary issues)")
            
            print(f"✓ Flux: {flux:.1f} ADU/s")
            print(f"  Aperture area: {ap_area:.1f} pixels")
            print(f"  Sky background: {sky_bckgnd:.3f} ADU/s/pixel")
            
        except Exception as e:
            print(f"❌ Aperture photometry failed: {e}")
            return {
                'status': 'NEEDS_MANUAL',
                'filename': str(self.fits_path),
                'name': self.name,
                'error': f'Photometry failed: {e}'
            }
        
        # Step 6: Calculate instrumental magnitude and error
        try:
            instrumental_mag = self.ap.instrumental_magnitude(flux)
            instrumental_mag_error = self.ap.get_inst_mag_error(
                flux, ap_area, sky_bckgnd, annulus_area,
                self.gain, self.exp_time, self.read_noise, self.stack_size
            )
            
            print(f"✓ Instrumental magnitude: {instrumental_mag:.3f} ± {instrumental_mag_error:.3f}")
            
            # Validate magnitude error
            if instrumental_mag_error > 0.5:
                print(f"⚠️  Large magnitude error: {instrumental_mag_error:.3f}")
            
        except Exception as e:
            print(f"❌ Magnitude calculation failed: {e}")
            return {
                'status': 'NEEDS_MANUAL',
                'filename': str(self.fits_path),
                'name': self.name,
                'error': f'Magnitude calculation failed: {e}'
            }
        
        # Success!
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS: {self.name}")
        print(f"{'='*60}\n")
        
        return {
            'status': 'SUCCESS',
            'm_inst': instrumental_mag,
            'm_err': instrumental_mag_error,
            'flux': flux,
            'fwhm': fwhm,
            'centroid_offset': centroid_offset,
            'ap_radius': ap_rad
        }

    def curve_of_growth(self, data, centroid, fwhm, inner=1.5, outer=2.0, plot=False):
        """Determine optimal aperture size from curve-of-growth."""
        ap_radii = np.arange(0.1, 4.1, 0.1) * fwhm
        fluxes = []
        
        for radius in ap_radii:
            result = self.ap.aperture_photometry(
                data, centroid, ap_rad=radius, 
                inner=inner, outer=outer, plot=False
            )
            flux = result[0]
            fluxes.append(flux if flux is not None else 0)
        
        fluxes = np.array(fluxes)
        max_flux = np.max(fluxes)
        target_flux = 0.90 * max_flux
        
        idx = np.argmax(fluxes >= target_flux)
        optimal_radius = ap_radii[idx]
        
        return optimal_radius

# ============================================================================
# FILE EXTRACTION UTILITIES
# ============================================================================

def extract_cepheid_id(filename):
    """Extract cepheid number from filename."""
    match = re.search(r'cepheid[_-]?(\d+)', filename.lower())
    return f"{int(match.group(1)):02d}" if match else None

def extract_standard_id(filename):
    """Extract standard star ID from filename."""
    filename_lower = filename.lower()
    for std_id in standard_catalogue.keys():
        std_id_pattern = std_id.lower().replace("_", "[_-]?")
        if re.search(std_id_pattern, filename_lower):
            return std_id
    return None

# ============================================================================
# MAIN DIAGNOSTIC FUNCTION
# ============================================================================

def run_diagnostics(input_dir, output_dir=None, star_type='cepheid', plot_failures=False):
    """
    Run diagnostics on all files in a directory.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing reduced FITS files
    output_dir : str or Path, optional
        Where to save diagnostic reports (default: same as input_dir)
    star_type : str
        Either 'cepheid' or 'standard'
    plot_failures : bool
        If True, show diagnostic plots for failed cases
    
    Returns
    -------
    results_df : DataFrame
        Summary of all results
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    night_name = input_path.name
    
    # Find files
    if star_type == 'cepheid':
        files = sorted(input_path.glob("cepheid_*_stacked.fits"))
        catalogue = cepheid_catalogue
        extract_id = extract_cepheid_id
    else:
        files = sorted(input_path.glob("standard_*_stacked.fits"))
        catalogue = standard_catalogue
        extract_id = extract_standard_id
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC RUN: {night_name}")
    print(f"Star type: {star_type}")
    print(f"Files found: {len(files)}")
    print(f"{'='*70}\n")
    
    all_results = []
    successes = []
    failures = []
    
    for fits_file in files:
        # Get star info
        star_id = extract_id(fits_file.name)
        
        if star_id is None:
            print(f"⚠️  Could not extract ID from {fits_file.name}, skipping...")
            continue
        
        if star_id not in catalogue:
            print(f"⚠️  ID {star_id} not in catalogue, skipping...")
            continue
        
        star_data = catalogue[star_id]
        
        # Create photometry object
        phot = DiagnosticPhotometry(
            fits_path=fits_file,
            ra=star_data["ra"],
            dec=star_data["dec"],
            name=star_data.get("name", star_id),
            ebv=star_data.get("e(b-v)", "0.0")
        )
        
        # Run photometry
        result = phot.raw_photometry(width=200, plot_diagnostics=False)
        
        # Add metadata
        result['star_id'] = star_id
        result['star_type'] = star_type
        result['night'] = night_name
        result['filepath'] = str(fits_file)
        
        all_results.append(result)
        
        if result['status'] == 'SUCCESS':
            successes.append(result)
        else:
            failures.append(result)
            
            # Optionally show diagnostic plot for failures
            if plot_failures:
                print(f"\n📊 Showing diagnostic plot for {result['name']}...")
                phot.raw_photometry(width=200, plot_diagnostics=True)
    
    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_file = output_path / f"diagnostic_report_{star_type}_{night_name}.csv"
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY: {night_name}")
    print(f"{'='*70}")
    print(f"Total files processed: {len(all_results)}")
    print(f"✓ Successes: {len(successes)} ({100*len(successes)/len(all_results):.1f}%)")
    print(f"❌ Failures: {len(failures)} ({100*len(failures)/len(all_results):.1f}%)")
    
    if failures:
        print(f"\nFailed stars:")
        for result in failures:
            print(f"  - {result['name']} ({result['star_id']}): {result['error']}")
    
    print(f"\n📄 Full report saved to: {output_file}")
    
    # Save separate file with just failures for manual intervention
    if failures:
        failures_df = pd.DataFrame(failures)
        failures_file = output_path / f"needs_manual_{star_type}_{night_name}.csv"
        failures_df.to_csv(failures_file, index=False)
        print(f"📄 Manual intervention list: {failures_file}")
    
    return results_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # CONFIGURATION - EDIT THESE
    # ========================================================================
    
    # Single night test
    TEST_NIGHT = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/2025-10-06"
    
    # Multiple nights (comment out if just testing one)
    # ALL_NIGHTS = [
    #     "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/2025-09-22",
    #     "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/2025-09-24",
    #     "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/2025-10-06",
    #     # ... add more nights
    # ]
    
    OUTPUT_DIR = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Diagnostics"
    
    # ========================================================================
    
    # Run on single night (for testing)
    print("\n" + "="*70)
    print("RUNNING DIAGNOSTICS ON SINGLE NIGHT")
    print("="*70)
    
    # Cepheids
    cep_results = run_diagnostics(
        input_dir=TEST_NIGHT,
        output_dir=OUTPUT_DIR,
        star_type='cepheid',
        plot_failures=False  # Set to True to see plots of failures
    )
    
    # Standards
    std_results = run_diagnostics(
        input_dir=TEST_NIGHT,
        output_dir=OUTPUT_DIR,
        star_type='standard',
        plot_failures=False
    )
    
    # ========================================================================
    # Uncomment this section to run on multiple nights
    # ========================================================================
    
    # print("\n" + "="*70)
    # print("RUNNING DIAGNOSTICS ON ALL NIGHTS")
    # print("="*70)
    # 
    # all_cep_results = []
    # all_std_results = []
    # 
    # for night_dir in ALL_NIGHTS:
    #     print(f"\n{'='*70}")
    #     print(f"Processing night: {Path(night_dir).name}")
    #     print(f"{'='*70}")
    #     
    #     cep_df = run_diagnostics(night_dir, OUTPUT_DIR, 'cepheid', plot_failures=False)
    #     std_df = run_diagnostics(night_dir, OUTPUT_DIR, 'standard', plot_failures=False)
    #     
    #     all_cep_results.append(cep_df)
    #     all_std_results.append(std_df)
    # 
    # # Combine all nights
    # combined_cep = pd.concat(all_cep_results, ignore_index=True)
    # combined_std = pd.concat(all_std_results, ignore_index=True)
    # 
    # # Save combined results
    # combined_cep.to_csv(Path(OUTPUT_DIR) / "diagnostic_report_cepheids_ALL_NIGHTS.csv", index=False)
    # combined_std.to_csv(Path(OUTPUT_DIR) / "diagnostic_report_standards_ALL_NIGHTS.csv", index=False)
    # 
    # # Overall summary
    # print("\n\n" + "="*70)
    # print("OVERALL SUMMARY - ALL NIGHTS")
    # print("="*70)
    # 
    # print("\nCEPHEIDS:")
    # print(f"  Total: {len(combined_cep)}")
    # print(f"  Successes: {(combined_cep['status'] == 'SUCCESS').sum()}")
    # print(f"  Failures: {(combined_cep['status'] == 'NEEDS_MANUAL').sum()}")
    # 
    # print("\nSTANDARDS:")
    # print(f"  Total: {len(combined_std)}")
    # print(f"  Successes: {(combined_std['status'] == 'SUCCESS').sum()}")
    # print(f"  Failures: {(combined_std['status'] == 'NEEDS_MANUAL').sum()}")