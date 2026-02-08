"""
Clean photometry pipeline for Cepheid variables.

Key principles:
1. One FITS file = one Cepheid (identified by filename)
2. Extract Cepheid ID from filename
3. Look up coordinates for that specific Cepheid
4. Do photometry on that one star
5. Save results

No nested madness, no looping through all Cepheids for every image.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
import re

from Cepheid_apertures import AperturePhotometry
from Cepheid_apertures import Airmass
import AirmassInfo


class CepheidPhotometryPipeline:
    """Clean pipeline: filename → Cepheid ID → coordinates → photometry → results"""
    
    def __init__(self, data_dir, output_dir, cepheid_catalog):
        """
        Args:
            data_dir: Path to reduced FITS files (organized by date)
            output_dir: Where to save photometry results
            cepheid_catalog: Dict mapping Cepheid IDs to SkyCoord objects
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.cepheid_catalog = cepheid_catalog
        
        # Pattern to extract Cepheid number from filename
        self.cepheid_pattern = re.compile(r'cepheid[_-]?(\d+)', re.IGNORECASE)
    
    def extract_cepheid_id(self, filename):
        """Extract Cepheid ID from filename (e.g., 'cepheid_03' → '03')"""
        match = self.cepheid_pattern.search(filename)
        if match:
            return f"{int(match.group(1)):02d}"  # Always 2 digits: 01, 02, etc.
        return None
    
    def get_date_directories(self):
        """Find all date-based subdirectories (YYYY-MM-DD format)"""
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
        return sorted([d for d in self.data_dir.iterdir() 
                      if d.is_dir() and date_pattern.match(d.name)])
    
    def process_single_image(self, fits_path, cepheid_id, coord):
        """
        Do aperture photometry on a single FITS file for one Cepheid.
        
        Returns:
            dict with photometry results, or None if failed
        """
        try:
            # Load image
            ap = AperturePhotometry(fits_path)
            
            # Get metadata
            exptime = ap.header.get("EXPTIME", ap.header.get("MEANEXP", 1.0))
            gain = ap.header.get("GAIN", ap.header.get("MEANGAIN", 1.0))
            read_noise = ap.header.get("RDNOISE", ap.header.get("TOTRN", 5.0))
            stack_size = ap.header.get("NSTACK", 1)
            time_obs = ap.header.get("DATE-OBS")
            

            if time_obs is None:
                raise ValueError("No DATE-OBS in header")
            
            # Find star position in image
            x_guess, y_guess = ap.get_pixel_coords(coord.ra.deg, coord.dec.deg)
            
            # Check if star is actually on the detector
            ny, nx = ap.data.shape
            if not (0 <= x_guess < nx and 0 <= y_guess < ny):
                print(f"  ⚠️  Cepheid {cepheid_id} not in field of view")
                return None
            
            # Create cutout around star
            cutout, x_offset, y_offset = self.extract_cutout(
                ap.data, x_guess, y_guess, half_size=25
            )
            
            if cutout is None:
                return None
            
            # Measure centroid and FWHM in cutout
            centroid_local, fwhm = ap.get_centroid_and_fwhm(cutout)
            
            # Convert back to full-image coordinates
            centroid_global = (
                centroid_local[0] + x_offset,
                centroid_local[1] + y_offset
            )
            
            # Aperture photometry
            ap_radius = 2.0 * fwhm
            target_flux, ap_area, sky_per_pix, ann_area = ap.aperture__photometry(
                data=ap.data,
                centroid=centroid_global,
                ap_rad=ap_radius,
                ceph_name=f"cepheid_{cepheid_id}",
                date=fits_path.parent.name,
                plot=True
            )
            
            # Calculate magnitude and error
            magnitude = ap.instrumental_magnitude(target_flux)
            mag_error = ap.get_inst_mag_error(
                target_counts=target_flux,
                aperture_area=ap_area,
                sky_counts=sky_per_pix,
                sky_ann_area=ann_area,
                gain=gain,
                exp_time=exptime,
                read_noise=read_noise,
                stack_size=stack_size
            )
            
            return {
                "name": f"cepheid_{cepheid_id}",
                "time": time_obs,
                "magnitude": magnitude,
                "magnitude_error": mag_error,
                "fwhm": fwhm,
                "x_pixel": centroid_global[0],
                "y_pixel": centroid_global[1]
            }
            
        except Exception as e:
            print(f"  ❌ Error processing {fits_path.name}: {e}")
            return None
    
    def extract_cutout(self, data, x_center, y_center, half_size=25):
        """
        Extract a cutout around a position, ensuring odd dimensions.
        
        Returns:
            (cutout_array, x_offset, y_offset) or (None, None, None)
        """
        ny, nx = data.shape
        
        x0 = int(np.round(x_center))
        y0 = int(np.round(y_center))
        
        # Define cutout boundaries
        x1 = max(0, x0 - half_size)
        x2 = min(nx, x0 + half_size)
        y1 = max(0, y0 - half_size)
        y2 = min(ny, y0 + half_size)
        
        cutout = data[y1:y2, x1:x2]
        
        # Check minimum size
        if cutout.shape[0] < 7 or cutout.shape[1] < 7:
            return None, None, None
        
        # Force odd dimensions (for centroid algorithms)
        if cutout.shape[0] % 2 == 0:
            cutout = cutout[:-1, :]
        if cutout.shape[1] % 2 == 0:
            cutout = cutout[:, :-1]
        
        return cutout, x1, y1
    
    def run(self):
        """Main pipeline: process all dates and files"""
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []
        
        date_dirs = self.get_date_directories()
        print(f"Found {len(date_dirs)} observation dates\n")
        
        for date_dir in date_dirs:
            date = date_dir.name
            print(f"📅 Processing {date}")
            
            date_output_dir = self.output_dir / date
            date_output_dir.mkdir(parents=True, exist_ok=True)
            
            night_results = []
            
            # Process each FITS file in this night
            fits_files = sorted(date_dir.glob("*.fits"))
            print(f"   Found {len(fits_files)} FITS files")
            
            for fits_path in fits_files:
                # Identify which Cepheid this file is for
                cepheid_id = self.extract_cepheid_id(fits_path.name)
                
                if cepheid_id is None:
                    print(f"  ⚠️  Skipping {fits_path.name}: can't identify Cepheid")
                    continue
                
                if cepheid_id not in self.cepheid_catalog:
                    print(f"  ⚠️  Skipping {fits_path.name}: Cepheid {cepheid_id} not in catalog")
                    continue
                
                # Get coordinates for THIS specific Cepheid
                coord = self.cepheid_catalog[cepheid_id]
                
                print(f"  → {fits_path.name} (Cepheid {cepheid_id})")
                
                # Do photometry
                result = self.process_single_image(fits_path, cepheid_id, coord)
                
                if result is not None:
                    night_results.append(result)
                    all_results.append(result)
                    print(f"    ✓ mag = {result['magnitude']:.3f} ± {result['magnitude_error']:.3f}")
            
            # Save this night's results
            
            if night_results:
                df = pd.DataFrame(night_results)
                df = df.sort_values(by=["name", "time"]).reset_index(drop=True)
                csv_path = date_output_dir / f"{date}_photometrv3.csv"
                df.to_csv(csv_path, index=False)
                print(f"   💾 Saved {len(night_results)} measurements → {csv_path}\n")
        
        # Save combined results
        if all_results:
            df_all = pd.DataFrame(all_results)
            df_all = df_all.sort_values(by=["name", "time"]).reset_index(drop=True)
            combined_path = self.output_dir / "all_nights_photometry.csv"
            df_all.to_csv(combined_path, index=False)
            print(f"\n✅ Pipeline complete!")
            print(f"   Total measurements: {len(all_results)}")
            print(f"   Combined CSV: {combined_path}")
        else:
            print("\n⚠️  No successful measurements!")


def build_cepheid_catalog():
    """Build catalog of Cepheid coordinates"""
    
    catalog_data = {
        "Ceph_num": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"],
        "RA": ["20 12 22.83", "20 54 57.53", "20 57 20.83", "21 04 16.63", 
               "21 57 52.69", "22 40 52.15", "22 48 38.00", "23 07 10.08", 
               "00 26 19.45", "00 29 58.59", "01 32 43.22"],
        "Dec": ["+32 52 17.8", "+47 32 01.7", "+40 10 39.1", "+39 58 20.1", 
                "+56 09 50.0", "+56 49 46.1", "+56 19 17.5", "+58 33 15.1", 
                "+51 16 49.3", "+60 12 43.1", "+63 35 37.7"]
    }
    
    catalog = {}
    for num, ra_str, dec_str in zip(
        catalog_data["Ceph_num"],
        catalog_data["RA"],
        catalog_data["Dec"]
    ):
        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")
        catalog[num] = coord
        print(f"Cepheid {num}: RA={coord.ra.deg:.4f}°, Dec={coord.dec.deg:.4f}°")
    
    print()
    return catalog

class StandardStarPhotometryPipeline:
    """Clean pipeline: filename → Standard Star ID → coordinates → photometry → results"""
    
    def __init__(self, data_dir, output_dir, standard_catalog):
        """
        Args:
            data_dir: Path to reduced FITS files (organized by date)
            output_dir: Where to save photometry results
            standard_catalog: Dict mapping standard star IDs to SkyCoord objects
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.standard_catalog = standard_catalog
        
        # Pattern to extract Standard Star number from filename
        self.standard_pattern = re.compile(r'standard[_-]?(\d+)', re.IGNORECASE)
    
    def extract_standard_id(self, filename):
        """Extract Standard Star ID from filename (e.g., 'standard_03' → '03')"""

        #match based on standard stars in the catalog (e.g., "SA_114_176" → "114176")
        for standard_id in self.standard_catalog.keys():
            if standard_id in filename:
                return standard_id
        return None
    
    def get_date_directories(self):
        """Find all date-based subdirectories (YYYY-MM-DD format)"""
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
        return sorted([d for d in self.data_dir.iterdir() 
                      if d.is_dir() and date_pattern.match(d.name)])
    
    def process_single_image(self, fits_path, standard_id, coord):
        """
        Do aperture photometry on a single FITS file for one Standard Star.
        
        Returns:
            dict with photometry results, or None if failed
        """
        try:
            # Load image
            ap = AperturePhotometry(fits_path)
            
            # Get metadata
            exptime = ap.header.get("EXPTIME", ap.header.get("MEANEXP", 1.0))
            gain = ap.header.get("GAIN", ap.header.get("MEANGAIN", 1.0))
            read_noise = ap.header.get("RDNOISE", ap.header.get("TOTRN", 5.0))
            stack_size = ap.header.get("NSTACK", 1)
            time_obs = ap.header.get("DATE-OBS")

            block = AirmassInfo().process_fits(fits_path)
            airmass = float(block.split("Airmass;")[1].strip())
            
            if time_obs is None:
                raise ValueError("No DATE-OBS in header")
            
            # Find star position in image
            x_guess, y_guess = ap.get_pixel_coords(coord.ra.deg, coord.dec.deg)
            
            # Check if star is actually on the detector
            ny, nx = ap.data.shape
            if not (0 <= x_guess < nx and 0 <= y_guess < ny):
                print(f"  ⚠️  Standard {standard_id} not in field of view")
                return None
            
            # Create cutout around star
            cutout, x_offset, y_offset = self.extract_cutout(
                ap.data, x_guess, y_guess, half_size=25
            )

            if cutout is None:                                                                                                                                                  
                return None
            
            # Measure centroid and FWHM in cutout
            centroid_local, fwhm = ap.get_centroid_and_fwhm(cutout)
            
            # Convert back to full-image coordinates
            centroid_global = (
                centroid_local[0] + x_offset,
                centroid_local[1] + y_offset
            )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
            
            # Aperture photometry
            ap_radius = 2.0 * fwhm
            target_flux, ap_area, sky_per_pix, ann_area = ap.aperture__photometry(
                data=ap.data,
                centroid=centroid_global,
                ap_rad=ap_radius,
                ceph_name=f"standard_{standard_id}",
                date=fits_path.parent.name,
                plot=False
            )
            
            # Calculate magnitude and error
            magnitude = ap.instrumental_magnitude(target_flux)
            mag_error = ap.get_inst_mag_error(
                target_counts=target_flux,
                aperture_area=ap_area,
                sky_counts=sky_per_pix,
                sky_ann_area=ann_area,
                gain=gain,
                exp_time=exptime,
                read_noise=read_noise,
                stack_size=stack_size
            )
            
            return {
                "name": f"standard_{standard_id}",
                "time": time_obs,
                "magnitude": magnitude,
                "magnitude_error": mag_error,
                "fwhm": fwhm,
                "x_pixel": centroid_global[0],
                "y_pixel": centroid_global[1],
                "airmass": airmass
            }
            
        except Exception as e:
            print(f"  ❌ Error processing {fits_path.name}: {e}")
            return None
    
    def extract_cutout(self, data, x_center, y_center, half_size=25):
        """
        Extract a cutout around a position, ensuring odd dimensions.
        
        Returns:
            (cutout_array, x_offset, y_offset) or (None, None, None)
        """
        ny, nx = data.shape
        
        # Use numpy's round, then convert to int
        x0 = int(np.round(x_center))
        y0 = int(np.round(y_center))
        
        # Define cutout boundaries
        x1 = max(0, x0 - half_size)
        x2 = min(nx, x0 + half_size)
        y1 = max(0, y0 - half_size)
        y2 = min(ny, y0 + half_size)
        
        cutout = data[y1:y2, x1:x2]
        
        # Check minimum size
        if cutout.shape[0] < 7 or cutout.shape[1] < 7:
            return None, None, None
        
        # Force odd dimensions (for centroid algorithms)
        if cutout.shape[0] % 2 == 0:
            cutout = cutout[:-1, :]
        if cutout.shape[1] % 2 == 0:
            cutout = cutout[:, :-1]
        
        return cutout, x1, y1
    
    def run(self):
        """Main pipeline: process all dates and files"""
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []
        
        date_dirs = self.get_date_directories()
        print(f"Found {len(date_dirs)} observation dates\n")
        
        for date_dir in date_dirs:
            date = date_dir.name
            print(f"📅 Processing {date}")
            
            date_output_dir = self.output_dir / date
            date_output_dir.mkdir(parents=True, exist_ok=True)
            
            night_results = []
            
            # Process each FITS file in this night
            fits_files = sorted(date_dir.glob("*.fits"))
            print(f"   Found {len(fits_files)} FITS files")
            
            for fits_path in fits_files:
                # Identify which Standard this file is for
                standard_id = self.extract_standard_id(fits_path.name)
                
                if standard_id is None:
                    print(f"  ⚠️  Skipping {fits_path.name}: can't identify Standard")
                    continue
                
                if standard_id not in self.standard_catalog:
                    print(f"  ⚠️  Skipping {fits_path.name}: Standard {standard_id} not in catalog")
                    continue
                
                # Get coordinates for THIS specific Standard
                coord = self.standard_catalog[standard_id]
                
                print(f"  → {fits_path.name} (Standard {standard_id})")
                
                # Do photometry
                result = self.process_single_image(fits_path, standard_id, coord)
                
                if result is not None:
                    night_results.append(result)
                    all_results.append(result)
                    print(f"    ✓ mag = {result['magnitude']:.3f} ± {result['magnitude_error']:.3f}")
            
            # Save this night's results
            if night_results:
                df = pd.DataFrame(night_results)
                df = df.sort_values(by=["name", "time"]).reset_index(drop=True)
                csv_path = date_output_dir / f"{date}_photometry.csv"
                df.to_csv(csv_path, index=False)
                print(f"   💾 Saved {len(night_results)} measurements → {csv_path}\n")
        
        # Save combined results
        if all_results:
            df_all = pd.DataFrame(all_results)
            df_all = df_all.sort_values(by=["name", "time"]).reset_index(drop=True)
            combined_path = self.output_dir / "all_nights_photometry.csv"
            df_all.to_csv(combined_path, index=False)
            print(f"\n✅ Pipeline complete!")
            print(f"   Total measurements: {len(all_results)}")
            print(f"   Combined CSV: {combined_path}")
        else:
            print("\n⚠️  No successful measurements!")

        #find airmass correction using standard stars
        #initialize Airmass class with arrays of airmass, Vmag, m_inst, and m_err for all standard stars
        airmass = df_all["airmass"].values
        Vmag = df_all["magnitude"].values
        m_inst = df_all["magnitude"].values #This is a placeholder. You need to calculate the instrumental magnitude for each standard star based on the measured flux and the known zero-point.
        m_err = df_all["magnitude_error"].values
        airmass_correction = Airmass(airmass, Vmag, m_inst, m_err)
        k, k_err, Z1, Z1_err = airmass_correction.fit_extinction_weighted()
        print(f"Recovered extinction coefficient k = {k:.3f} ± {k_err:.3f}")
        print(f"Recovered zero-point (X=1) = {Z1:.2f} ± {Z1_err:.2f}")
        
        #plot graph of Vmag - m_inst vs airmass with error bars and best-fit line using airmass class and also plot parameter space of k and Z1 with their uncertainties
        airmass_correction.plot_atmospheric_extinction(airmass, Vmag, m_inst, m_err)
        airmass_correction.plot_parameter_space(k, k_err, Z1, Z1_err)


def build_standard_catalog():
    """Build catalog of Standard coordinates"""
    
    catalog_data = {
        "Standard_num": ["114176", "SA111775", "F_108", "SA112_595", "GD_246", "G93_48", "G156_31"],
        "RA": ["22 43 11", "19 37 17", "23 16 12", "20 41 19", 
               "23 12 21.6", "21 52 25.4", "22 38 28"],
        "Dec": ["+00 21 16.0", "+00 11 14.0", "-01 50 35.0", "+00 16 11.0", 
                "+10 47 04", " +02 23 23", "-15 19 17"]
    }
    
    catalog = {}
    for num, ra_str, dec_str in zip(
        catalog_data["Standard_num"],
        catalog_data["RA"],
        catalog_data["Dec"]
    ):
        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")
        catalog[num] = coord
        print(f"Standard {num}: RA={coord.ra.deg:.4f}°, Dec={coord.dec.deg:.4f}°")
    
    print()
    return catalog

if __name__ == "__main__":
    
    # Configuration
    DATA_DIR = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_standard_stars_V"
    OUTPUT_DIR = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_standard_stars_V_Photometry"
    
    # Build catalog
    print("=" * 60)
    print("Building Standard catalog...")
    print("=" * 60)
    standard_catalog = build_standard_catalog()
    
    # Run pipeline
    print("=" * 60)
    print("Starting photometry pipeline...")
    print("=" * 60)
    pipeline = StandardStarPhotometryPipeline(DATA_DIR, OUTPUT_DIR, standard_catalog)
    pipeline.run()

    # Configuration
    DATA_DIR = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_test"
    OUTPUT_DIR = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_P"
    
    # Build catalog
    print("=" * 60)
    print("Building Cepheid catalog...")
    print("=" * 60)
    cepheid_catalog = build_cepheid_catalog()
    
    # Run pipeline
    print("=" * 60)
    print("Starting photometry pipeline...")
    print("=" * 60)
    pipeline = CepheidPhotometryPipeline(DATA_DIR, OUTPUT_DIR, cepheid_catalog)
    pipeline.run()

   