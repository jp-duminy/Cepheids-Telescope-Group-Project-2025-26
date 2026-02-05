"""
Reduction Pipeline Fixes

This module provides functions to correct common issues in reduced FITS files
before attempting photometry.
"""

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from pathlib import Path
import warnings


class ReductionFixer:
    """Fix common issues in reduced FITS files"""
    
    def __init__(self, fits_path):
        self.fits_path = Path(fits_path)
        
        with fits.open(fits_path) as hdul:
            self.data = hdul[0].data.astype(float)
            self.header = hdul[0].header.copy()
        
        self.original_data = self.data.copy()
        self.fixes_applied = []
    
    def fix_nan_inf(self):
        """Replace NaN/Inf values with local median"""
        
        n_nan = np.sum(np.isnan(self.data))
        n_inf = np.sum(np.isinf(self.data))
        
        if n_nan == 0 and n_inf == 0:
            return
        
        # Identify bad pixels
        bad_mask = np.isnan(self.data) | np.isinf(self.data)
        
        # Replace with local median (3x3 box)
        from scipy.ndimage import median_filter
        filtered = median_filter(np.nan_to_num(self.data, nan=0, posinf=0, neginf=0), size=3)
        
        self.data[bad_mask] = filtered[bad_mask]
        
        self.fixes_applied.append(f"Fixed {n_nan + n_inf} NaN/Inf pixels")
    
    def fix_negative_pixels(self, method='floor'):
        """Handle negative pixels from over-subtraction
        
        Parameters
        ----------
        method : str
            'floor' - Set to small positive value
            'median' - Replace with local median
            'stats' - Add offset based on global statistics
        """
        
        n_negative = np.sum(self.data < 0)
        
        if n_negative == 0:
            return
        
        if method == 'floor':
            # Set to 0.1 (avoids issues with log)
            self.data[self.data < 0] = 0.1
            self.fixes_applied.append(f"Set {n_negative} negative pixels to 0.1")
        
        elif method == 'median':
            # Replace with local median
            from scipy.ndimage import median_filter
            filtered = median_filter(self.data, size=5)
            
            negative_mask = self.data < 0
            self.data[negative_mask] = filtered[negative_mask]
            self.fixes_applied.append(f"Replaced {n_negative} negative pixels with median")
        
        elif method == 'stats':
            # Add offset to make minimum = read noise level
            read_noise = self.header.get('RDNOISE', self.header.get('TOTRN', 5.0))
            exptime = self.header.get('EXPTIME', self.header.get('MEANEXP', 1.0))
            
            offset = abs(np.min(self.data)) + read_noise / exptime
            self.data += offset
            
            self.fixes_applied.append(
                f"Added offset of {offset:.2f} to fix negative pixels"
            )
            self.header['BGOFFSET'] = (offset, 'Added offset to fix negative pixels')
    
    def improve_background_subtraction(self):
        """Re-estimate and subtract background more carefully"""
        
        # Use sigma-clipping to get robust background estimate
        mean, median, std = sigma_clipped_stats(self.data, sigma=3.0, maxiters=5)
        
        # If background is significantly non-zero, subtract it
        read_noise = self.header.get('RDNOISE', self.header.get('TOTRN', 5.0))
        exptime = self.header.get('EXPTIME', self.header.get('MEANEXP', 1.0))
        expected_bg = 0
        tolerance = 3 * read_noise / exptime
        
        if abs(median) > tolerance:
            self.data -= median
            self.fixes_applied.append(
                f"Re-subtracted background: median={median:.3f} cts/s"
            )
            self.header['BGRESUB'] = (median, 'Re-subtracted background value')
    
    def cosmic_ray_clean(self, threshold=5.0):
        """Clean cosmic rays using sigma clipping
        
        Parameters
        ----------
        threshold : float
            Sigma threshold for cosmic ray detection
        """
        
        from astropy.convolution import Gaussian2DKernel, convolve
        from scipy.ndimage import median_filter
        
        # Smooth image to estimate "clean" version
        kernel = Gaussian2DKernel(x_stddev=2)
        smoothed = convolve(self.data, kernel)
        
        # Find outliers (cosmic rays)
        residual = self.data - smoothed
        mean, median, std = sigma_clipped_stats(residual, sigma=3.0)
        
        cr_mask = residual > (median + threshold * std)
        n_cr = np.sum(cr_mask)
        
        if n_cr > 0:
            # Replace cosmic rays with local median
            filtered = median_filter(self.data, size=3)
            self.data[cr_mask] = filtered[cr_mask]
            
            self.fixes_applied.append(f"Cleaned {n_cr} cosmic ray pixels")
    
    def clip_saturated(self, saturation_level=None):
        """Flag or fix saturated pixels
        
        Parameters
        ----------
        saturation_level : float, optional
            Saturation level in ADU. If None, estimated from header.
        """
        
        if saturation_level is None:
            gain = self.header.get('GAIN', self.header.get('MEANGAIN', 1.0))
            saturation_level = 60000 / gain  # Typical CCD full well
        
        n_saturated = np.sum(self.data > saturation_level)
        
        if n_saturated > 0:
            self.fixes_applied.append(
                f"WARNING: {n_saturated} saturated pixels (>{saturation_level:.0f})"
            )
            # Add flag to header
            self.header['SATURATE'] = (n_saturated, 'Number of saturated pixels')
    
    def add_noise_model(self):
        """Add proper noise statistics to header for error propagation"""
        
        # Get existing noise parameters
        gain = self.header.get('GAIN', self.header.get('MEANGAIN', None))
        read_noise = self.header.get('RDNOISE', self.header.get('TOTRN', None))
        exptime = self.header.get('EXPTIME', self.header.get('MEANEXP', None))
        nstack = self.header.get('NSTACK', 1)
        
        if gain is None:
            self.fixes_applied.append("WARNING: No GAIN in header. Using 1.0 e-/ADU")
            gain = 1.0
            self.header['GAIN'] = (gain, 'Assumed gain (e-/ADU)')
        
        if read_noise is None:
            self.fixes_applied.append("WARNING: No RDNOISE in header. Using 5.0 e-")
            read_noise = 5.0
            self.header['RDNOISE'] = (read_noise, 'Assumed read noise (e-)')
        
        if exptime is None:
            self.fixes_applied.append("WARNING: No EXPTIME in header. Using 1.0 s")
            exptime = 1.0
            self.header['EXPTIME'] = (exptime, 'Assumed exposure time (s)')
        
        # For stacked images, effective read noise decreases
        if nstack > 1:
            effective_rn = read_noise / np.sqrt(nstack)
            self.header['EFFRDNOI'] = (effective_rn, 'Effective read noise after stacking (e-)')
    
    def normalize_to_counts_per_second(self):
        """Ensure data is in counts per second"""
        
        exptime = self.header.get('EXPTIME', self.header.get('MEANEXP', None))
        
        if exptime is None:
            self.fixes_applied.append("WARNING: Cannot normalize - no EXPTIME")
            return
        
        # Check if already normalized
        if 'BUNIT' in self.header:
            if 'per' in self.header['BUNIT'].lower() or '/s' in self.header['BUNIT']:
                return  # Already normalized
        
        # Check data magnitude to guess if already normalized
        median_value = np.median(self.data)
        if median_value < 10:  # Likely already in cts/s
            self.header['BUNIT'] = 'counts/s'
            return
        
        # Normalize
        self.data = self.data / exptime
        self.header['BUNIT'] = 'counts/s'
        self.fixes_applied.append(f"Normalized to counts/s (exptime={exptime:.1f}s)")
    
    def apply_all_fixes(self, save_fixed=False, output_dir=None):
        """Apply all fixes to the data
        
        Parameters
        ----------
        save_fixed : bool
            If True, save fixed file
        output_dir : str or Path
            Directory to save fixed files
        """
        
        print(f"\nProcessing: {self.fits_path.name}")
        
        # Apply fixes in order
        self.fix_nan_inf()
        self.improve_background_subtraction()
        self.fix_negative_pixels(method='floor')  # Do after background sub
        self.cosmic_ray_clean(threshold=5.0)
        self.clip_saturated()
        self.add_noise_model()
        self.normalize_to_counts_per_second()
        
        # Report fixes
        if self.fixes_applied:
            print("  Fixes applied:")
            for fix in self.fixes_applied:
                print(f"    • {fix}")
        else:
            print("  No fixes needed")
        
        # Save if requested
        if save_fixed:
            if output_dir is None:
                output_dir = self.fits_path.parent / "fixed"
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / self.fits_path.name
            
            # Add history to header
            self.header['HISTORY'] = 'Fixed by ReductionFixer'
            for fix in self.fixes_applied:
                self.header['HISTORY'] = fix
            
            # Save
            hdu = fits.PrimaryHDU(data=self.data, header=self.header)
            hdu.writeto(output_path, overwrite=True)
            
            print(f"  Saved fixed file: {output_path}")
            
            return output_path
        
        return None


def batch_fix_reduction(data_dir, output_dir=None, apply_fixes=True):
    """Fix all FITS files in directory tree
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing reduced FITS files
    output_dir : str or Path, optional
        Directory to save fixed files. If None, creates 'fixed' subdirectories.
    apply_fixes : bool
        If True, save fixed files. If False, just report what would be fixed.
    """
    
    data_dir = Path(data_dir)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    fits_files = list(data_dir.rglob("*.fits"))
    
    print("="*60)
    print(f"FIXING {len(fits_files)} FITS FILES")
    print("="*60)
    
    fixed_paths = []
    
    for i, fits_path in enumerate(fits_files, 1):
        print(f"\n[{i}/{len(fits_files)}]", end=" ")
        
        try:
            fixer = ReductionFixer(fits_path)
            
            if apply_fixes:
                # Determine output directory
                if output_dir is None:
                    file_output_dir = fits_path.parent / "fixed"
                else:
                    # Preserve directory structure
                    rel_path = fits_path.parent.relative_to(data_dir)
                    file_output_dir = output_dir / rel_path
                
                fixed_path = fixer.apply_all_fixes(save_fixed=True, output_dir=file_output_dir)
                fixed_paths.append(fixed_path)
            else:
                fixer.apply_all_fixes(save_fixed=False)
        
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\n" + "="*60)
    print(f"COMPLETE: {len(fixed_paths)} files fixed")
    print("="*60)
    
    return fixed_paths


if __name__ == "__main__":
    
    # Configuration
    DATA_DIR = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_test"
    OUTPUT_DIR = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_test_fixed"
    
    # Run batch fixing
    print("REDUCTION PIPELINE - APPLYING FIXES")
    
    fixed_files = batch_fix_reduction(
        DATA_DIR,
        OUTPUT_DIR,
        apply_fixes=True
    )
    
    print(f"\nFixed files saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Run quality_control.py on fixed files to verify improvement")
    print("2. Update photometry_pipeline_clean.py to use fixed files")
    print("3. Re-run photometry pipeline")
