"""
Quality Control Module for Cepheid Reduction Pipeline

This module provides diagnostic tools to identify and flag problematic
reduced FITS files before attempting photometry.
"""

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, mad_std
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class FITSQualityControl:
    """Quality control checks for reduced FITS files"""
    
    def __init__(self, fits_path):
        self.fits_path = Path(fits_path)
        self.filename = self.fits_path.name
        
        with fits.open(fits_path) as hdul:
            self.data = hdul[0].data.astype(float)
            self.header = hdul[0].header
        
        # Normalize to counts/sec if EXPTIME exists
        self.exptime = self.header.get('EXPTIME', self.header.get('MEANEXP', 1.0))
        self.data_normalized = self.data / self.exptime
        
        self.issues = []
        self.warnings = []
        self.quality_metrics = {}
    
    def check_data_integrity(self):
        """Check for basic data integrity issues"""
        
        # Check for NaNs/Infs
        n_nan = np.sum(np.isnan(self.data))
        n_inf = np.sum(np.isinf(self.data))
        
        if n_nan > 0:
            self.issues.append(f"Contains {n_nan} NaN pixels")
        if n_inf > 0:
            self.issues.append(f"Contains {n_inf} Inf pixels")
        
        # Check for negative values
        n_negative = np.sum(self.data < 0)
        pct_negative = 100 * n_negative / self.data.size
        
        if pct_negative > 1.0:
            self.issues.append(f"{pct_negative:.1f}% negative pixels (poor background subtraction?)")
        elif pct_negative > 0.1:
            self.warnings.append(f"{pct_negative:.2f}% negative pixels")
        
        self.quality_metrics['n_nan'] = n_nan
        self.quality_metrics['n_inf'] = n_inf
        self.quality_metrics['pct_negative'] = pct_negative
        
        return len(self.issues) == 0
    
    def check_background_statistics(self):
        """Check if background subtraction was reasonable"""
        
        # Use sigma-clipped stats to estimate background
        mean, median, std = sigma_clipped_stats(self.data_normalized, sigma=3.0)
        
        # Background should be ~0 after proper subtraction
        # Allow some tolerance based on read noise
        read_noise = self.header.get('RDNOISE', self.header.get('TOTRN', 5.0))
        expected_std = read_noise / self.exptime
        
        if abs(median) > 3 * expected_std:
            self.warnings.append(
                f"Background median = {median:.2f} cts/s (expected ~0). "
                f"Background subtraction may be poor."
            )
        
        if std > 5 * expected_std:
            self.warnings.append(
                f"Background std = {std:.2f} cts/s (expected ~{expected_std:.2f}). "
                f"High noise or contamination."
            )
        
        self.quality_metrics['bg_mean'] = mean
        self.quality_metrics['bg_median'] = median
        self.quality_metrics['bg_std'] = std
        self.quality_metrics['expected_std'] = expected_std
        
        return True
    
    def check_saturation(self):
        """Check for saturated pixels"""
        
        # Get gain and saturation level
        gain = self.header.get('GAIN', self.header.get('MEANGAIN', 1.0))
        
        # Typical CCD full well ~65000 e-, saturation ~60000 e-
        saturation_level = 60000 / gain  # in ADU
        
        # Count pixels near saturation (>90% of saturation)
        n_saturated = np.sum(self.data > 0.9 * saturation_level)
        pct_saturated = 100 * n_saturated / self.data.size
        
        if pct_saturated > 0.1:
            self.warnings.append(
                f"{pct_saturated:.2f}% saturated pixels. "
                f"Cepheid may be too bright for accurate photometry."
            )
        
        self.quality_metrics['n_saturated'] = n_saturated
        self.quality_metrics['saturation_level'] = saturation_level
        
        return True
    
    def check_cosmic_rays(self):
        """Estimate cosmic ray contamination"""
        
        # Look for pixels significantly above background
        mean, median, std = sigma_clipped_stats(self.data_normalized, sigma=3.0)
        
        # Pixels > 10-sigma are likely cosmic rays or hot pixels
        threshold = median + 10 * std
        n_cr = np.sum(self.data_normalized > threshold)
        pct_cr = 100 * n_cr / self.data.size
        
        if pct_cr > 0.01:
            self.warnings.append(
                f"{pct_cr:.3f}% cosmic ray/hot pixels. "
                f"May affect aperture photometry."
            )
        
        self.quality_metrics['n_cosmic_rays'] = n_cr
        self.quality_metrics['pct_cosmic_rays'] = pct_cr
        
        return True
    
    def check_stacking_quality(self):
        """Check quality of stacked image (if applicable)"""
        
        nstack = self.header.get('NSTACK', 1)
        
        if nstack > 1:
            # Check if stacking header keywords are consistent
            keys_to_check = ['MEANEXP', 'MEANGAIN', 'TOTRN']
            missing = [k for k in keys_to_check if k not in self.header]
            
            if missing:
                self.warnings.append(
                    f"Stacked image missing keywords: {missing}. "
                    f"Error propagation may be incorrect."
                )
        
        self.quality_metrics['nstack'] = nstack
        
        return True
    
    def check_wcs(self):
        """Check World Coordinate System validity"""
        
        from astropy.wcs import WCS
        
        try:
            wcs = WCS(self.header)
            
            # Check if WCS is valid
            if not wcs.is_celestial:
                self.warnings.append("WCS is not celestial. Coordinate matching may fail.")
            
            # Check pixel scale
            try:
                pixel_scale = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
                
                # Typical telescope pixel scales are 0.1-5 arcsec/pixel
                if pixel_scale < 0.05 or pixel_scale > 10:
                    self.warnings.append(
                        f"Unusual pixel scale: {pixel_scale:.2f} arcsec/pixel. "
                        f"WCS may be incorrect."
                    )
                
                self.quality_metrics['pixel_scale'] = pixel_scale
            except:
                self.warnings.append("Cannot determine pixel scale from WCS")
        
        except Exception as e:
            self.warnings.append(f"WCS error: {e}")
        
        return True
    
    def check_image_dimensions(self):
        """Check if image dimensions are reasonable"""
        
        ny, nx = self.data.shape
        
        # Most CCDs are 1k-4k on a side
        if nx < 100 or ny < 100:
            self.issues.append(f"Image too small: {nx}×{ny}. May be a cutout or corrupt.")
        
        if nx > 10000 or ny > 10000:
            self.warnings.append(f"Very large image: {nx}×{ny}. Processing may be slow.")
        
        self.quality_metrics['nx'] = nx
        self.quality_metrics['ny'] = ny
        
        return True
    
    def check_snr_estimate(self):
        """Estimate rough SNR for brightest source"""
        
        # Find brightest pixels (likely a star)
        mean, median, std = sigma_clipped_stats(self.data_normalized, sigma=3.0)
        
        # Get 99th percentile (bright stars)
        bright_level = np.percentile(self.data_normalized, 99)
        signal = bright_level - median
        
        # Rough SNR estimate (ignoring source photon noise)
        snr_estimate = signal / std if std > 0 else 0
        
        if snr_estimate < 10:
            self.warnings.append(
                f"Low SNR estimate: {snr_estimate:.1f}. "
                f"Image may be too faint or noisy."
            )
        
        self.quality_metrics['snr_estimate'] = snr_estimate
        self.quality_metrics['brightest_level'] = bright_level
        
        return True
    
    def run_all_checks(self):
        """Run all quality control checks"""
        
        checks = [
            self.check_data_integrity,
            self.check_background_statistics,
            self.check_saturation,
            self.check_cosmic_rays,
            self.check_stacking_quality,
            self.check_wcs,
            self.check_image_dimensions,
            self.check_snr_estimate,
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                self.warnings.append(f"Check failed: {check.__name__}: {e}")
        
        # Determine overall quality
        if len(self.issues) > 0:
            quality = "FAIL"
        elif len(self.warnings) > 2:
            quality = "POOR"
        elif len(self.warnings) > 0:
            quality = "MARGINAL"
        else:
            quality = "GOOD"
        
        self.quality_metrics['quality'] = quality
        
        return quality
    
    def print_report(self):
        """Print quality control report"""
        
        print(f"\n{'='*60}")
        print(f"Quality Control Report: {self.filename}")
        print(f"{'='*60}")
        
        print(f"\nOverall Quality: {self.quality_metrics.get('quality', 'UNKNOWN')}")
        
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   • {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        print(f"\n📊 KEY METRICS:")
        print(f"   Dimensions: {self.quality_metrics.get('nx', '?')} × "
              f"{self.quality_metrics.get('ny', '?')}")
        print(f"   Exposure time: {self.exptime:.1f} s")
        print(f"   N stacked: {self.quality_metrics.get('nstack', 1)}")
        print(f"   Background median: {self.quality_metrics.get('bg_median', 0):.3f} cts/s")
        print(f"   Background std: {self.quality_metrics.get('bg_std', 0):.3f} cts/s")
        print(f"   Negative pixels: {self.quality_metrics.get('pct_negative', 0):.2f}%")
        print(f"   Saturated pixels: {self.quality_metrics.get('n_saturated', 0)}")
        print(f"   SNR estimate: {self.quality_metrics.get('snr_estimate', 0):.1f}")
        
        print(f"\n{'='*60}\n")


def batch_quality_control(data_dir, output_csv="qc_report.csv"):
    """Run quality control on all FITS files in directory tree"""
    
    data_dir = Path(data_dir)
    fits_files = list(data_dir.rglob("*.fits"))
    
    print(f"Found {len(fits_files)} FITS files to check...\n")
    
    results = []
    
    for i, fits_path in enumerate(fits_files, 1):
        print(f"[{i}/{len(fits_files)}] Checking {fits_path.name}...", end=" ")
        
        try:
            qc = FITSQualityControl(fits_path)
            quality = qc.run_all_checks()
            
            print(f"{quality}")
            
            # Compile results
            result = {
                'filename': fits_path.name,
                'directory': fits_path.parent.name,
                'quality': quality,
                'n_issues': len(qc.issues),
                'n_warnings': len(qc.warnings),
            }
            result.update(qc.quality_metrics)
            results.append(result)
            
            # Print full report for failed images
            if quality == "FAIL":
                qc.print_report()
        
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'filename': fits_path.name,
                'directory': fits_path.parent.name,
                'quality': 'ERROR',
                'error': str(e)
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(df['quality'].value_counts())
    print(f"\nDetailed report saved to: {output_csv}")
    
    return df


def identify_problematic_images(qc_df):
    """Identify images that should be excluded from photometry"""
    
    # Images to exclude
    exclude = qc_df[
        (qc_df['quality'] == 'FAIL') |
        (qc_df['n_issues'] > 0) |
        (qc_df['pct_negative'] > 5.0) |
        (qc_df['snr_estimate'] < 5.0)
    ]
    
    print(f"\n{'='*60}")
    print(f"IMAGES TO EXCLUDE: {len(exclude)}/{len(qc_df)}")
    print(f"{'='*60}")
    
    if len(exclude) > 0:
        print("\nReasons for exclusion:")
        for _, row in exclude.iterrows():
            print(f"\n{row['directory']}/{row['filename']}:")
            if row['n_issues'] > 0:
                print(f"  • {row['n_issues']} critical issues")
            if row.get('pct_negative', 0) > 5.0:
                print(f"  • {row['pct_negative']:.1f}% negative pixels")
            if row.get('snr_estimate', 100) < 5.0:
                print(f"  • Low SNR: {row['snr_estimate']:.1f}")
    
    return exclude


if __name__ == "__main__":
    
    # Configuration
    DATA_DIR = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_test"
    OUTPUT_CSV = "cepheid_qc_report.csv"
    
    # Run batch QC
    print("="*60)
    print("CEPHEID REDUCTION PIPELINE - QUALITY CONTROL")
    print("="*60)
    
    qc_results = batch_quality_control(DATA_DIR, OUTPUT_CSV)
    
    # Identify problematic images
    exclude_list = identify_problematic_images(qc_results)
    
    # Save exclusion list
    if len(exclude_list) > 0:
        exclude_list[['directory', 'filename']].to_csv(
            "exclude_from_photometry.csv", 
            index=False
        )
        print(f"\nExclusion list saved to: exclude_from_photometry.csv")
