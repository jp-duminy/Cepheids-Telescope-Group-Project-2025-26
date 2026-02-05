"""
Quick Diagnostic for Single FITS File

Use this to quickly inspect a problematic FITS file and understand
what's wrong with it.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import sys


def quick_diagnose(fits_path, show_plots=True):
    """Quick diagnostic of a FITS file"""
    
    print("\n" + "="*70)
    print(f"QUICK DIAGNOSTIC: {fits_path}")
    print("="*70)
    
    # Load data
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
    
    # Basic info
    print(f"\n📋 BASIC INFO:")
    print(f"   Dimensions: {data.shape}")
    print(f"   Data type: {data.dtype}")
    
    # Header info
    exptime = header.get('EXPTIME', header.get('MEANEXP', '???'))
    gain = header.get('GAIN', header.get('MEANGAIN', '???'))
    rdnoise = header.get('RDNOISE', header.get('TOTRN', '???'))
    nstack = header.get('NSTACK', '???')
    
    print(f"\n🔧 INSTRUMENT SETTINGS:")
    print(f"   Exposure time: {exptime} s")
    print(f"   Gain: {gain} e-/ADU")
    print(f"   Read noise: {rdnoise} e-")
    print(f"   Stacked frames: {nstack}")
    
    # Data statistics
    print(f"\n📊 DATA STATISTICS:")
    
    # Raw stats
    data_min = np.min(data)
    data_max = np.max(data)
    data_mean = np.mean(data)
    data_median = np.median(data)
    data_std = np.std(data)
    
    print(f"   Min: {data_min:.2f}")
    print(f"   Max: {data_max:.2f}")
    print(f"   Mean: {data_mean:.2f}")
    print(f"   Median: {data_median:.2f}")
    print(f"   Std: {data_std:.2f}")
    
    # Sigma-clipped stats (excluding stars)
    mean_clip, median_clip, std_clip = sigma_clipped_stats(data, sigma=3.0)
    
    print(f"\n   Background (sigma-clipped):")
    print(f"   Mean: {mean_clip:.3f}")
    print(f"   Median: {median_clip:.3f}")
    print(f"   Std: {std_clip:.3f}")
    
    # Check for problems
    print(f"\n⚠️  PROBLEM DETECTION:")
    
    n_nan = np.sum(np.isnan(data))
    n_inf = np.sum(np.isinf(data))
    n_negative = np.sum(data < 0)
    pct_negative = 100 * n_negative / data.size
    
    if n_nan > 0:
        print(f"   ❌ {n_nan} NaN pixels")
    else:
        print(f"   ✓ No NaN pixels")
    
    if n_inf > 0:
        print(f"   ❌ {n_inf} Inf pixels")
    else:
        print(f"   ✓ No Inf pixels")
    
    if pct_negative > 5:
        print(f"   ❌ {pct_negative:.1f}% negative pixels (SEVERE)")
    elif pct_negative > 1:
        print(f"   ⚠️  {pct_negative:.1f}% negative pixels (concerning)")
    elif pct_negative > 0:
        print(f"   ⚠️  {pct_negative:.2f}% negative pixels (minor)")
    else:
        print(f"   ✓ No negative pixels")
    
    # Background check
    if exptime != '???':
        expected_bg = 0
        expected_std = float(rdnoise) / float(exptime) if rdnoise != '???' else 1.0
        
        if abs(median_clip) > 3 * expected_std:
            print(f"   ⚠️  Background not well subtracted (median={median_clip:.2f})")
        else:
            print(f"   ✓ Background well subtracted")
        
        if std_clip > 5 * expected_std:
            print(f"   ⚠️  High noise (std={std_clip:.2f}, expected ~{expected_std:.2f})")
    
    # Saturation check
    if gain != '???':
        saturation = 60000 / float(gain)
        n_saturated = np.sum(data > 0.9 * saturation)
        if n_saturated > 0:
            print(f"   ⚠️  {n_saturated} saturated pixels")
        else:
            print(f"   ✓ No saturation")
    
    # SNR estimate
    bright_level = np.percentile(data, 99)
    signal = bright_level - median_clip
    snr_est = signal / std_clip if std_clip > 0 else 0
    
    print(f"\n📈 SNR ESTIMATE:")
    print(f"   Brightest pixels: {bright_level:.1f}")
    print(f"   Estimated SNR: {snr_est:.1f}")
    
    if snr_est < 10:
        print(f"   ⚠️  Low SNR - image may be too faint")
    
    # WCS check
    print(f"\n🌍 WCS CHECK:")
    try:
        wcs = WCS(header)
        if wcs.is_celestial:
            print(f"   ✓ Valid celestial WCS")
            try:
                pixel_scale = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
                print(f"   Pixel scale: {pixel_scale:.2f} arcsec/pixel")
            except:
                print(f"   ⚠️  Cannot determine pixel scale")
        else:
            print(f"   ⚠️  WCS not celestial")
    except Exception as e:
        print(f"   ❌ WCS error: {e}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    critical_issues = []
    if n_nan > 0 or n_inf > 0:
        critical_issues.append("NaN/Inf pixels")
    if pct_negative > 5:
        critical_issues.append("Excessive negative pixels")
    if snr_est < 5:
        critical_issues.append("Very low SNR")
    
    if critical_issues:
        print(f"   ❌ FAIL - Critical issues: {', '.join(critical_issues)}")
        print(f"   → This file needs fixing before photometry")
    elif pct_negative > 1 or abs(median_clip) > 10:
        print(f"   ⚠️  MARGINAL - Some issues detected")
        print(f"   → Consider applying fixes")
    else:
        print(f"   ✓ GOOD - No major issues detected")
    
    # Plots
    if show_plots:
        print(f"\n📊 Generating diagnostic plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Diagnostic: {fits_path}", fontsize=14)
        
        # 1. Full image
        ax = axes[0, 0]
        im = ax.imshow(data, origin='lower', vmin=np.percentile(data, 1), 
                      vmax=np.percentile(data, 99), cmap='gray')
        ax.set_title('Full Image')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Counts')
        
        # 2. Zoomed center
        ax = axes[0, 1]
        ny, nx = data.shape
        cy, cx = ny // 2, nx // 2
        size = 200
        cutout = data[max(0, cy-size):cy+size, max(0, cx-size):cx+size]
        im = ax.imshow(cutout, origin='lower', vmin=np.percentile(cutout, 1),
                      vmax=np.percentile(cutout, 99), cmap='gray')
        ax.set_title('Central Region')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Counts')
        
        # 3. Histogram
        ax = axes[0, 2]
        # Clip extreme outliers for better visualization
        data_clipped = np.clip(data, np.percentile(data, 0.1), np.percentile(data, 99.9))
        ax.hist(data_clipped.flatten(), bins=100, log=True, alpha=0.7)
        ax.axvline(median_clip, color='r', linestyle='--', label=f'Median={median_clip:.2f}')
        ax.axvline(0, color='k', linestyle='-', linewidth=2, label='Zero')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Count')
        ax.set_title('Histogram (log scale)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Negative pixels map
        ax = axes[1, 0]
        negative_map = (data < 0).astype(float)
        im = ax.imshow(negative_map, origin='lower', cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f'Negative Pixels ({pct_negative:.2f}%)')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Is Negative')
        
        # 5. Row/Column profiles
        ax = axes[1, 1]
        row_profile = np.median(data, axis=1)
        col_profile = np.median(data, axis=0)
        ax.plot(row_profile, label='Row median', alpha=0.7)
        ax.plot(col_profile, label='Column median', alpha=0.7)
        ax.axhline(median_clip, color='r', linestyle='--', label=f'Global median={median_clip:.2f}')
        ax.axhline(0, color='k', linestyle='-', linewidth=2)
        ax.set_xlabel('Pixel index')
        ax.set_ylabel('Median value')
        ax.set_title('Row/Column Profiles')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 6. Text summary
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
DATA SUMMARY

Shape: {data.shape[0]} × {data.shape[1]}
Min/Max: {data_min:.1f} / {data_max:.1f}
Median: {data_median:.2f}
Std: {data_std:.2f}

BACKGROUND
Median (clipped): {median_clip:.3f}
Std (clipped): {std_clip:.3f}

PROBLEMS
NaN pixels: {n_nan}
Inf pixels: {n_inf}
Negative: {pct_negative:.2f}%

SNR estimate: {snr_est:.1f}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.show()
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        fits_path = sys.argv[1]
    else:
        # Example path - change this
        fits_path = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_test/2025-09-22/cepheid_01_stacked.fits"
        print(f"No file specified. Using example: {fits_path}")
        print(f"Usage: python quick_diagnose.py <path_to_fits_file>\n")
    
    quick_diagnose(fits_path, show_plots=True)
