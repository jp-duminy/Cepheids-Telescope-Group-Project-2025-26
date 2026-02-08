# Quality Control Strategy for Cepheid Reduction Pipeline

## Overview of Issues

Based on your error logs, your reduced FITS files likely have:

1. **Negative flux values** → NaN magnitudes
2. **Poor background subtraction** → over-subtraction
3. **Missing/inconsistent header keywords** → error propagation failures
4. **Possible cosmic rays** → contaminating apertures
5. **WCS issues** → centroid placement errors

## Root Causes in Reduction Pipeline

### Common Reduction Mistakes:

#### 1. **Over-Aggressive Background Subtraction**
```python
# BAD: Subtracting too much background
background = np.median(image)  # This includes stars!
image_sub = image - background  # Now stars can be negative

# GOOD: Use sigma-clipping to exclude stars
from astropy.stats import sigma_clipped_stats
mean, median, std = sigma_clipped_stats(image, sigma=3.0)
image_sub = image - median
```

#### 2. **Not Handling Stacking Properly**
```python
# BAD: Just averaging frames
stacked = np.mean(frames, axis=0)

# GOOD: Track exposure time and noise
stacked = np.sum(frames, axis=0)
total_exptime = exptime * nframes
header['EXPTIME'] = total_exptime
header['NSTACK'] = nframes
header['TOTRN'] = read_noise  # Total, not per-frame
```

#### 3. **Ignoring Negative Pixels**
```python
# BAD: Leave negative pixels
# (They propagate to photometry as NaN)

# GOOD: Either add offset or floor at 0
image[image < 0] = 0.1  # Avoid log(0) issues
```

#### 4. **Missing Metadata**
Ensure your reduction pipeline preserves/creates:
- `EXPTIME` or `MEANEXP`: Total exposure time
- `GAIN` or `MEANGAIN`: Gain (e-/ADU)
- `RDNOISE` or `TOTRN`: Read noise (electrons)
- `NSTACK`: Number of frames stacked
- `DATE-OBS`: Observation time
- WCS keywords: `CRVAL`, `CRPIX`, `CD` matrix

## Quality Control Workflow

### Step 1: Run Quality Checks

```bash
python quality_control.py
```

This will:
- Check all FITS files for data integrity
- Measure background statistics
- Detect saturation, cosmic rays, negative pixels
- Generate `cepheid_qc_report.csv` with metrics
- Create `exclude_from_photometry.csv` for bad files

### Step 2: Review QC Report

```python
import pandas as pd
qc = pd.read_csv('cepheid_qc_report.csv')

# Summary by quality
print(qc['quality'].value_counts())

# Worst offenders
print(qc.nlargest(10, 'pct_negative'))
print(qc.nsmallest(10, 'snr_estimate'))

# Check specific issues
print(qc[qc['pct_negative'] > 5.0])
```

### Step 3: Apply Fixes

```bash
python reduction_fixer.py
```

This will:
- Fix NaN/Inf pixels
- Re-estimate and subtract background properly
- Floor negative pixels at 0.1
- Clean cosmic rays
- Add missing header keywords
- Save fixed files to new directory

### Step 4: Re-run QC on Fixed Files

```bash
python quality_control.py  # Point to fixed directory
```

Compare metrics before/after fixing.

### Step 5: Update Photometry Pipeline

Point `photometry_pipeline_clean.py` to the fixed files:

```python
# In photometry_pipeline_clean.py
DATA_DIR = "/path/to/Cepheid_test_fixed"  # Use fixed files
```

## Specific Fixes for Your Reduction Pipeline

### If You Control the Reduction Pipeline:

#### Fix 1: Better Background Estimation

```python
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

# Estimate background properly
sigma_clip = SigmaClip(sigma=3.0)
bkg_estimator = MedianBackground()

bkg = Background2D(
    image,
    box_size=(50, 50),
    filter_size=(3, 3),
    sigma_clip=sigma_clip,
    bkg_estimator=bkg_estimator
)

image_sub = image - bkg.background
```

#### Fix 2: Preserve Noise Statistics in Stacking

```python
def stack_images_properly(images, headers):
    """Stack images with proper error propagation"""
    
    # Get individual exposure times and gains
    exptimes = [h['EXPTIME'] for h in headers]
    gains = [h.get('GAIN', 1.0) for h in headers]
    read_noises = [h.get('RDNOISE', 5.0) for h in headers]
    
    # Stack by summing (preserves Poisson statistics)
    stacked = np.sum(images, axis=0)
    
    # Total exposure time
    total_exptime = np.sum(exptimes)
    
    # Mean gain (if different)
    mean_gain = np.mean(gains)
    
    # Total read noise (adds in quadrature)
    total_rn = np.sqrt(np.sum([rn**2 for rn in read_noises]))
    
    # Update header
    header = headers[0].copy()
    header['EXPTIME'] = total_exptime
    header['MEANEXP'] = np.mean(exptimes)
    header['NSTACK'] = len(images)
    header['GAIN'] = mean_gain
    header['MEANGAIN'] = mean_gain
    header['TOTRN'] = total_rn
    header['RDNOISE'] = total_rn / np.sqrt(len(images))  # Effective per-pixel
    
    return stacked, header
```

#### Fix 3: Add Data Validation

```python
def validate_reduced_image(data, header):
    """Check reduced image before saving"""
    
    # Check for NaN/Inf
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Image contains NaN/Inf")
    
    # Check background level
    median = np.median(data)
    if abs(median) > 100:
        warnings.warn(f"Background not well subtracted: median={median}")
    
    # Check for excessive negative pixels
    pct_negative = 100 * np.sum(data < 0) / data.size
    if pct_negative > 5:
        warnings.warn(f"{pct_negative:.1f}% negative pixels")
    
    # Check required keywords
    required = ['EXPTIME', 'GAIN', 'RDNOISE', 'DATE-OBS']
    missing = [k for k in required if k not in header]
    if missing:
        raise ValueError(f"Missing required keywords: {missing}")
    
    return True
```

### If You DON'T Control the Reduction Pipeline:

Use the `reduction_fixer.py` module to post-process files:

```python
from reduction_fixer import ReductionFixer

fixer = ReductionFixer("path/to/problem.fits")
fixer.apply_all_fixes(save_fixed=True, output_dir="fixed/")
```

## Photometry Pipeline Improvements

### In `Cepheid_apertures.py`:

#### Fix 1: Robust Background Subtraction Before Centroiding

```python
def get_centroid_and_fwhm(self, data, plot=False):
    from astropy.stats import sigma_clipped_stats
    
    # Robust background estimate
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    
    # Subtract background, but keep positive
    crude_sub_data = np.maximum(data - median, 0)
    
    centroid = centroid_2dg(crude_sub_data)
    fwhm = psf.fit_fwhm(data=crude_sub_data, xypos=centroid).item()
    
    return centroid, fwhm
```

#### Fix 2: Validate Flux and Handle Failures

```python
def aperture_photometry(self, data, centroid, ap_rad, ...):
    # ... existing aperture setup ...
    
    # Sigma-clip sky annulus
    from astropy.stats import sigma_clipped_stats
    annulus_mask = sky_annulus.to_mask(method='center')
    annulus_data = annulus_mask.multiply(data)
    annulus_data_1d = annulus_data[annulus_mask.data > 0]
    
    mean_sky, median_sky, std_sky = sigma_clipped_stats(
        annulus_data_1d, sigma=3.0
    )
    
    # Use median (more robust)
    mean_sky_bckgnd_per_pixel = median_sky
    total_sky_bckgnd = mean_sky_bckgnd_per_pixel * target_aperture.area
    
    target_flux = total_flux - total_sky_bckgnd
    
    # VALIDATE
    if target_flux <= 0:
        print(f"  WARNING: Negative flux ({target_flux:.1f}). "
              f"Total={total_flux:.1f}, Sky={total_sky_bckgnd:.1f}")
        return None, None, None, None
    
    return target_flux, target_aperture.area, mean_sky_bckgnd_per_pixel, sky_annulus.area
```

#### Fix 3: Increase Aperture Sizes

```python
# In photometry_pipeline_clean.py:
ap_radius = 2.5 * fwhm  # Was 2.0
# ...
inner=2.5,  # Was 1.5
outer=4.0,  # Was 2.0
```

## Testing Protocol

### 1. Test on Single Good Image

```python
from Cepheid_apertures import AperturePhotometry

# Pick a bright, isolated Cepheid from a good night
ap = AperturePhotometry("2025-09-22/cepheid_02_stacked.fits")

# Check data
print(f"Data range: {ap.data.min():.1f} to {ap.data.max():.1f}")
print(f"Median: {np.median(ap.data):.2f}")
print(f"Negative pixels: {np.sum(ap.data < 0)}")

# Try photometry
centroid, fwhm = ap.get_centroid_and_fwhm(ap.data, plot=True)
print(f"FWHM: {fwhm:.2f} pixels")

result = ap.aperture_photometry(
    ap.data, centroid, ap_rad=2.5*fwhm,
    inner=2.5, outer=4.0, plot=True
)

if result[0] is not None:
    flux = result[0]
    mag = ap.instrumental_magnitude(flux)
    print(f"Magnitude: {mag:.3f}")
else:
    print("FAILED: Negative flux")
```

### 2. Progressive Testing

1. Test on 1 good image
2. Test on 1 date (11 Cepheids)
3. Test on all dates
4. Check for systematic failures

### 3. Validation Checks

```python
import pandas as pd

# Load results
df = pd.read_csv('all_nights_photometry.csv')

# Check for NaNs
print(f"NaN magnitudes: {df['magnitude'].isna().sum()}")
print(f"NaN errors: {df['magnitude_error'].isna().sum()}")

# Check magnitude range (should be ~ -10 to +5)
print(f"Magnitude range: {df['magnitude'].min():.1f} to {df['magnitude'].max():.1f}")

# Check for outliers
outliers = df[(df['magnitude'] < -15) | (df['magnitude'] > 10)]
print(f"Outlier measurements: {len(outliers)}")

# Check by Cepheid
print("\nMeasurements per Cepheid:")
print(df['name'].value_counts().sort_index())

# Check error distribution
print(f"\nMedian error: {df['magnitude_error'].median():.3f}")
print(f"Mean error: {df['magnitude_error'].mean():.3f}")
```

## Expected Results

After applying QC and fixes, you should see:

- **Negative flux failures**: <5% of measurements
- **Typical magnitudes**: -10 to +2 (instrumental, will vary by telescope)
- **Typical errors**: 0.01 to 0.1 mag for bright Cepheids
- **Convergence warnings**: <10% of measurements

## When to Reject Data

### Reject entire images if:
- >10% negative pixels
- SNR estimate < 5
- Contains NaN/Inf
- Missing critical header keywords
- Obvious reduction artifacts

### Reject individual measurements if:
- Negative flux
- Error > 0.5 mag
- Centroid didn't converge after 2 iterations
- FWHM < 1 pixel or > 20 pixels (likely failure)
- Star near edge of detector

## Summary Checklist

- [ ] Run `quality_control.py` to assess data quality
- [ ] Review QC report and identify problematic files
- [ ] Apply fixes with `reduction_fixer.py`
- [ ] Update photometry pipeline with fixes from this guide
- [ ] Increase aperture/cutout sizes
- [ ] Add flux validation
- [ ] Test on single image before batch processing
- [ ] Re-run full pipeline
- [ ] Validate results with magnitude/error distributions
- [ ] Exclude poor-quality measurements
- [ ] Generate light curves and check for reasonable variability

## Additional Resources

- Photutils documentation: https://photutils.readthedocs.io/
- CCD equation: Howell, "Handbook of CCD Astronomy" (2006)
- Aperture photometry best practices: Stetson (1987), PASP 99, 191
