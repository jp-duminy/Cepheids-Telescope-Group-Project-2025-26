# Cepheid Photometry Quality Control Tools

This package provides comprehensive quality control and fixing tools for your Cepheid reduction pipeline.

## Files Included

1. **`quality_control.py`** - Batch quality assessment of reduced FITS files
2. **`reduction_fixer.py`** - Automatically fix common reduction pipeline issues
3. **`quick_diagnose.py`** - Quick inspection of individual FITS files
4. **`QC_GUIDE.md`** - Comprehensive quality control strategy guide

## Quick Start

### Step 1: Diagnose a Single Image

First, look at one problematic file to understand what's wrong:

```bash
python quick_diagnose.py /path/to/cepheid_01_stacked.fits
```

This will:
- Print detailed statistics
- Identify problems (NaN, negative pixels, poor background)
- Show diagnostic plots
- Give overall assessment

### Step 2: Run Batch Quality Control

Check all your reduced FITS files:

```bash
# Edit quality_control.py to set DATA_DIR path
python quality_control.py
```

Output:
- `cepheid_qc_report.csv` - Metrics for all files
- `exclude_from_photometry.csv` - List of bad files to skip
- Terminal summary of quality grades

### Step 3: Review Results

```python
import pandas as pd

# Load QC report
qc = pd.read_csv('cepheid_qc_report.csv')

# Summary
print(qc['quality'].value_counts())

# Worst files
print(qc.nlargest(10, 'pct_negative'))

# Files to exclude
exclude = pd.read_csv('exclude_from_photometry.csv')
print(f"\nFiles to exclude: {len(exclude)}")
```

### Step 4: Apply Fixes

Fix problematic files automatically:

```bash
# Edit reduction_fixer.py to set DATA_DIR and OUTPUT_DIR
python reduction_fixer.py
```

This creates a new directory with fixed FITS files:
- Removes NaN/Inf pixels
- Re-estimates background properly
- Floors negative pixels
- Cleans cosmic rays
- Adds missing header keywords
- Normalizes to counts/sec

### Step 5: Re-run QC on Fixed Files

```bash
# Point quality_control.py to fixed directory
python quality_control.py  # Update DATA_DIR first
```

Compare metrics before/after to verify improvement.

### Step 6: Update Photometry Pipeline

Point your photometry pipeline to the fixed files:

```python
# In photometry_pipeline_clean.py
DATA_DIR = "/path/to/Cepheid_test_fixed"
```

Also apply the photometry improvements from `QC_GUIDE.md`.

## Common Issues and Solutions

### Issue: NaN Magnitudes

**Symptom:**
```
RuntimeWarning: invalid value encountered in log10
✓ mag = nan ± -0.315
```

**Cause:** Negative flux (target_flux < 0)

**Solutions:**
1. Run `reduction_fixer.py` to fix background subtraction
2. Increase aperture sizes in photometry pipeline
3. Use sigma-clipped sky background
4. Add flux validation to skip bad measurements

### Issue: Array Overlap Errors

**Symptom:**
```
❌ Error processing cepheid_05_stacked.fits: Arrays do not overlap.
```

**Cause:** Aperture/annulus extends beyond cutout boundaries

**Solutions:**
1. Increase cutout size: `half_size=50` (was 25)
2. Adjust annulus ratios: `inner=2.5, outer=4.0` (was 1.5, 2.0)
3. Check WCS - star may not be where expected

### Issue: High Magnitude Errors

**Symptom:**
```
✓ mag = -3.049 ± 0.483
```

**Cause:** Low signal-to-noise ratio

**Solutions:**
1. Use larger aperture (2.5-3.0 × FWHM)
2. Ensure background properly subtracted
3. Check for saturation or cosmic rays in aperture
4. Verify gain/read noise in header

### Issue: Excessive Negative Pixels

**Symptom:**
```
⚠️  10.5% negative pixels (poor background subtraction?)
```

**Cause:** Over-aggressive background subtraction in reduction

**Solutions:**
1. Run `reduction_fixer.py`
2. Fix your reduction pipeline:
   - Use sigma-clipped stats to estimate background
   - Don't subtract too much
   - Consider using Background2D from photutils

## Configuration

### Paths to Update

In `quality_control.py`:
```python
DATA_DIR = "/storage/.../Cepheid_test"  # Your reduced files
OUTPUT_CSV = "cepheid_qc_report.csv"
```

In `reduction_fixer.py`:
```python
DATA_DIR = "/storage/.../Cepheid_test"   # Input
OUTPUT_DIR = "/storage/.../Cepheid_test_fixed"  # Output
```

In `quick_diagnose.py`:
```bash
python quick_diagnose.py /path/to/specific/file.fits
```

### Quality Thresholds

Edit `quality_control.py` to adjust thresholds:

```python
# In identify_problematic_images()
exclude = qc_df[
    (qc_df['quality'] == 'FAIL') |
    (qc_df['n_issues'] > 0) |
    (qc_df['pct_negative'] > 5.0) |  # Adjust this threshold
    (qc_df['snr_estimate'] < 5.0)    # Adjust this threshold
]
```

## Expected Improvements

After running the QC + fix workflow, you should see:

### Before:
- 50%+ measurements with NaN magnitudes
- Many "negative flux" warnings
- "Arrays do not overlap" errors
- High magnitude errors (>0.5 mag)

### After:
- <5% NaN magnitudes (only genuine failures)
- Most measurements with errors 0.01-0.1 mag
- Fewer convergence warnings
- Reasonable magnitude range (-10 to +2)

## Testing Strategy

1. **Single file test:**
   ```bash
   python quick_diagnose.py test.fits
   ```

2. **Test photometry on one good file:**
   ```python
   from Cepheid_apertures import AperturePhotometry
   ap = AperturePhotometry("fixed/cepheid_02_stacked.fits")
   # ... do photometry ...
   ```

3. **Test on one date (all Cepheids):**
   ```bash
   # Update photometry_pipeline_clean.py to process one date
   python photometry_pipeline_clean.py
   ```

4. **Full pipeline:**
   ```bash
   # Process all dates
   python photometry_pipeline_clean.py
   ```

## Validation Checks

After running photometry, validate results:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('all_nights_photometry.csv')

# Check for NaNs
print(f"NaN mags: {df['magnitude'].isna().sum()}/{len(df)}")

# Check magnitude range
print(f"Mag range: {df['magnitude'].min():.1f} to {df['magnitude'].max():.1f}")

# Check errors
print(f"Median error: {df['magnitude_error'].median():.3f}")
good_errors = df['magnitude_error'] < 0.5
print(f"Good errors (<0.5): {good_errors.sum()}/{len(df)}")

# Plot light curves
for ceph in df['name'].unique():
    mask = df['name'] == ceph
    plt.errorbar(df[mask]['time'], df[mask]['magnitude'], 
                yerr=df[mask]['magnitude_error'], fmt='o', label=ceph)
plt.xlabel('Time')
plt.ylabel('Instrumental Magnitude')
plt.legend()
plt.gca().invert_yaxis()  # Brighter = lower magnitude
plt.show()
```

## Troubleshooting

### "No FITS files found"
Check that `DATA_DIR` path is correct and accessible.

### "Module not found"
Install required packages:
```bash
pip install numpy astropy photutils scipy pandas matplotlib
```

### "Permission denied"
Check file/directory permissions:
```bash
ls -la /path/to/data
```

### Still getting NaN magnitudes after fixes
1. Check individual file with `quick_diagnose.py`
2. Look at flux values before magnitude calculation
3. Verify sky background isn't over-subtracting
4. Try larger apertures

## Additional Help

See `QC_GUIDE.md` for:
- Detailed explanation of each quality check
- Root causes of common problems
- How to fix your reduction pipeline
- Best practices for aperture photometry
- Full testing protocol

## Summary Workflow

```
┌─────────────────────────────────────────────┐
│ 1. quick_diagnose.py on sample file        │
│    → Understand what's wrong               │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 2. quality_control.py on all files         │
│    → Generate QC report                     │
│    → Identify bad files                     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 3. reduction_fixer.py                       │
│    → Fix common issues                      │
│    → Save cleaned files                     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 4. Re-run quality_control.py               │
│    → Verify improvement                     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 5. Update photometry_pipeline_clean.py     │
│    → Use fixed files                        │
│    → Apply improvements from QC_GUIDE.md    │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 6. Run photometry pipeline                 │
│    → Should see <5% failures               │
│    → Most errors 0.01-0.1 mag              │
└─────────────────────────────────────────────┘
```

## Contact

For issues or questions about these tools, refer to `QC_GUIDE.md` or check the inline documentation in each script.
