"""
Full differential photometry pipeline for Andromeda CV1 across all LT stacked images.
Uses pixel guesses + DAOStarFinder, calibrates against Pan-STARRS g-band, converts to Johnson V.
@author: jp
"""

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from matplotlib import pyplot as plt

from Cepheid_apertures import AperturePhotometry
from photometry_per_night import SinglePhotometry
from andromeda_catalogue import andromeda_reference_catalogue, cv1_pixel_guesses

import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False

# ========================
# DIRECTORIES
# ========================
stacked_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Andromeda"
output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Andromeda/Photometry"

# ========================
# CONSTANTS
# ========================
EBV_CV1 = 0.06
REFERENCE_NIGHT = "20170608"

# Nights where the field is rotated 90 degrees anticlockwise
rotated_nights = {"20170728", "20170802", "20170818"}

BV_LITERATURE = 1.28 


def compute_ref_offsets():
    """
    Compute reference star pixel offsets from CV1, using the reference night.
    """
    cv1_x, cv1_y = cv1_pixel_guesses[REFERENCE_NIGHT]
    refs = andromeda_reference_catalogue["CV1"]

    offsets = {}
    for ref_id, ref_data in refs.items():
        dx = ref_data["x-coord"] - cv1_x
        dy = ref_data["y-coord"] - cv1_y
        offsets[ref_id] = (dx, dy)

    return offsets

def faint_star_photometry(fits_path, x_guess, y_guess, name, ebv,
                        ap_rad=8.0, width=50):
    """
    Photometry for faint stars with fixed aperture, no curve-of-growth.
    """
    phot = SinglePhotometry(
        fits_path=str(fits_path),
        x_rough=float(x_guess),
        y_rough=float(y_guess),
        name=name,
        ebv=str(ebv),
    )

    x, y = phot.locate_star(x_guess, y_guess, fwhm=4.0, sigma=3.0, plot=False)

    masked_data, x_off, y_off = phot.ap.mask_data_and_plot(
        x, y, width=width, name=name, date="LT", plot=False
    )

    centroid, _ = phot.ap.get_centroid_and_fwhm(masked_data, name, plot=False)
    # ignore returned FWHM, use fixed aperture

    flux, ap_area, sky_bkg, ann_area = phot.ap.aperture_photometry(
        masked_data, centroid, ap_rad,
        ceph_name=name, date="LT",
        inner=1.5, outer=2.0, plot=False, savefig=False
    )

    m_inst = phot.ap.instrumental_magnitude(flux)
    m_err = phot.ap.get_inst_mag_error(
        flux, ap_area, sky_bkg, ann_area,
        phot.gain, phot.exp_time, phot.read_noise, phot.stack_size
    )

    return m_inst, m_err

m_sky, m_sky_err = faint_star_photometry(
    "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Andromeda/h_e_20170608_stacked.fits",
    x_guess=766.999, y_guess=1953.36,  # adjust to empty patch
    name="empty_sky", ebv="0.0",
    ap_rad=10.0, width=100, plot=True
)

print(f"Sky aperture instrumental mag: {m_sky:.3f}")

def get_ref_position(cv1_x, cv1_y, dx, dy, date_str):
    """Compute reference star pixel position, accounting for 90 degree anticlockwise rotation."""
    if date_str in rotated_nights:
        return cv1_x + (-dy), cv1_y + dx
    return cv1_x + dx, cv1_y + dy


def extract_date(filename):
    """Extract YYYYMMDD date string from LT filename like h_e_20170608_stacked.fits."""
    return filename.split("_")[2]


def run_andromeda_photometry(plot=False):
    """
    Perform differential aperture photometry on CV1 across all LT stacked images.

    For each stacked image:
        1. Look up CV1 pixel guess for that night
        2. Compute reference star positions from offsets (with rotation correction)
        3. DAOStarFinder + aperture photometry on CV1 and reference stars
        4. Differential zero-point calibration against Pan-STARRS g
        5. Jester et al. (2005) g -> V transformation
        6. Dust correction in V-band

    Returns
    -------
    df : pd.DataFrame
        Columns: MJD, ISOT, V_mag, V_err, g_calibrated, g_err, zp, n_refs
    """
    stacked_path = Path(stacked_dir)
    stacked_files = sorted(stacked_path.glob("h_e_*_stacked.fits"))

    if len(stacked_files) == 0:
        raise FileNotFoundError(f"No stacked files found in {stacked_dir}")

    print(f"Found {len(stacked_files)} stacked LT images\n")

    # Compute offsets from reference night
    ref_offsets = compute_ref_offsets()
    refs = andromeda_reference_catalogue["CV1"]

    results = []

    for fits_path in stacked_files:
        print(f"\n{'='*60}")
        print(f"Processing: {fits_path.name}")
        print(f"{'='*60}")

        date_str = extract_date(fits_path.name)

        if date_str not in cv1_pixel_guesses:
            print(f"  SKIPPING: no pixel guess for date {date_str}")
            continue

        hdr = fits.getheader(str(fits_path))
        mjd = hdr['MJD']
        isot = hdr['DATE-OBS']

        cv1_x, cv1_y = cv1_pixel_guesses[date_str]

        # --- CV1 photometry ---
        try:
            cv1_phot = SinglePhotometry(
                fits_path=str(fits_path),
                x_rough=float(cv1_x),
                y_rough=float(cv1_y),
                name="CV1",
                ebv=str(EBV_CV1),
            )
            m_cv1, m_cv1_err = cv1_phot.raw_photometry(width=150, plot=plot)
        except Exception as e:
            print(f"  CV1 FAILED: {e}")
            continue

        # --- Empirical background correction ---
        SKY_CONTAMINATION = 124.0  # counts/s, measured from empty aperture near CV1
        cv1_flux = 10**(-0.4 * m_cv1)
        cv1_corrected_flux = cv1_flux - SKY_CONTAMINATION
        if cv1_corrected_flux <= 0:
            print(f"  SKIPPING: corrected flux negative")
            continue
        m_cv1 = -2.5 * np.log10(cv1_corrected_flux)

        # --- Reference star photometry ---
        offsets = []
        for ref_id, (dx, dy) in ref_offsets.items():
            try:
                ref_x, ref_y = get_ref_position(cv1_x, cv1_y, dx, dy, date_str)

                ref_phot = SinglePhotometry(
                    fits_path=str(fits_path),
                    x_rough=float(ref_x),
                    y_rough=float(ref_y),
                    name=ref_id,
                    ebv="0.0",
                )
                m_ref, m_ref_err = ref_phot.raw_photometry(width=150, plot=False)
                offset = refs[ref_id]["g_true"] - m_ref
                offsets.append(offset)
                print(f"  {ref_id}: m_inst = {m_ref:.3f}, offset = {offset:.3f}")

            except Exception as e:
                print(f"  {ref_id} FAILED: {e}")
                continue

        if len(offsets) < 3:
            print(f"  SKIPPING: only {len(offsets)} refs succeeded (need >= 3)")
            continue

        # --- Calibrate via differential zero-point ---
        offsets = np.array(offsets)
        zp = np.median(offsets)
        zp_err = np.std(offsets) / np.sqrt(len(offsets))

        g_cal = m_cv1 + zp
        g_cal_err = np.sqrt(m_cv1_err**2 + zp_err**2)

        # --- Convert SDSS-g to Johnson V using Jester et al. (2005) ---
        # V = g - 0.58*(g-r) - 0.01   (RMS = 0.02 mag)
        # g-r derived from literature B-V via inverse Jester:
        #   B-V = 0.98*(g-r) + 0.22  =>  g-r = (B-V - 0.22) / 0.98
        g_r = (BV_LITERATURE - 0.22) / 0.98  # ~ 0.796
        jester_offset = 0.58 * g_r + 0.01     # ~ 0.472
        jester_rms = 0.02  # mag, from Jester et al. (2005)

        V_cal = g_cal - jester_offset
        V_cal_err = np.sqrt(g_cal_err**2 + jester_rms**2)

        # --- Dust correction in V-band ---
        # A_V = 3.1 * E(B-V) (standard R_V = 3.1)
        A_V = 3.1 * EBV_CV1
        V_corrected = V_cal - A_V

        results.append({
            'MJD': mjd,
            'ISOT': isot,
            'V_mag': V_corrected,
            'V_err': V_cal_err,
            'g_calibrated': g_cal,
            'g_err': g_cal_err,
            'zp': zp,
            'n_refs': len(offsets),
        })

        print(f"\n  >> V = {V_corrected:.3f} +/- {V_cal_err:.3f}  (g = {g_cal:.3f}, ZP = {zp:.3f}, {len(offsets)} refs)")

    # --- Compile results ---
    df = pd.DataFrame(results)
    df = df.sort_values('MJD').reset_index(drop=True)

    # --- Save main CSV ---
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "andromeda_CV1_lightcurve.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nLight curve saved to: {csv_path}")
    print(f"Total epochs: {len(df)}")

    # --- Save period-fit-ready CSV ---
    period_fit_df = df[['ISOT', 'V_mag', 'V_err']].copy()
    period_fit_df.columns = ['ISOT', 'm_differential', 'm_differential_err']
    period_fit_df['Name'] = 'Andromeda_CV1'
    period_fit_path = out_path / "andromeda_CV1_period_fit.csv"
    period_fit_df.to_csv(period_fit_path, index=False)
    print(f"Period-fit-ready CSV saved to: {period_fit_path}")

    return df

if __name__ == "__main__":
    df = run_andromeda_photometry(plot=False)