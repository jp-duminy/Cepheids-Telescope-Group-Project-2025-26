"""
@author: jp

Full differential photometry pipeline for Andromeda CV1 across all LT stacked images.

Very similar to photometry_per_night, but Andromeda data is one cepheid across all nights in a folder. This meant I had
to adapt the syntax accordingly to do per-cepheid rather than per-night (no data_manager.py)

Calibrates against Pan-STARRS g-band, converts to Johnson V.

"""

# so file runs in folder
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# default packages
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from matplotlib import pyplot as plt
import scienceplots

# our packages
from photometry.photometry_functions import AperturePhotometry
from photometry.photometry_per_night import SinglePhotometry
from utils.andromeda_catalogue import andromeda_reference_catalogue, cv1_pixel_guesses

plt.style.use('science')
plt.rcParams['text.usetex'] = False

# ========================
# DIRECTORIES
# ========================
stacked_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Andromeda"
output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Analysis/AndromedaData"

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

def get_ref_position(cv1_x, cv1_y, dx, dy, date_str):
    """
    Compute reference star pixel position, accounting for 90 degree anticlockwise rotation.
    """
    if date_str in rotated_nights:
        return cv1_x + (-dy), cv1_y + dx
    return cv1_x + dx, cv1_y + dy


def extract_date(filename):
    """
    Extract YYYYMMDD date string from LT filename like h_e_20170608_stacked.fits.
    """
    return filename.split("_")[2]


def run_andromeda_photometry(plot=False):
    """
    Perform differential aperture photometry on CV1 across all LT stacked images.
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
    all_ref_records = []
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
            m_cv1, m_cv1_err = cv1_phot.raw_photometry(width=50, plot=plot, override=True)
        except Exception as e:
            print(f"  CV1 FAILED: {e}")
            continue

        # --- Reference star photometry ---
        offsets = []
        ref_records = []

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
                m_ref, m_ref_err = ref_phot.raw_photometry(width=80, plot=False, override=False)
                g_known = refs[ref_id]["g_true"]
                offset = refs[ref_id]["g_true"] - m_ref
                offsets.append(offset)
                print(f"  {ref_id}: m_inst = {m_ref:.3f}, offset = {offset:.3f}")

                ref_records.append({
                    "cepheid_id": "CV1",
                    "ref_id": ref_id,
                    "ISOT": isot,
                    "m_inst": m_ref,
                    "m_inst_err": m_ref_err,
                    "g_true": g_known,
                    "offset": offset,
                })

            except Exception as e:
                print(f"  {ref_id} FAILED: {e}")
                continue

        if len(offsets) < 3:
            print(f"  SKIPPING: only {len(offsets)} refs succeeded (need >= 3)")
            continue

        all_ref_records.extend(ref_records)

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

        m_inst = m_cv1 - jester_offset
        m_inst_err = np.sqrt(m_cv1_err**2 + jester_rms**2)

        # --- Dust correction in V-band ---
        # A_V = 3.1 * E(B-V) (standard R_V = 3.1)
        A_V = 3.1 * EBV_CV1

        # Midline Correction: empty apertures
        BLENDING_CORRECTION = 19.4 - 16.645
        V_corrected = V_cal - A_V + BLENDING_CORRECTION

        results.append({
            'MJD': mjd,
            'ISOT': isot,
            'V_mag': V_corrected,
            'V_err': V_cal_err,
            'V_inst': m_inst,
            'V_inst_err': m_inst_err,
            'g_calibrated': g_cal,
            'g_err': g_cal_err,
            'zp': zp,
            'n_refs': len(offsets),
        })

        print(f"\n  >> V = {V_corrected:.3f} +/- {V_cal_err:.3f}  (g = {g_cal:.3f}, ZP = {zp:.3f}, {len(offsets)} refs)")

    # results collection
    out_path = Path(output_dir)
    df = pd.DataFrame(results)
    df = df.sort_values('MJD').reset_index(drop=True)

    ref_df = pd.DataFrame(all_ref_records)
    ref_csv_path = out_path / "Andromeda_CV1_references.csv"
    ref_df.to_csv(ref_csv_path, index=False)

    # make a .csv that is ready for straight period fitting (change column names to match)
    period_fit_df = df[['ISOT', 'V_mag', 'V_err', 'V_inst', 'V_inst_err']].copy()
    period_fit_df.columns = ['ISOT', 'm_differential', 'm_differential_err', 'm_standard', 'm_standard_err']
    period_fit_df['Name'] = 'Andromeda_CV1'
    period_fit_path = out_path / "andromeda_CV1_period_fit.csv"
    period_fit_df.to_csv(period_fit_path, index=False)
    print(f"Period-fit-ready CSV saved to: {period_fit_path}")

    return df

if __name__ == "__main__":
    df = run_andromeda_photometry(plot=False)