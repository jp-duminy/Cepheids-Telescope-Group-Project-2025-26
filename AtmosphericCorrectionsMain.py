"""
Photometric corrections for Cepheid observations.

Applies:
1. Instrumental magnitude calculation
2. Atmospheric extinction correction
3. Zero-point calibration
4. Interstellar dust extinction correction

Requires:
- airmass_metadata.py (your existing FITS metadata code)
"""

import numpy as np
import csv
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import your existing FITS metadata / airmass code
import AirmassInfo as am
import Cepheid_apertures as cap


# ------------------------------------------------------------
# Extinction coefficient fitting
# ------------------------------------------------------------

def fit_extinction_weighted(airmass, Vmag, m_inst, m_err):
    """
    Weighted fit to determine atmospheric extinction coefficient (k)
    and photometric zero-point (Z).

    Fits:
        V - m_inst = Z + kX

    where:
        m_inst = -2.5 log10(counts / exptime)

    Parameters
    ----------
    airmass : array-like
        Airmass values
    Vmag : array-like
        Catalog V magnitudes
    counts : array-like
        Measured counts
    count_err : array-like
        Uncertainty in counts
    exptime : array-like
        Exposure times (seconds)

    Returns
    -------
    k : float
        Extinction coefficient (mag/airmass)
    Z : float
        Zero-point at airmass = 0
    Z_airmass1 : float
        Zero-point at airmass = 1
    k_err : float
        Uncertainty in k
    Z_err : float
        Uncertainty in Z
    """

    # Convert to numpy arrays
    airmass = np.asarray(airmass)
    Vmag = np.asarray(Vmag) #Magnitudes for standard stars

    # Dependent variable
    y = Vmag - m_inst
    X = airmass

    # Weights
    w = 1 / m_err**2

    #use statsmodels for weighted least squares to get uncertainties (for more details see https://www.geeksforgeeks.org/machine-learning/weighted-least-squares-regression-in-python/)
    X_sm = sm.add_constant(X)
    wls_model = sm.WLS(y, X_sm, weights=w)
    results = wls_model.fit()
    Z, k = results.params
    Z_err, k_err = results.bse
    #Comment

    Z_airmass1 = Z + k * 1.0
    Z_airmass1_err = np.sqrt(Z_err ** 2 + k_err ** 2)
    return k, Z_airmass1, k_err, Z_airmass1_err

# ------------------------------------------------------------
# Plotting atmospheric extinction fit
# ------------------------------------------------------------

def plot_atmospheric_extinction(airmass, Vmag, counts, count_err, exptime):
    """
    Plot atmospheric extinction fit.

    Parameters
    ----------
    airmass : array-like
        Airmass values
    Vmag : array-like
        Catalog V magnitudes
    counts : array-like
        Measured counts
    count_err : array-like
        Uncertainty in counts
    exptime : array-like
        Exposure times (seconds)
    """
    import matplotlib.pyplot as plt

    k, Z, Z1, k_err, Z_err = fit_extinction_weighted(
        airmass,
        Vmag,
        counts,
        count_err,
        exptime
    )

    m_inst = -2.5 * np.log10(counts / exptime)
    y = Vmag - m_inst

    plt.errorbar(airmass, y, yerr=1.086 * (count_err / counts), fmt='o', label='Data')
    x_fit = np.linspace(min(airmass), max(airmass), 100)
    y_fit = Z + k * x_fit
    plt.plot(x_fit, y_fit, 'r-', label=f'Fit: k={k:.3f}±{k_err:.3f}, Z(airmass=1)={Z1:.2f}')
    plt.xlabel('Airmass')
    plt.ylabel('V - m_inst')
    plt.title('Atmospheric Extinction Fit')
    plt.legend()
    plt.grid()
    plt.show()

#-------------------------------------------------------------
# Plot Parameter Space for k and Z as rings of standard deviations
#-------------------------------------------------------------
def plot_parameter_space(airmass, Vmag, counts, count_err, exptime):
    '''
    Plot parameter space for extinction coefficient (k) and zero-point (Z)
    as rings of standard deviations (1σ, 2σ, 3σ).
    Parameters
    ----------
    airmass : array-like
        Airmass values
    Vmag : array-like
        Catalog V magnitudes
    counts : array-like
        Measured counts
    count_err : array-like
        Uncertainty in counts
    exptime : array-like
        Exposure times (seconds)
    
    Returns
    -------
    None
    '''

    k_best, Z_best, Z1_best, k_err, Z_err = fit_extinction_weighted(
        airmass,
        Vmag,
        counts,
        count_err,
        exptime
    )

    k_vals = np.linspace(k_best - 3*k_err, k_best + 3*k_err, 100)
    Z_vals = np.linspace(Z_best - 3*Z_err, Z_best + 3*Z_err, 100)
    K, ZM = np.meshgrid(k_vals, Z_vals)

    chi2_map = np.zeros(K.shape)

    m_inst = -2.5 * np.log10(counts / exptime)
    y = Vmag - m_inst
    m_err = 1.086 * (count_err / counts)

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            model = ZM[i,j] + K[i,j] * airmass
            chi2_map[i,j] = np.sum(((y - model) / m_err) ** 2)

    chi2_min = np.min(chi2_map)
    delta_chi2 = chi2_map - chi2_min

    plt.contour(K, ZM, delta_chi2, levels=[2.30, 6.17, 11.8], colors=['blue', 'orange', 'red'])
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='orange', lw=2),
                    Line2D([0], [0], color='red', lw=2)]
    plt.legend(custom_lines, ['1σ', '2σ', '3σ'])
    plt.plot(k_best, Z_best, '.', markersize=10, label='Best Fit')
    plt.xlabel('Extinction Coefficient k')
    plt.ylabel('Zero-point Z')
    plt.title('Parameter Space for k and Z')
    plt.grid()
    plt.show()

# ------------------------------------------------------------
# Testing atmospheric extinction recovery
# ------------------------------------------------------------

def test_zero_point_and_extinction():
    print("\n=== TEST 1: Zero-point & atmospheric extinction ===")

    Vmag = np.array([12.43, 13.38, 8.58, 8.91])
    altitude = np.array([33.9, 74.2, 51.1, 43.7])
    exptime = np.array([30, 60, 10, 15])
    counts = np.array([73980, 74820, 954928, 1047480])
    count_err = np.array([2714, 2059, 8702, 9620])

    airmass = 1 / np.cos(np.radians(90 - altitude))

    k, Z, Z1, k_err, Z_err = fit_extinction_weighted(
        airmass,
        Vmag,
        counts,
        count_err,
        exptime
    )

    print(f"Recovered extinction coefficient k = {k:.3f} ± {k_err:.3f}")
    print(f"Recovered zero-point (X=1) = {Z1:.2f}")
    print("Target values: ZP ≈ 21.10, k ≈ 0.21")
    
    plot_atmospheric_extinction(airmass, Vmag, counts, count_err, exptime)



# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------------
    # USER INPUTS
    # --------------------------------------------------------

    fits_folder = "path/to/fits/files"
    flux_csv = "path/to/cepheid_fluxes.csv" # Input flux measurements (CHANGE)
    output_csv = "corrected_magnitudes.csv"

    #test_zero_point_and_extinction()  Comment/uncomment as needed

    """
    Assuming flux is stored in a CSV with columns: 
    filename,flux

    (change i guess (maybe get fluxes from aperture photometry code directly?))
    """
    flux_data = {}
    with open(flux_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flux_data[row["filename"]] = float(row["flux"])


    fits_files = am.collect_fits_files(fits_folder) #get airmass metadata

    results = []

    for fpath in fits_files:
        fname = fpath.split("/")[-1]

        if fname not in flux_data:
            continue

        meta = am.process_fits(fpath)
        if meta is None:
            continue

        airmass = meta["airmass"]
        flux = flux_data[fname]

        #m_final = correct_magnitude(flux, airmass, ZERO_POINT, E_BV, K_V)

        results.append([
            fname,
            flux,
            airmass,
            #m_final
        ])


    # --------------------------------------------------------
    # Write output as CSV
    # --------------------------------------------------------

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "flux",
            "airmass",
            "corrected_magnitude"
        ])
        writer.writerows(results)

    print(f"Corrected magnitudes saved to: {output_csv}")