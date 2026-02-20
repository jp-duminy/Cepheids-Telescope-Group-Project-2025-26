"""
@author: jp

Class which shows light curves for standard + differential magnitudes and shows the offset of reference stars.

Useful diagnostic.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from general_functions import Astro_Functions

import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False # this avoids an annoying latex installation

cepheid_file = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Analysis/CalibratedData/cepheid_01_MW_Cyg.csv"
reference_file = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Analysis/CalibratedData/references_01_MW_Cyg.csv"

cep_df = pd.read_csv(cepheid_file)
ref_df = pd.read_csv(reference_file)

cep_time_list = cep_df["ISOT"].dropna().astype(str).str.strip().tolist()
ref_time_list = ref_df["ISOT"].dropna().astype(str).str.strip().tolist()

class Lightcurves:

    def __init__(self, name: str, cep_time: float, ref_time, m_stand, m_stand_err,
                 m_diff, m_diff_err, ref_df):
        """
        Initialise cepheid parameters relevant to period fitting.
        """
        self.name = name
        self.cep_mjd_time = Astro_Functions.modified_julian_date_converter(np.array(cep_time)) # convert ISOT to MJD
        self.ref_mjd_time = Astro_Functions.modified_julian_date_converter(np.array(ref_time))

        self.cep_time = self.cep_mjd_time - self.cep_mjd_time[0]
        self.ref_time = self.ref_mjd_time - self.ref_mjd_time[0]

        self.m_stand = m_stand
        self.m_stand_err = m_stand_err

        self.m_diff = m_diff
        self.m_diff_err = m_diff_err

        self.ref_df = ref_df

    def standard_mag_light_curve(self):
        """
        Plot the light curve of a cepheid using its standard magnitudes (theoretical version).
        """
        fig, ax = plt.subplots()
        
        ax.errorbar(self.cep_mjd_time, self.m_stand, yerr=self.m_stand_err, fmt='o', 
                    label=self.name, color='black', capsize=5)
        ax.set_xlabel('Time [MJD]')
        ax.set_ylabel('Standard Magnitude [mag]')
        ax.set_title(f"Light Curve for Cepheid {self.name} (Standard Magnitudes)")
        ax.legend()
        ax.invert_yaxis() # brighter -> lower magnitude

        plt.show()

    def differential_mag_light_curve(self):
        """
        Plot the light curve of a cepheid using its differential magnitudes (empirical version)
        """
        fig, ax = plt.subplots()
        
        ax.errorbar(self.cep_mjd_time, self.m_diff, yerr=self.m_diff_err, fmt='o', 
                    label=self.name, color='black', capsize=5)
        ax.set_xlabel('Time [MJD]')
        ax.set_ylabel('Differential Magnitude [mag]')
        ax.set_title(f"Light Curve for Cepheid {self.name} (Differential Magnitudes)")
        ax.legend()
        ax.invert_yaxis() # brighter -> lower magnitude

        plt.show()

    def plot_reference_stability(self):
        """
        Plots the magnitudes of the reference stars across the nights for a cepheid.
        Can diagnose whether the differential photometry is accurate.
        """
        if self.ref_df is None or self.ref_df.empty:
            print(f"No reference data for {self.name}")
            return
        
        df = self.ref_df.copy()
        df["MJD"] = self.ref_mjd_time
        fig, ax = plt.subplots()
        for ref_id, group in df.groupby("ref_id"):
            ax.plot(group["MJD"], group["offset"], 'o-', label=ref_id, markersize=4)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel("MJD")
        ax.set_ylabel("Offset (m_cal - V_true) [mag]")
        ax.legend()
        ax.set_title(f"Reference Star Magnitudes — {self.name}")
        plt.show()

    def compare_light_curves(self):
        """
        Overplot standard and differential magnitudes to compare scatter.
        """
        fig, ax = plt.subplots()

        ax.errorbar(self.cep_mjd_time, self.m_stand, yerr=self.m_stand_err, fmt='o',
                    label='Standard', color='black', capsize=5)
        ax.errorbar(self.cep_mjd_time, self.m_diff, yerr=self.m_diff_err, fmt='s',
                    label='Differential', color='red', capsize=5)

        ax.set_xlabel('Time [MJD]')
        ax.set_ylabel('Magnitude [mag]')
        ax.set_title(f"Standard vs Differential — {self.name}")
        ax.legend()
        ax.invert_yaxis()
        plt.show()

def main():

    lc = Lightcurves(
    name="MW Cyg",
    cep_time=cep_time_list,
    ref_time=ref_time_list,
    m_stand=cep_df["m_standard"].values,
    m_stand_err=cep_df["m_standard_err"].values,
    m_diff=cep_df["m_differential"].values,
    m_diff_err=cep_df["m_differential_err"].values,
    ref_df=ref_df,
    )

    lc.standard_mag_light_curve()
    lc.differential_mag_light_curve()
    lc.plot_reference_stability()
    lc.compare_light_curves()

if __name__ == "__main__":
    main()