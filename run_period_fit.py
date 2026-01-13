import pandas as pd
from period_fitting_sinusoid import Sinusoid_Period_Finder
from matplotlib import pyplot as plt
import numpy as np

import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False # this avoids an annoying latex installation

filename = r"C:\Users\jp\OneDrive\Documents\1 Edinburgh University\Year 4\Telescope Group Project\Cepheids Data 3.csv"
df = pd.read_csv(filename)

time_list = df["Time"].dropna().astype(str).str.strip().tolist()

finder = Sinusoid_Period_Finder(
    name="Test",
    time=time_list,
    magnitude=df["Magnitude"].values,
    magnitude_error=df["Magnitude Error"].values,
)


class Finder(Sinusoid_Period_Finder):

    def __init__(self, name, time, magnitude, magnitude_error):
        super().__init__(name, time, magnitude, magnitude_error)
        self.finder = Sinusoid_Period_Finder(
            name="Test",
            time=time_list,
            magnitude=df["Magnitude"].values,
            magnitude_error=df["Magnitude Error"].values,
        )

    def run_period_analysis(self):
        best_period, best_params, best_uncertainties, best_chisqu = self.finder.fit_sinusoid()

        a0, p0, m0 = best_params
        a0_err, p0_err, m0_err = best_uncertainties

        print(
            f"Chi-Squares Best Period for Cepheid {self.name} is {best_period:.4f} days\n"
            f"Best-fit parameters:\n"
            f"Amplitude: {a0:.3f} \u00B1 {a0_err:.3f}\n"
            f"Phase: {p0:.3f} \u00B1 {p0_err:.3f}\n"
            f"Midline: {m0:.3f} \u00B1 {m0_err:.3f}\n"
            f"$\chi^2$ value of {best_chisqu:.3f}\n"
            )
        
        params, errors, tau, period0 = self.finder.sine_run_mcmc()
        mc_a, mc_p, mc_m, mc_period = params

        upper_errors, lower_errors = errors
        mc_a_err_upper, mc_p_err_upper, mc_m_err_upper, mc_period_err_upper = upper_errors
        mc_a_err_lower, mc_p_err_lower, mc_m_err_lower, mc_period_err_lower = lower_errors


        print(
            f"Emcee Best Period for Cepheid {self.name}:\n"
            f"  {mc_period:.3f} +{mc_period_err_upper:.3f} -{mc_period_err_lower:.3f} days\n"
            f"Best-fit parameters:\n"
            f"  Amplitude: {mc_a:.3f} +{mc_a_err_upper:.3f} -{mc_a_err_lower:.3f}\n"
            f"  Phase:     {mc_p:.3f} +{mc_p_err_upper:.3f} -{mc_p_err_lower:.3f}\n"
            f"  Midline:   {mc_m:.3f} +{mc_m_err_upper:.3f} -{mc_m_err_lower:.3f}\n"
            f"Autocorrelation times: {tau}\n"
            f"Mean autocorrelation time: {np.mean(tau):.3f}\n"
            f"Alternative best period: {period0}"
        )

        # original light curve
        self.finder.light_curve()

        # chi squares fitting
        self.finder.plot_sinusoid_fit()
        self.finder.sine_chisqu_contour_plot()

        # emcee fitting
        self.finder.sine_parameter_time_series()
        self.finder.sine_plot_corner()
        self.finder.sine_plot_emcee_fit()
        
if __name__ == "__main__":
    finder = Finder(
        name="Test",
        time=time_list,
        magnitude=df["Magnitude"].values,
        magnitude_error=df["Magnitude Error"].values,
    )

    finder.run_period_analysis()
