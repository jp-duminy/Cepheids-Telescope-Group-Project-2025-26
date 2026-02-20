import pandas as pd
from period_fitting_sinusoid import Sinusoid_Period_Finder
from period_fitting_sawtooth import Sawtooth_Period_Finder
from matplotlib import pyplot as plt
import numpy as np
import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False # this avoids an annoying latex installation

# test data files

#filename = r"C:\Users\jp\OneDrive\Documents\1 Edinburgh University\Year 4\Telescope Group Project\Cepheids Data 3.csv"
#filename = r"C:\Users\jp\OneDrive\Documents\1 Edinburgh University\Year 4\Telescope Group Project\Sawtooth Data.csv"


filename = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Analysis/CalibratedData/cepheid_01_MW_Cyg.csv"

df = pd.read_csv(filename)

time_list = df["ISOT"].dropna().astype(str).str.strip().tolist()

class Finder(Sawtooth_Period_Finder):

    def __init__(self, name, time, magnitude, magnitude_error):
        super().__init__(name, time, magnitude, magnitude_error)
        # Finder class already inherits Sinusoid_Period_Finder and Sawtooth_Period_Finder 
        # so no need to redefine them here
        #self.finder = Sinusoid_Period_Finder(
            #name="Test",
            #time=time_list,
            #magnitude=df["Magnitude"].values,
            #magnitude_error=df["Magnitude Error"].values,
        #)

    def bayesian_information_criterion(self, chisqu, n_params):
        """
        Computes the Bayesian Information Criterion, defined as BIC = kln(n) - 2ln(L).
        k = number of parameters; n = number of datapoints; L = maximised value of likelihood function
        Since ln(L) = -0.5*chisqu, -2ln(L) = chisqu
        So BIC = chisqu + kln(n)
        """
        n = len(self.magnitude) # number of datapoints

        return chisqu + n_params * np.log(n)
    
    def reduced_chisqu(self, chisqu, n_params):
        """
        Computes reduced chi-square statistic. Nice and simple.
        """
        n = len(self.magnitude)
        dof = n - n_params

        return chisqu / dof
    
    def compare_period_fit(self):
        #sine_period, sine_params, sine_uncertainties, sine_chisqu = self.finder.fit_sinusoid()

        # Fit both models 
        sine_period, sine_params, sine_uncertainties, sine_chisqu = self.fit_sinusoid()
        saw_period, saw_params, saw_uncertainties, saw_chisqu = self.fit_sawtooth()

        # Calculate BIC for both models
        n = len(self.magnitude) # number of datapoints
        sine_bic = self.bayesian_information_criterion(sine_chisqu, n_params=4)
        sine_red_chisqu = self.reduced_chisqu(sine_chisqu, n_params=4)
        saw_bic = self.bayesian_information_criterion(saw_chisqu, n_params=4)
        saw_red_chisqu = self.reduced_chisqu(saw_chisqu, n_params=4)

        # Determine which model is better (lower BIC)
        if sine_bic < saw_bic:
            best_model = "sinusoid"
            best_period = sine_period
            best_params = sine_params
            best_uncertainties = sine_uncertainties
            best_chisqu = sine_chisqu
            red_chisqu = sine_red_chisqu
            delta_bic = saw_bic - sine_bic
            print(f"Sinusoid model preferred with ΔBIC = {delta_bic:.2f}")
        else:
            best_model = "sawtooth"
            best_period = saw_period
            best_params = saw_params
            best_uncertainties = saw_uncertainties
            best_chisqu = saw_chisqu
            red_chisqu = saw_red_chisqu
            delta_bic = sine_bic - saw_bic
            print(f"Sawtooth model preferred with ΔBIC = {delta_bic:.2f}")

            # Print comparison results
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON FOR CEPHEID {self.name}")
        print(f"{'='*60}")
        print(f"Number of data points: {n}")
        print(f"\nSINUSOID MODEL:")
        print(f"  χ² = {sine_chisqu:.2f}")
        print(f"  BIC = {sine_bic:.2f}")
        print(f"  Reduced χ² = {sine_red_chisqu:.3f}")
        print(f"  Period = {sine_period:.4f} days")
        print(f"\nSAWTOOTH MODEL:")
        print(f"  χ² = {saw_chisqu:.2f}")
        print(f"  BIC = {saw_bic:.2f}")
        print(f"  Reduced χ² = {saw_red_chisqu:.3f}")
        print(f"  Period = {saw_period:.4f} days")
        print(f"\n{'='*60}")
        print(f"BEST: {best_model.upper()} (ΔBIC = {delta_bic:.2f})")
        if delta_bic > 6:
            print(f"(Strong evidence for {best_model})")
        elif delta_bic > 2:
            print(f"(Positive evidence for {best_model})")
        else:
            print(f"(Weak evidence for {best_model})")
        print(f"{'='*60}\n")
        
        return best_model, best_period, best_params, best_uncertainties, best_chisqu
    
    def run_period_analysis(self):
        """
            Complete period analysis pipeline:
            1. Compare models using BIC
            2. Show chi-square plots for both models
            3. Ask user to accept or override BIC recommendation
            4. Run MCMC on the chosen model
            5. Generate final plots
        """
        # Step 1: Compare models using BIC
        best_model, best_period, best_params, best_uncertainties, best_chisqu = self.compare_period_fit()
        
        # Step 2: Show chi-square plots for BOTH models so user can visually compare
        print("\nGenerating chi-square comparison plots for both models...")
        
        # Fit and plot sinusoid
        self.fit_sinusoid()
        mask = self.clip_from_fit(sigma=3.0)  # 3σ clipping
        if np.sum(~mask) > 0:
            self.time = self.time[mask]
            self.magnitude = self.magnitude[mask]
            self.magnitude_error = self.magnitude_error[mask]
        sine_period, sine_params, sine_uncertainties, sine_chisqu = self.fit_sinusoid()
        print("Displaying SINUSOID chi-square plot...")
        self.plot_sinusoid_fit()
        
        # Fit and plot sawtooth
        saw_period, saw_params, saw_uncertainties, saw_chisqu = self.fit_sawtooth()
        print("Displaying SAWTOOTH chi-square plot...")
        self.plot_sawtooth_fit()
        
        # Step 3: Ask user to accept or reject the BIC recommendation
        print(f"\n{'='*60}")
        while True:
            user_input = input(f"BIC recommends {best_model.upper()}. Accept this model? (y/n): ").strip().lower()
            if user_input == 'y':
                print(f"✓ Using {best_model.upper()} model\n")
                break
            elif user_input == 'n':
                # Switch to the other model
                if best_model == "sinusoid":
                    best_model = "sawtooth"
                    best_period = saw_period
                    best_params = saw_params
                    best_uncertainties = saw_uncertainties
                    best_chisqu = saw_chisqu
                else:
                    best_model = "sinusoid"
                    best_period = sine_period
                    best_params = sine_params
                    best_uncertainties = sine_uncertainties
                    best_chisqu = sine_chisqu
                print(f"✓ Switching to {best_model.upper()} model\n")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        print(f"{'='*60}\n")
        
        # Step 4: Refit the chosen model to ensure instance variables are set correctly
        if best_model == "sinusoid":
            best_period, best_params, best_uncertainties, best_chisqu = self.fit_sinusoid()
        else:
            best_period, best_params, best_uncertainties, best_chisqu = self.fit_sawtooth()
        
        # Print chi-square results for chosen model
        if best_model == "sinusoid":
            a0, p0, m0 = best_params
            a0_err, p0_err, m0_err = best_uncertainties
            print(
                f"Chi-Square Best Period for Cepheid {self.name} is {best_period:.4f} days\n"
                f"Best-fit parameters (Sinusoid):\n"
                f"Amplitude: {a0:.3f} ± {a0_err:.3f}\n"
                f"Phase: {p0:.3f} ± {p0_err:.3f}\n"
                f"Midline: {m0:.3f} ± {m0_err:.3f}\n"
                f"χ² value: {best_chisqu:.3f}\n"
            )
        else:  # sawtooth
            a0, p0, m0 = best_params
            a0_err, p0_err, m0_err = best_uncertainties
            print(
                f"Chi-Square Best Period for Cepheid {self.name} is {best_period:.4f} days\n"
                f"Best-fit parameters (Sawtooth):\n"
                f"Amplitude: {a0:.3f} ± {a0_err:.3f}\n"
                f"Phase: {p0:.3f} ± {p0_err:.3f}\n"
                f"Midline: {m0:.3f} ± {m0_err:.3f}\n"
                f"χ² value: {best_chisqu:.3f}\n"
            )
        
        # Step 2: Run MCMC on the winning model
        if best_model == "sinusoid":
            params, errors, tau, period0 = self.sine_run_mcmc()
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
        else:  # sawtooth
            params, errors, tau, period0 = self.saw_run_mcmc()
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
        
        # Step 3: Generate plots
        self.light_curve()  # original light curve
        
        if best_model == "sinusoid":
            # Chi-square fitting plots
            self.plot_sinusoid_fit()
            self.sine_chisqu_contour_plot()
            # MCMC plots
            self.sine_parameter_time_series()
            self.sine_plot_corner()
            self.sine_plot_emcee_fit()
        else:  # sawtooth
            # Chi-square fitting plots
            self.plot_sawtooth_fit()
            #self.saw_chisqu_contour_plot()
            # MCMC plots
            self.saw_parameter_time_series()
            self.saw_plot_corner()
            self.saw_plot_emcee_fit()


if __name__ == "__main__":
    finder = Finder(
        name="MW Cyg",
        time=time_list,
        magnitude=df["m_differential"].values,
        magnitude_error=df["m_differential_err"].values,
    )

    finder.run_period_analysis()
