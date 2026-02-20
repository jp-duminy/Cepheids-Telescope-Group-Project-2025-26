"""
Sawtooth period fitting Cepheids TGP 25/26
@author: jp
"""

import numpy as np
import scipy
import emcee as mc
from matplotlib import pyplot as plt
from period_fitting_sinusoid import Sinusoid_Period_Finder
import corner

output_path = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Analysis/TestChains"

class Sawtooth_Period_Finder(Sinusoid_Period_Finder):

    def __init__(self, name, time, magnitude, magnitude_error):
        super().__init__(name, time, magnitude, magnitude_error)

    def sawtooth_model(self, t, amplitude, phase, midline, width, period):
        """
        Sawtooth fitting function.
        """
        return amplitude * scipy.signal.sawtooth(((2 * np.pi * t)/period) + phase, width) + midline
    
    def saw_chi_sq(self, theta, period):
        """
        Chi-square function taking period as an argument so period may be iterated over.
        """
        a, p, m = theta
        M_model = self.sawtooth_model(self.time, a, p, m, self.w0, period)

        chisq = np.sum(((self.magnitude - M_model) / self.magnitude_error)**2)

        return chisq

    def fit_sawtooth(self):
        """
        Iterates over the literature range of classical cepheid periods, fixing periods whilst fitting
        other free parameters. Returns chisqu plot for period range and best-fit parameters.
        """
        p_min = 1.49107 # days, Breger (1980)
        p_max = 78.14 # days, Soszyński et al. (2024)
        self.period_range = np.linspace(p_min, p_max, 1000) # approximate period value lies in this range

        param_vals = []
        param_uncertainties = []

        _, sine_params, _, _ = super().fit_sinusoid()

        self.chisqu_vals = []

        a0, p0, m0 = sine_params
        
        self.w0 = 0.7 # width / period are highly degenerate

        # hold period fixed and fit other parameters
        for period in self.period_range:
            # lambda function allows period to be held fixed
            theta = [a0, p0, m0]
            parameters, cov = scipy.optimize.curve_fit(lambda t, a, p, m: self.sawtooth_model(t, a, p, m, self.w0, 
                                                                                              period), 
                                                   self.time, self.magnitude, p0=theta, 
                                                   sigma=self.magnitude_error, absolute_sigma=True,
                                                   maxfev=5000)  # Increase max iterations for sawtooth complexity

            param_vals.append(parameters)
            uncertainties = np.sqrt(np.diag(cov)) # uncertainties are square root of covariance matrix diagonals
            param_uncertainties.append(uncertainties)

            chisqu = self.saw_chi_sq(parameters, period)
            self.chisqu_vals.append(chisqu)

        # extract best fit parameters and uncertainties
        best_period_index = np.argmin(self.chisqu_vals)
        best_chisqu_period = self.period_range[best_period_index]
        best_chisqu_params = param_vals[best_period_index]
        best_chisqu_uncertainties = param_uncertainties[best_period_index]
        best_chisqu = self.chisqu_vals[best_period_index]

        self.a0, self.p0, self.m0 = best_chisqu_params
        self.period0 = best_chisqu_period

        return best_chisqu_period, best_chisqu_params, best_chisqu_uncertainties, best_chisqu
    
    def clip_from_fit(self, sigma=2.0):
        """
        Sigma clip on residuals from the chi-square best fit.
        """
        model = self.sawtooth_model(self.time, self.a0, self.p0, self.m0, self.w0, self.period0)
        residuals = self.magnitude - model
        std = np.std(residuals)
        mask = np.abs(residuals) < sigma * std
        
        n_clipped = np.sum(~mask)
        if n_clipped > 0:
            print(f"Clipped {n_clipped} points beyond {sigma}σ")
            fig, ax = plt.subplots()
            ax.errorbar(self.time[mask], residuals[mask],
                        yerr=self.magnitude_error[mask],
                        fmt='o', color='black', capsize=3, label='Kept')
            ax.errorbar(self.time[~mask], residuals[~mask],
                        yerr=self.magnitude_error[~mask],
                        fmt='x', color='red', capsize=3, markersize=10,
                        label=f'Clipped ({sigma}σ)')
            ax.axhline(0, color='red', linestyle='--', linewidth=1)
            ax.axhline(sigma * std, color='orange', linestyle=':')
            ax.axhline(-sigma * std, color='orange', linestyle=':')
            ax.set_xlabel('Time [Days]')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{sigma}σ Clipping — {self.name}')
            ax.legend()
            plt.show()
        
        
        return mask
    
    def plot_sawtooth_fit(self):
        """
        Overplot the modelled data from curve_fit onto the data.
        """
        # diagnostic plot to analyse curve fitting quality
        fig, ax = plt.subplots()
        ax.plot(self.period_range, self.chisqu_vals, label=f"$\chi^2$", color='r')
        ax.legend()
        ax.set_xlabel('Period [days]')
        ax.set_ylabel(f"$\chi^2$ Value")
        ax.set_title(f"$\chi^2$ value for period range")
        plt.show()
        
        # overplot model onto original light curve
        fig, ax = plt.subplots()

        # generate data to visualise the fit
        x = np.linspace(self.time.min(), self.time.max(), 100)
        y = self.sawtooth_model(x, self.a0, self.p0, self.m0, self.w0, self.period0)

        ax.errorbar(self.time, self.magnitude, yerr=self.magnitude_error, fmt='o', 
                    label='Original Data', color='black', capsize=5)
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Corrected Magnitude')
        ax.plot(x, y, label='Least-Squares Model', color='r')
        ax.legend()
        ax.invert_yaxis() # brighter -> lower magnitude
        ax.set_title(f"Least-Squares Sawtooth Fit for Cepheid {self.name}")

        plt.show()

    def saw_ln_likelihood(self, theta):
        """
        Log-likelihood function that takes a parameter vector as its input and returns the log-likelihood.
        The likelihood function is the probability of the dataset given the parameters.
        """
        a, p, m, period = theta
        if period <= 0:
            return -np.inf
        modelled_magnitude = self.sawtooth_model(self.time, a, p, m, self.w0, period)
        if not np.all(np.isfinite(modelled_magnitude)):
            return -np.inf
        residuals = self.magnitude - modelled_magnitude
        constant = np.log(2 * np.pi * self.magnitude_error**2) # constant term added for completeness

        return -0.5 * np.sum((residuals / self.magnitude_error)**2 + constant)

    def saw_ln_prior(self, theta):

        """
        Generate priors for the posterior distribution that will be sampled from.
        """
        a_min, a_max = 0.1, 3 # range is currently quite large
        p_min, p_max = 0, 2*np.pi # explores all of phase space
        m_min, m_max = self.m0 - 0.5, self.m0 + 0.5 # also quite large
        period_min, period_max = 0.9 * self.period0,  1.1 * self.period0 # period cannot be negative

        a, p, m, period = theta

        if (a_min < a < a_max and
        p_min < p < p_max and
        m_min < m < m_max and
        period_min < period < period_max):
            return 0.0 # ln(1) = 0: flat prior, all values are equally likely
        else:
            return -np.inf # prior is also bounded to physical region of parameter space

    def saw_ln_prob(self, theta):
        """
        Full log-probability function combining the priors and the likelihood.
        """
        lp = self.saw_ln_prior(theta)
        if not np.isfinite(lp): # if prior is outside of expected range
            return -np.inf # declare it impossible
        return lp + self.saw_ln_likelihood(theta) # new probability in log space

    def saw_walker_initialisation(self, nwalkers, ndim):
        """
        Uses chi-squared result to determine best initial position for walkers.
        """
        original_a0 = self.a0
        self.a0 = abs(self.a0)
        if original_a0 < 0:
            self.p0 = self.p0 + np.pi
            
        pos = np.array([self.a0, self.p0, self.m0, self.period0]) # stitch into parameter vector

        # recreate boundaries from saw_ln_prior
        a_min, a_max = 0.1, 2*abs(self.a0) # range is currently quite large
        p_min, p_max = 0, 2*np.pi # explores all of phase space
        m_min, m_max = self.m0 - 1, self.m0 + 1 # also quite large
        period_min, period_max = 0.9 * self.period0,  1.1 * self.period0# period cannot be negative 
        
        scales = np.array([0.5 * abs(self.a0),
                            1.0,
                            0.5,
                            0.10 * self.period0])
        
        starting_position = pos + scales * np.random.randn(nwalkers, ndim)
        
        # Clip to ensure all walkers are within bounds
        starting_position[:, 0] = np.clip(starting_position[:, 0], a_min, a_max)
        starting_position[:, 1] = np.clip(starting_position[:, 1], p_min, p_max)
        starting_position[:, 2] = np.clip(starting_position[:, 2], m_min, m_max)
        starting_position[:, 3] = np.clip(starting_position[:, 3], period_min, period_max)

        return starting_position
    
    def saw_run_mcmc(self):
        """
        Run the emcee Monte Carlo Markov Chain.
        """
        ndim = 4 # number of dimensions
        nwalkers = 100 # numbers of walkers to explore parameter space

        pos = self.saw_walker_initialisation(nwalkers, ndim)

        sampler = mc.EnsembleSampler(nwalkers, ndim, self.saw_ln_prob) # sample the distribution

        # first allow walkers to explore parameter space
        print(f"Running burn-in for {self.name}...")
        pos = sampler.run_mcmc(pos, 1000) # small burn-in of 1000 steps
        print(f"Burn-in complete.")
        sampler.reset() # reset sampler before main chain

        # with walkers settled in, run the main chain
        print(f"Running production...")
        sampler.run_mcmc(pos, 10000, progress=True) # progress=True generates a progress bar

        self.sampler = sampler # keep sampler for analysis of quality of chain

        try:
            tau = self.sampler.get_autocorr_time()
            if np.any(np.isnan(tau)):
                print(f"NaN in autocorrelation, using fixed thin.")
                thin= 10
            else:
                thin = int(np.mean(tau) / 2) # thinning helps speed up computing
        except mc.autocorr.AutocorrError:
            print("Warning: chain too short for reliable autocorrelation estimate. Using fixed thin=10.")
            thin = 10 
            tau = np.full(ndim, np.nan)
        self.thin = thin
        self.flat_samples = self.sampler.get_chain(thin=self.thin, flat=True)

        chain_filename = f"{output_path}/{self.name}_sawtooth_chain.npy"
        np.save(chain_filename, self.flat_samples)
        print(f"Chain saved to {chain_filename}")
        
        # Also save metadata as plain text
        metadata = {
            'name': self.name,
            'model': 'sawtooth',
            'n_params': ndim,
            'n_walkers': nwalkers,
            'n_steps': 10000,
            'thin': thin,
            'autocorr_time': tau.tolist() if hasattr(tau, 'tolist') else tau,
            'chain_shape': self.flat_samples.shape
        }
        
        metadata_filename = f"{output_path}/{self.name}_sawtooth_metadata.txt"
        with open(metadata_filename, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        quantiles = [16, 50, 84]  # 0.025-0.975 is ~ 2σ gaussian error
        lower, median, upper = np.percentile(self.flat_samples, quantiles, axis=0)

        # the median (0.5) summarises the central tendency of the posterior distribution
        self.mc_a0, self.mc_p0, self.mc_m0, self.mc_period0 = median

        # compute errors from quartiles
        self.mc_a0_err = (upper[0] - median[0], median[0] - lower[0])
        self.mc_p0_err = (upper[1] - median[1], median[1] - lower[1])
        self.mc_m0_err = (upper[2] - median[2], median[2] - lower[2])
        self.mc_period0_err = (upper[3] - median[3], median[3] - lower[3])
        
        errors = (upper - median, median - lower)
        
        log_prob = self.sampler.get_log_prob(thin=self.thin, flat=True)
        best_index = np.argmax(log_prob)
        a0, p0, o0, period0 = self.flat_samples[best_index]
        
        return median, errors, tau, period0

    def saw_parameter_time_series(self):
        """
        Plots the parameter time series to analyse how long it takes walkers to explore
        parameter space.
        """
        ndim = 4
        fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()
        labels = ["Amplitude","Phase", "Midline", "Period"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("Step Number")

    def saw_plot_corner(self):
        """
        First diagnostic plot: corner plot to analyse quality of fit.
        """
        labels = ["Amplitude", "Phase", "Midline", "Period"]
        fig = corner.corner(
            self.flat_samples, labels=labels, show_titles=True, # displays uncertainties
            quantiles = [0.16, 0.50, 0.84], # 0.025-0.975 ~ 2σ gaussian error, 0.5 is the median
            title_fmt=".3f",
            range=[0.99] * self.flat_samples.shape[1]
            ) 
        plt.show()

    def saw_sample_walkers(self, nsamples, flattened_chain, time_array):
        """
        Sample random parameter sets from MCMC chain to calculate model spread.
        
        Parameters:
        -----------
        nsamples : int
            Number of random samples to draw from posterior
        flattened_chain : array
            Flattened MCMC chain of shape (n_samples, n_params)
        time_array : array
            Time points where to evaluate the models (can be dense for smooth curves)
        
        Returns:
        --------
        med_model : array
            Median model across all sampled parameters
        spread : array
            Standard deviation (1-sigma spread) at each time point
        """
        models = []
    
        # Randomly select nsamples parameter sets from the chain
        draw = np.floor(np.random.uniform(0, len(flattened_chain), size=nsamples)).astype(int)
        thetas = flattened_chain[draw]
        
        # Generate a model for each sampled parameter set
        for theta in thetas:
            a, p, m, period = theta
            mod = self.sawtooth_model(time_array, a, p, m, self.w0, period)
            models.append(mod)
        
        # Calculate statistics across all models
        spread = np.std(models, axis=0)  # Standard deviation at each time point
        med_model = np.median(models, axis=0)  # Median model
        
        return med_model, spread
    
    """
    def saw_plot_emcee_fit(self):

        fig, ax = plt.subplots()

        # Generate dense time array for smooth curves
        x = np.linspace(self.time.min(), self.time.max(), 200)

        # Sample from posterior to get uncertainty bands (on dense grid)
        med_model, spread = self.saw_sample_walkers(100, self.flat_samples, x)

        # Plot data points 
        ax.errorbar(self.time, self.magnitude, yerr=self.magnitude_error, fmt='o', 
                    label='Data', color='black', capsize=5, zorder=3)
        
        # Plot median model with uncertainty band 
        ax.plot(x, med_model, label='Median Posterior Model', 
                color='red', linewidth=2, zorder=2)
        ax.fill_between(x, med_model - spread, med_model + spread, 
                        color='grey', alpha=0.5, label=r'$1\sigma$ Posterior Spread', zorder=1)
        
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Corrected Magnitude')
        ax.legend()
        ax.invert_yaxis()  # brighter -> lower magnitude
        ax.set_title(f"Emcee Sawtooth Fit for Cepheid {self.name}")
        ax.grid(True, alpha=0.3)

        plt.show()
    """

    def saw_plot_emcee_fit(self):
        """
        Phase-folded emcee fit with 2-sigma posterior uncertainty band.
        """
        phase = (self.time % self.mc_period0) / self.mc_period0

        fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(8, 6),
                                      gridspec_kw={'height_ratios': [3, 1]},
                                      sharex=True)
        fig.subplots_adjust(hspace=0.05)

        ax.errorbar(phase, self.magnitude, yerr=self.magnitude_error,
                    fmt='o', color='black', capsize=3, label='Data', zorder=3)
        ax.errorbar(phase + 1, self.magnitude, yerr=self.magnitude_error,
                    fmt='o', color='gray', capsize=3, alpha=0.4, zorder=2)

        x_phase = np.linspace(0, 2, 500)
        x_time = x_phase * self.mc_period0
        med_model, spread = self.saw_sample_walkers(100, self.flat_samples, x_time)

        ax.plot(x_phase, med_model, 'r-', linewidth=2, label='Median Model', zorder=4)
        ax.fill_between(x_phase, med_model - spread, med_model + spread,
                        color='grey', alpha=0.4, label=r'$1\sigma$', zorder=1)
        ax.fill_between(x_phase, med_model - 2*spread, med_model + 2*spread,
                        color='grey', alpha=0.2, label=r'$2\sigma$', zorder=0)
        
        model_at_data = self.sawtooth_model(self.time, self.mc_a0, self.mc_p0,
                                         self.mc_m0, self.w0, self.mc_period0)
        residuals = self.magnitude - model_at_data

        ax_res.errorbar(phase, residuals, yerr=self.magnitude_error,
                        fmt='o', color='black', capsize=3)
        ax_res.errorbar(phase + 1, residuals, yerr=self.magnitude_error,
                        fmt='o', color='gray', capsize=3, alpha=0.4)
        ax_res.axhline(0, color='red', linestyle='--', linewidth=1)
        ax_res.set_xlabel('Phase')
        ax_res.set_ylabel('Residuals')
        ax_res.set_xlim(0, 2)

        # remove x-tick labels from top panel
        ax.tick_params(labelbottom=False)

        ax.set_ylabel('Apparent Magnitude')
        ax.set_title(f'Emcee Sawtooth Fit — {self.name} (P = {self.mc_period0:.3f} d)')
        ax.invert_yaxis()
        ax.legend()
        ax.set_xlim(0, 2)
        plt.show()

    

