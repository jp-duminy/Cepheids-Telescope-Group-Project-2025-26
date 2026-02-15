"""
Sinusoidal period fitting Cepheids TGP 25/26
@author: jp
"""

import numpy as np
import scipy
import emcee as mc
from matplotlib import pyplot as plt
from general_functions import Astro_Functions
import corner

output_path = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Analysis/AliceInChains"

class Sinusoid_Period_Finder:
    
    def __init__(self, name: str, time: float, magnitude: float, magnitude_error: float):
        """
        Initialise cepheid parameters relevant to period fitting.
        """
        self.name = name
        self.mjd_time = Astro_Functions.modified_julian_date_converter(np.array(time)) # convert ISOT to MJD
        self.time = self.mjd_time - self.mjd_time[0]
        self.magnitude = magnitude
        self.magnitude_error = magnitude_error
        
    def light_curve(self):
        """
        Plot the light curve of each cepheid variable.
        """
        fig, ax = plt.subplots()
        
        ax.errorbar(self.mjd_time, self.magnitude, yerr=self.magnitude_error, fmt='o', 
                    label=self.name, color='black', capsize=5)
        ax.set_xlabel('Time [MJD]')
        ax.set_ylabel('Corrected Magnitude')
        ax.set_title(f"Light Curve for Cepheid {self.name}")
        ax.legend()
        ax.invert_yaxis() # brighter -> lower magnitude

        plt.show()

    def sinusoid_model(self, t, amplitude, phase, midline, period): # period at end so it can be held fixed
        """
        Returns a sinusoid with four free parameters.
        """
        return amplitude * np.sin(((2 * np.pi * t)/period) + phase) + midline
    
    # chi-squares fitting functions

    def sine_chi_sq(self, theta: list, period: float):
        """
        Chi-square function taking period as an argument so period may be iterated over.
        """
        a, p, m = theta # free parameters
        M_model = self.sinusoid_model(self.time, a, p, m, period)

        chisq = np.sum(((self.magnitude - M_model) / self.magnitude_error)**2)

        return chisq

    def fit_sinusoid(self):
        """
        Iterates over the literature range of classical cepheid periods, fixing periods whilst fitting
        other free parameters. Returns best-fit parameters, stores periods and corresponding chi square values.
        """
        p_min = 1.49107 # days, Breger (1980)
        p_max = 78.14 # days, Soszyński et al. (2024)
        self.period_range = np.linspace(p_min, p_max, 1000) # approximate period value lies in this range

        self.chisqu_vals = []
        param_vals = []
        param_uncertainties = []

        # hold period fixed and fit other parameters
        for period in self.period_range:
            # define a very rough initial guess
            a0 = (np.max(self.magnitude) - np.min(self.magnitude)) / 2 
            p0 = 0.0
            m0 = np.median(self.magnitude)
            theta = [a0, p0, m0]
            # lambda function allows period to be held fixed
            parameters, cov = scipy.optimize.curve_fit(lambda t, a, p, m: self.sinusoid_model(t, a, p, m, period), 
                                                   self.time, self.magnitude, p0=theta, 
                                                   sigma=self.magnitude_error, absolute_sigma=True)

            param_vals.append(parameters)
            uncertainties = np.sqrt(np.diag(cov)) # uncertainties are square root of covariance matrix diagonals
            param_uncertainties.append(uncertainties)

            chisqu = self.sine_chi_sq(parameters, period)
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
    
    def plot_sinusoid_fit(self):
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
        y = self.sinusoid_model(x, self.a0, self.p0, self.m0, self.period0)

        ax.errorbar(self.time, self.magnitude, yerr=self.magnitude_error, fmt='o', 
                    label='Original Data', color='black', capsize=5)
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Corrected Magnitude')
        ax.plot(x, y, label='Least-Squares Model', color='r')
        ax.legend()
        ax.invert_yaxis() # brighter -> lower magnitude
        ax.set_title(f"Least-Squares Sinusoid Fit for Cepheid {self.name}")

        plt.show()

    def sine_chisqu_contour_plot(self):
        """
        Plots a colour-mapped contour plot for the three free parameters in the chi-squares fit.
        Currently somewhat computationally expensive.
        """
        period, best_params, best_uncertainties, chisqu_min = self.fit_sinusoid()
        a0, p0, m0 = best_params
        a0_err, p0_err, o0_err = best_uncertainties
        a_vals = np.linspace(a0 - 0.5, a0 + 0.5, 100)
        p_vals = np.linspace(p0 - (0.25 * np.pi), p0 + (0.25 * np.pi), 100)
        o_vals = np.linspace(m0 - 0.5, m0 + 0.5, 100)

        # create 3D array of chi-square values
        chisqu_grid = np.array([[[self.sine_chi_sq([a, p, m], period) for a in a_vals]
                                  for p in p_vals] for m in o_vals])
        
        # since we made a linspace we find the closest value
        i = np.argmin(np.abs(a_vals - a0))
        j = np.argmin(np.abs(p_vals - p0))
        k = np.argmin(np.abs(o_vals - m0))
        
        # slice array into 2D for plot
        chisqu_a_p = chisqu_grid[k, :, :]
        chisqu_a_o = chisqu_grid[:, j, :]
        chisqu_p_o = chisqu_grid[:, :, i]

        # confidence intervals 1σ, 2σ, 3σ (see workshop 3)
        levels = [chisqu_min + 2.30,
          chisqu_min + 6.17,
          chisqu_min + 11.8]

        fig, axes = plt.subplots(1, 3)

        # amplitude vs phase
        ax = axes[0]
        cp = ax.contour(a_vals, p_vals, chisqu_a_p, levels=levels, colors='white') # white for cmap contrast
        im = ax.imshow(chisqu_a_p.T, origin='lower', extent=[a_vals[0], a_vals[-1], p_vals[0], p_vals[-1]],
                   aspect='auto', cmap='viridis', alpha=0.7) # .T transposes array (imshow syntax)
        ax.set_title(f"Amplitude vs Phase  (Midline={m0:.3f} \u00B1 {o0_err:.3f})")
        ax.set_xlabel("Amplitude [Mag]")
        ax.set_ylabel("Phase [Rad]")
        fig.colorbar(im, ax=ax, label='$\chi^2$') # add colour bar for chi-square values

        # amplitude vs midline
        ax = axes[1]
        cp = ax.contour(a_vals, o_vals, chisqu_a_o, levels=levels, colors='white')
        im = ax.imshow(chisqu_a_o.T, origin='lower', extent=[a_vals[0], a_vals[-1], o_vals[0], o_vals[-1]],
                aspect='auto', cmap='viridis', alpha=0.7)
        ax.set_title(f"Amplitude vs Midline  (Phase={p0:.3f} \u00B1 {p0_err:.3f})")
        ax.set_xlabel("Amplitude [Mag]")
        ax.set_ylabel("Midline [Mag]")
        fig.colorbar(im, ax=ax, label='$\chi^2$')

        # phase vs midline
        ax = axes[2]
        cp = ax.contour(p_vals, o_vals, chisqu_p_o, levels=levels, colors='white')
        im = ax.imshow(chisqu_p_o.T, origin='lower', extent=[p_vals[0], p_vals[-1], o_vals[0], o_vals[-1]],
                aspect='auto', cmap='viridis', alpha=0.7)
        ax.set_title(f"Phase vs Midline  (Amplitude={a0:.3f} \u00B1 {a0_err:.3f})")
        ax.set_xlabel("Phase [Rad]")
        ax.set_ylabel("Midline [Mag]")
        fig.colorbar(im, ax=ax, label='$\chi^2$')

        fig.subplots_adjust(wspace=0.4, hspace=0.3)
        plt.show()

    # emcee fitting functions

    def sine_ln_likelihood(self, theta):
        """
        The likelihood function is the probability of observing the data given the model parameters.
        Emcee works in log of parameter space, so we define the ln-likelihood function.
        This function takes a parameter vector [a, p, m, f] and returns the ln-likelihood.
        """
        a, p, m, period = theta
        modelled_magnitude = self.sinusoid_model(self.time, a, p, m, period)
        residuals = self.magnitude - modelled_magnitude
        constant = np.log(2 * np.pi * self.magnitude_error**2) # constant term added for completeness

        return -0.5 * np.sum((residuals / self.magnitude_error)**2 + constant)

    def sine_ln_prior(self, theta):
        """
        Generate flat, physically-bounded priors in parameter space.
        """
        a_min, a_max = -2 * abs(self.a0), 2 * abs(self.a0) # range is currently quite large
        p_min, p_max = -1 * np.pi, np.pi # explores all of phase space
        m_min, m_max = self.m0 - 2*abs(self.m0), self.m0 + 2*abs(self.m0) # also quite large
        period_min, period_max = 0.9 * self.period0,  1.1 * self.period0 # period cannot be negative

        a, p, m, period = theta

        if (a_min < a < a_max and

            p_min < p < p_max and
            m_min < m < m_max and
            period_min < period < period_max):
            return 0.0 # ln(1) = 0: flat prior, all values are equally likely
        else:
            return -np.inf # prior is also bounded to physical region of parameter space

    def sine_ln_prob(self, theta):
        """
        Implementation of Bayes' theorem in log-space; starts with the prior belief, computes the
        likelihood, then returns the posterior, the updated belief.
        """
        lp = self.sine_ln_prior(theta)
        if not np.isfinite(lp): # if prior is outside of physical range
            return -np.inf # declare it impossible
        return lp + self.sine_ln_likelihood(theta) # otherwise update and return log of posterior

    def sine_walker_initialisation(self, nwalkers, ndim):
        """
        Uses chi-squared result to determine best initial position for walkers.
        """
        pos = np.array([self.a0, self.p0, self.m0, self.period0]) # stitch into parameter vector

        # recreate boundaries from sine_ln_prior
        a_min, a_max = -2 * abs(self.a0), 2 * abs(self.a0)
        p_min, p_max = -np.pi, np.pi
        m_min, m_max = self.m0 - 2*abs(self.m0), self.m0 + 2*abs(self.m0)
        period_min, period_max = 0.9 * self.period0, 1.1 * self.period0  
        
        starting_position = pos + 1e-1 * np.random.randn(nwalkers, ndim)
        
        # Clip to ensure all walkers are within bounds
        starting_position[:, 0] = np.clip(starting_position[:, 0], a_min, a_max)
        starting_position[:, 1] = np.clip(starting_position[:, 1], p_min, p_max)
        starting_position[:, 2] = np.clip(starting_position[:, 2], m_min, m_max)
        starting_position[:, 3] = np.clip(starting_position[:, 3], period_min, period_max)
        return starting_position

    def sine_run_mcmc(self):
        """
        Run the emcee Monte Carlo Markov Chain.
        """
        ndim = 4 # number of dimensions
        nwalkers = 100 # numbers of walkers to explore parameter space

        pos = self.sine_walker_initialisation(nwalkers, ndim)

        sampler = mc.EnsembleSampler(nwalkers, ndim, self.sine_ln_prob) # sample the distribution

        # first allow walkers to explore parameter space
        print(f"Running burn-in for {self.name}...")
        pos = sampler.run_mcmc(pos, 500) # small burn-in of 500 steps
        print(f"Burn-in complete.")
        sampler.reset() # reset sampler before main chain

        # with walkers settled in, run the main chain
        print(f"Running production...")
        sampler.run_mcmc(pos, 10000, progress=True) # progress=True generates a progress bar

        self.sampler = sampler # keep sampler for analysis of quality of chain

        try:
            tau = self.sampler.get_autocorr_time()
            thin = int(np.mean(tau) / 2) # thinning helps speed up computing
        except mc.autocorr.AutocorrError:
            print("Warning: chain too short for reliable autocorrelation estimate. Using fixed thin=10.")
            thin = 10 
        self.thin = thin
        self.flat_samples = self.sampler.get_chain(thin=self.thin, flat=True)

        chain_filename = f"{output_path}/{self.name}_sinusoid_chain.npy"
        np.save(chain_filename, self.flat_samples)
        print(f"Chain saved to {chain_filename}")

        # Also save metadata as plain text
        metadata = {
            'name': self.name,
            'model': 'sinusoid',
            'n_params': ndim,
            'n_walkers': nwalkers,
            'n_steps': 10000,
            'thin': thin,
            'autocorr_time': tau.tolist() if hasattr(tau, 'tolist') else tau,
            'chain_shape': self.flat_samples.shape
        }
        
        metadata_filename = f"{output_path}/{self.name}_sinusoid_metadata.txt"
        with open(metadata_filename, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

        quantiles = [2.5, 50, 97.5]  # 0.025-0.975 is ~ 2σ gaussian error
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
        _, _, _, period0 = self.flat_samples[best_index]
        
        return median, errors, tau, period0

    def sine_parameter_time_series(self):
        """
        Plots the parameter time series to analyse how long it takes walkers to explore
        parameter space.
        """
        ndim = 4
        fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()
        labels = ["Amplitude", "Phase", "Midline", "Period"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3) # low opacity so we can see walker paths clearly
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("Step Number")

    def sine_plot_corner(self):
        """
        First diagnostic plot: corner plot to analyse quality of fit.
        """
        labels = ["Amplitude", "Phase", "Midline", "Period"]
        fig = corner.corner(
            self.flat_samples, labels=labels, show_titles=True, # displays uncertainties
            quantiles = [0.025, 0.5, 0.975], # 0.025-0.975 ~ 2σ gaussian error, 0.5 is the median
            title_fmt=".3f"
            ) 
        plt.show()


    #def sine_plot_emcee_fit(self):
        """
        Plot the model returned by emcee.
        """
        """
        fig, ax = plt.subplots()

        # generate some plotting data
        x = np.linspace(self.time.min(), self.time.max(), 100)
        y = self.sinusoid_model(x, self.mc_a0, self.mc_p0, self.mc_m0, self.mc_period0)

        ax.errorbar(self.time, self.magnitude, yerr=self.magnitude_error, fmt='o', 
                    label='Original Data', color='black', capsize=5)
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Corrected Magnitude')
        ax.plot(x, y, label='Modelled Data', color='r')
        ax.legend()
        ax.invert_yaxis() # brighter -> lower magnitude
        ax.set_title(f"Emcee Sinusoid Fit for Cepheid {self.name}")

        plt.show()
        """

    def sine_sample_walkers(self, nsamples, flattened_chain, time_array):
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
            mod = self.sinusoid_model(time_array, a, p, m, period)
            models.append(mod)
        
        # Calculate statistics across all models
        spread = np.std(models, axis=0)  # Standard deviation at each time point
        med_model = np.median(models, axis=0)  # Median model
        
        return med_model, spread

    def sine_plot_emcee_fit(self):
        """
        Plot the model returned by emcee with 1-sigma posterior uncertainty band.
        """
        fig, ax = plt.subplots()

        # Generate dense time array for smooth curves
        x = np.linspace(self.time.min(), self.time.max(), 200)

        # Sample from posterior to get uncertainty bands (on dense grid)
        med_model, spread = self.sine_sample_walkers(100, self.flat_samples, x)

        # Plot data points 
        ax.errorbar(self.time, self.magnitude, yerr=self.magnitude_error, fmt='o', 
                    label='Original Data', color='black', capsize=5, zorder=3)
        
        # Plot median model with uncertainty band 
        ax.plot(x, med_model, label='Median Posterior Model', 
                color='red', linewidth=2, zorder=2)
        ax.fill_between(x, med_model - spread, med_model + spread, 
                        color='grey', alpha=0.5, label=r'$1\sigma$ Posterior Spread', zorder=1)
        
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Corrected Magnitude')
        ax.legend()
        ax.invert_yaxis()  # brighter -> lower magnitude
        ax.set_title(f"Emcee Sinusoid Fit for Cepheid {self.name}")
        ax.grid(True, alpha=0.3)

        plt.show()
