"""
Period fitting Cepheids TGP 25/26
@author: jp
"""

import numpy as np
import scipy
import emcee as mc
from matplotlib import pyplot as plt
import astropy
from p_l_relation_chisqu import Cepheid_Chi_Error_Analysis
from general_functions import Astro_Functions
import emcee
import corner

class Cepheid_Period_Finder:
    
    def __init__(self, name, time: float, magnitude: float, magnitude_error: float, snr: float):
        """
        Initialise cepheid parameters relevant to period fitting.
        """
        self.name = name
        self.time = Astro_Functions.modified_julian_date_converter(np.array(time)) # convert ISO to MJD
        self.magnitude = magnitude
        self.snr = snr
        self.magnitude_error = magnitude_error
        
    def light_curve(self):
        """
        Plot the light curve of each cepheid variable.
        """
        fig, ax = plt.subplots()
        t = self.time - self.time[0] # start x axis at t=0 for clarity
        
        ax.errorbar(t, self.magnitude, yerr=self.magnitude_error, fmt='o')
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Corrected Magnitude')
        ax.set_title(self.name)
        ax.invert_yaxis() # brighter -> lower magnitude

        ax.set_title(f"Light Curve for Cepheid {self.name}")
        plt.show()

    def sinusoid_model(self, t, amplitude, frequency, phase, offset):
        """
        Sinusoid fitting function.
        """
        return amplitude * np.sin(2 * np.pi * frequency * (t - phase)) + offset
    
    def fit_sinusoid(self):
        """
        Fit a sinusoid to the data and estimate the best-fit parameters using chi-squares.
        """
        parameters, cov = scipy.optimize.curve_fit(self.sinusoid_model, self.time, self.magnitude, 
                                                   sigma=self.magnitude_error, absolute_sigma=True)
        uncertainties = np.sqrt(np.diag(cov))
        print(f"Best-fit parameters:")
        for i in range(len(parameters)):
            print(f"{parameters[i]} \u00B1 {uncertainties[i]}")

        self.a0, self.f0, self.p0, self.o0 = parameters

        return parameters, uncertainties
    
    def plot_sinusoid_fit(self):
        """
        Overplot the modelled data from curve_fit onto the data.
        """
        fig, ax = plt.subplots()
        y = self.sinusoid_model(self.time, self.a0, self.f0, self.p0, self.o0)
        ax.errorbar(self.time, self.magnitude, yerr=self.magnitude_error, fmt='o', label='Data')
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Corrected Magnitude')
        ax.set_title(self.name)
        ax.plot(self.time, y, label='Modelled Data')
        ax.legend()
        ax.invert_yaxis() # brighter -> lower magnitude
        ax.set_title(f"Fitting for Cepheid {self.name}")

        plt.show()

    def sawtooth_model(self, t, amplitude, frequency, phase, offset, width):
        """
        Sawtooth fitting function.
        """
        return amplitude * scipy.signal.sawtooth(2 * np.pi * frequency * (t - phase), width) + offset
    
    def ln_likelihood(self, theta):
        """
        Log-likelihood function that takes a parameter vector as its input and returns the log-likelihood.
        The likelihood function is the probability of the dataset given the parameters.
        """
        a, f, p, o, w = theta
        modelled_magnitude = self.sawtooth_model(self.time, a, f, p, o, w)
        residuals = self.magnitude - modelled_magnitude
        constant = np.log(2 * np.pi * self.magnitude_error**2)

        return -0.5 * np.sum((residuals / self.magnitude_error)**2 + constant)

    def ln_prior(self, theta):
        """
        Generate priors for the posterior distribution that will be sampled from.
        """
        a_min, a_max = -2 * abs(self.a0), 2 * abs(self.a0)
        f_min, f_max = 0.5 * self.f0, 2 * self.f0 # frequency can't be negative
        p_min, p_max = self.p0 - 2*abs(self.p0), self.p0 + 2*abs(self.p0)
        o_min, o_max = self.o0 - 2*abs(self.o0), self.o0 + 2*abs(self.o0)
        w_min, w_max = 0.0, 1.0  # width must be in range 0-1

        a, f, p, o, w = theta
        
        if (a_min < a < a_max and
            f_min < f < f_max and
            p_min < p < p_max and
            o_min < o < o_max and
            w_min <= w <= w_max):
            return 0.0
        else:
            return -np.inf
        
    def ln_prob(self, theta):
        """
        Full log-probability function combining the priors and the likelihood.
        """
        lp = self.ln_prior(theta)
        if not np.isfinite(lp): # if prior is outside of expected range
            return -np.inf # declare it impossible
        return lp + self.ln_likelihood(theta) # new probability in log space

    def walker_initialisation(self, nwalkers, ndim):
        """
        Uses chi-squared result to determine best initial position for walkers.
        """
        w0 = 0.5
        pos = np.array([self.a0, self.f0, self.p0, self.o0, w0]) # stitch into parameter vector
        
        starting_position = pos + 1e-2 * np.random.randn(nwalkers, ndim) # add gaussian noise
        
        return starting_position
    
    def run_mcmc(self):
        """
        Run the emcee Monte Carlo Markov Chain.
        """
        ndim = 5 # number of dimensions
        nwalkers = 100 # numbers of walkers to explore parameter space

        pos = self.walker_initialisation(nwalkers, ndim)

        sampler = mc.EnsembleSampler(nwalkers, ndim, self.ln_prob) # sample the distribution

        print(f"Running burn-in for cepheid...")
        pos = sampler.run_mcmc(pos, 100) # let walkers explore parameter space
        print(f"Burn-in complete.")
        sampler.reset() # reset sampler before main chain


        print(f"Running production...")
        sampler.run_mcmc(pos, 5000, progress=True) # run main chain

        self.sampler = sampler # keep sampler for analysis of quality of chain
        return sampler

    def parameter_time_series(self):
        """
        Plots the parameter time series to analyse how long it takes walkers to explore
        parameter space.
        """
        ndim = 5
        fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()
        labels = ["Amplitude", "Frequency", "Phase", "Offset", "Width"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("Step Number")

    def plot_corner(self, truths):
        """
        First diagnostic plot: corner plot to analyse quality of fit.
        """
        tau = self.sampler.get_autocorr_time()
        print(f"Autocorrelation times: {tau}")
        thin  = int(np.mean(tau) / 2)
        self.flat_samples = self.sampler.get_chain(thin=thin, flat=True)
        
        labels = ["Amplitude", "Frequency", "Phase", "Offset", "Width"]
        fig = corner.corner(
            self.flat_samples, labels=labels, truths=truths
            )
        plt.show()
    
    def plot_emcee_fit(self):
        """
        Plot the model returned by emcee.
        """
        a0, f0, p0, o0, w0 = np.mean(self.flat_samples)

        fig, ax = plt.subplots()
        y = self.sawtooth_model(self.time, a0, f0, p0, o0, w0)
        ax.errorbar(self.time, self.magnitude, yerr=self.magnitude_error, fmt='o', label='Data')
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Corrected Magnitude')
        ax.set_title(self.name)
        ax.plot(self.time, y, label='Modelled Data')
        ax.legend()
        ax.invert_yaxis() # brighter -> lower magnitude
        ax.set_title(f"Emcee Fit for Cepheid {self.name}")

        plt.show()




    
        

        
        
        
