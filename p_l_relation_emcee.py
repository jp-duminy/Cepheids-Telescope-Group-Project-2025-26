"""
MCMC Class Cepheids
@author: jp
"""

import numpy as np
import scipy
import emcee as mc
from matplotlib import pyplot as plt
from p_l_relation_chisqu import Cepheid_Chi_Error_Analysis
import corner

class Cepheid_MCMC(Cepheid_Chi_Error_Analysis):
    
    def __init__(self, name, period, magnitude, snr, distance):
        """
        Initialisation function that inherits the initial chi-square model parameters.
        """
        super().__init__(name, period, magnitude, snr, distance) # inherit parameters from chi squared
        
    def ln_likelihood(self, theta):
        """
        Log-likelihood function that takes a parameter vector as its input and returns the log-likelihood.
        The likelihood function is the probability of the dataset given the parameters.
        """
        a, b = theta
        chi_squared = self.chi_squared_model(theta) 
        M_err = self.magnitude_error()
        constant = np.sum(np.log(2 * np.pi * M_err**2))
        # likelihood goes as exp(-chi2/2) + const.
        
        return -0.5 * chi_squared + constant # remember ln(exp)

    def ln_prior(self, theta):
        """
        Generate priors for the posterior distribution that will be sampled from.
        """
        a, b = theta
        if -5 < a < 5 and -10 < b < 10: # broad, uninformed priors that take reasonable values 
            return -0.0 # ln(0) = 1, parameter values equally likely
        return -np.inf # prior is outside of expected range

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
        minimised_a, minimised_b, _ = self.minimise_chi_square() # start with values from chi square fit
        pos = np.array([minimised_a, minimised_b]) # stitch into parameter vector
        
        starting_position = pos + 1e-4 * np.random.randn(nwalkers, ndim) # add gaussian noise
        
        return starting_position

    def run_mcmc(self):
        """
        Run the emcee Monte Carlo Markov Chain.
        """
        ndim = 2 # number of dimensions
        nwalkers = 32 # numbers of walkers to explore parameter space
        initial_guess = [2.43, 4.05] # from Benedict et al. (2007)

        pos = self.walker_initialisation(nwalkers, ndim)

        sampler = mc.EnsembleSampler(nwalkers, ndim, self.ln_prob) # sample the distribution

        print(f"Running burn-in...")
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
        ndim = 2
        fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()
        labels = ["a", "b"]
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
        flat_samples = self.sampler.get_chain(thin=thin, flat=True)
        
        labels = ["a", "b"]
        fig = corner.corner(
            flat_samples, labels=labels, truths=truths
            )
        plt.show()
        return flat_samples

