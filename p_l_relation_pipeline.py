"""
Hierarchical Period-Luminosity Relation for Cepheids TGP 25/26
@author: jp
"""

import numpy as np
import scipy
import emcee as mc
from matplotlib import pyplot as plt
from general_functions import Astro_Functions
import corner

from distance_catalogues import cepheid_distances, cepheid_simbad_distances, cepheid_vizier_distances
import pandas as pd
from pathlib import Path
from run_period_fit import Finder

output_path = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Analysis/PLRelationChain"

class PLRelation:
    def __init__(self, objects, distances):
        self.objects = objects
        self.distances = distances

    def resample_chains(self, thin=100):
        """
        Resample from stored cepheid chains to get period and magnitude posteriors.
        """
        self.period_posteriors = []
        self.magnitude_posteriors = []
        self.names = []

        for i, obj in enumerate(self.objects):
            self.names.append(obj.name)

            # extract 1000 samples from each chain (has length 10,000 so our 1000 samples should be uncorrelated)
            period_chain = obj.flat_samples[:, -1][::thin]
            magnitude_chain = obj.flat_samples[:, 2][::thin] # magnitude comes from midline

            absolute_mag_chain = Astro_Functions.apparent_to_absolute(
            magnitude_chain, self.distances[i])
                        
            # now append each chain to the posterior list
            self.period_posteriors.append(period_chain)
            self.magnitude_posteriors.append(absolute_mag_chain)

        print(f"Successfully resampled {len(self.objects)} Cepheid chains.")

    def pl_model(self, period, a, b):
        """
        Period luminosity function.
        a: Gradient
        b: Intercept
        """
        return a * (np.log10(period) - 1) + b # luminosity-period relationship
    
    def pl_ln_prior(self, theta):
        """
        Generate priors for the posterior distribution that will be sampled from.
        """
        a, b, sigma = theta
        if -5 < a < 5 and -10 < b < 10 and 0.01 < sigma < 2.5: # broad, uninformed priors that take reasonable values 
            return -0.0 # flat prior
        return -np.inf # prior is outside of expected range
    
    def hierarchical_ln_likelihood(self, theta):
        """
        Hierarchical Bayesian inference model.
        Uncertainties are already contained within posteriors.
        """
        a, b, sigma = theta

        ln_prob = 0.0
        
        for i in range(len(self.objects)):
            period_samples = self.period_posteriors[i]
            mag_samples = self.magnitude_posteriors[i]
            n_samples = len(period_samples)

            M_model = self.pl_model(period_samples, a, b)
            residuals = mag_samples - M_model

            ln_likes = -0.5 * (residuals / sigma)**2 - np.log(sigma * np.sqrt(2 * np.pi))
            # marginalise over posterior samples using logsumexp trick
            # log(1/N * Σ exp(log_like_j)) = logsumexp(log_likes) - log(N)
            ln_prob += scipy.special.logsumexp(ln_likes) - np.log(n_samples)

        return ln_prob

    def pl_ln_prob(self, theta):
        """
        Full log-probability function combining the priors and the likelihood.
        """
        lp = self.pl_ln_prior(theta)
        if not np.isfinite(lp): # if prior is outside of expected range
            return -np.inf # declare it impossible
        return lp + self.hierarchical_ln_likelihood(theta) # new probability in log space
    
    def pl_walker_initialisation(self, nwalkers, ndim):
        """
        Initialise walkers near literature values.
        """
        pos = np.array([-2.43, -4.05, 0.1]) # from Benedict et al. (2007)
        starting_position = pos + np.array([1e-2, 1e-2, 1e-3]) * np.random.randn(nwalkers, ndim) # scale noise
    
        # clip scatter parameter as it should not be negative
        starting_position[:, 2] = np.abs(starting_position[:, 2])
        
        return starting_position

    def run_mcmc(self):
        """
        Generate the P-L relation and compute the uncertainties.    
        """

        ndim = 3 # number of dimensions
        nwalkers = 50 # numbers of walkers to explore parameter space

        pos = self.pl_walker_initialisation(nwalkers, ndim)

        sampler = mc.EnsembleSampler(nwalkers, ndim, self.pl_ln_prob) # sample the distribution

        # first allow walkers to explore parameter space
        print(f"Running burn-in for P-L Relation...")
        pos = sampler.run_mcmc(pos, 1000) # small burn-in of 100 steps
        print(f"Burn-in complete.")
        sampler.reset() # reset sampler before main chain

        # with walkers settled in, run the main chain
        print(f"Running production...")
        sampler.run_mcmc(pos, 5000, progress=True) # progress=True generates a progress bar

        self.sampler = sampler # keep sampler for analysis of quality of chain

        try:
            tau = self.sampler.get_autocorr_time()
            thin = int(np.mean(tau) / 2) # thinning helps speed up computing
        except mc.autocorr.AutocorrError:
            print("Warning: chain too short for reliable autocorrelation estimate. Using fixed thin=10.")
            thin = 10 
        self.thin = thin
        self.flat_samples = self.sampler.get_chain(thin=self.thin, flat=True)

        chain_filename = f"{output_path}/PL_relation_chain.npy"
        np.save(chain_filename, self.flat_samples)
        print(f"P-L chain saved to {chain_filename}")

        metadata = {
            'model': 'PL_relation',
            'n_params': ndim,
            'n_walkers': nwalkers,
            'n_steps': 5000,
            'thin': thin,
            'autocorr_time': tau.tolist() if hasattr(tau, 'tolist') else tau,
            'chain_shape': self.flat_samples.shape,
            'n_cepheids': len(self.objects),
            'cepheid_names': [obj.name for obj in self.objects],
        }

        metadata_filename = f"{output_path}/PL_relation_metadata.txt"
        with open(metadata_filename, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

        quantiles = [16, 50, 84]  # 0.025-0.975 is ~ 2σ gaussian error
        lower, median, upper = np.percentile(self.flat_samples, quantiles, axis=0)

        # the median (0.5) summarises the central tendency of the posterior distribution

        self.a0, self.b0, self.sigma = median

        self.a0_err = (upper[0] - median[0], median[0] - lower[0])
        self.b0_err = (upper[1] - median[1], median[1] - lower[1])
        self.sigma_err = (upper[2] - median[2], median[2] - lower[2])

        errors = (upper - median, median - lower)

        return median, errors
    
    def pl_parameter_time_series(self):
        """
        Plots the parameter time series to analyse how long it takes walkers to explore
        parameter space.
        """
        ndim = 3
        fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()
        labels = ["a (Gradient)", "b (Intercept)", "Scatter [mag]"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3) # low opacity so we can see walker paths clearly
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("Step Number")

    def pl_plot_corner(self):
        """
        First diagnostic plot: corner plot to analyse quality of fit.
        """
        labels = ["a (Gradient)", "b (Intercept)", "Scatter [mag]"]
        fig = corner.corner(
            self.flat_samples, labels=labels, show_titles=True, # displays uncertainties
            quantiles = [0.16, 0.50, 0.84], # 0.025-0.975 ~ 2σ gaussian error, 0.5 is the median
            title_fmt=".3f"
            ) 
        plt.show()
    """
    def pl_plot_emcee_fit(self):
        fig, ax = plt.subplots()

        # generate some plotting data
        all_periods = np.concatenate(self.period_posteriors)
        all_mags = np.concatenate(self.magnitude_posteriors)

        ax.scatter(np.log10(all_periods), all_mags, alpha=0.3, s=10,
               label='Posterior Samples', color='gray')

        median_periods = [np.median(p) for p in self.period_posteriors]
        median_mags = [np.median(m) for m in self.magnitude_posteriors]
        ax.errorbar(np.log10(median_periods), median_mags, fmt='o',
                label='Median Values for Each Cepheid', color='black', markersize=8)

        period_range = np.linspace(min(median_periods)*0.9, max(median_periods)*1.1, 100)
        M_fit = self.pl_model(period_range, self.a0, self.b0)
        ax.plot(np.log10(period_range), M_fit, 'r-', linewidth=2,
            label=f'Fit: M = {self.a0:.2f}(log P - 1) + {self.b0:.2f}')

        ax.set_xlabel('log₁₀(Period) [days]')
        ax.set_ylabel('Absolute Magnitude')
        ax.legend()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title('Emcee Period-Luminosity Relation Fit')
        plt.show()
    """
    """
    def pl_plot_emcee_fit(self):

        fig, ax = plt.subplots()

        colors = plt.cm.tab20(np.linspace(0, 1, len(self.objects)))

        for i in range(len(self.objects)):
            ax.scatter(np.log10(self.period_posteriors[i]), self.magnitude_posteriors[i],
                    alpha=0.1, s=5, color=colors[i])

            med_p = np.median(self.period_posteriors[i])
            med_m = np.median(self.magnitude_posteriors[i])
            ax.errorbar(np.log10(med_p), med_m, fmt='o', color=colors[i],
                        markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                        label=self.names[i])

        # fit line with uncertainty bands
        all_median_periods = [np.median(p) for p in self.period_posteriors]
        period_range = np.linspace(min(all_median_periods) * 0.9, max(all_median_periods) * 1.1, 100)
        log_p = np.log10(period_range)

        # sample from posterior to get spread
        draw = np.random.randint(0, len(self.flat_samples), size=200)
        models = np.array([self.pl_model(period_range, self.flat_samples[d, 0], self.flat_samples[d, 1])
                        for d in draw])

        med_model = np.median(models, axis=0)
        spread = np.std(models, axis=0)

        ax.plot(log_p, med_model, 'r-', linewidth=2,
                label=f'Fit: M = {self.a0:.2f}(log P - 1) {self.b0:.2f}')
        ax.fill_between(log_p, med_model - spread, med_model + spread,
                        color='gray', alpha=0.4, label=r'$1\sigma$')
        ax.fill_between(log_p, med_model - 2*spread, med_model + 2*spread,
                        color='gray', alpha=0.2, label=r'$2\sigma$')

        ax.set_xlabel(r'$\log_{10}$(Period) [days]')
        ax.set_ylabel('Absolute Magnitude')
        ax.legend(fontsize=7, ncol=2)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title('Period-Luminosity Relation')
        plt.show()
    """
    def pl_plot_emcee_fit(self):
        """
        Plot the P-L relation with residuals.
        """
        fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(8, 6),
                                        gridspec_kw={'height_ratios': [3, 1]},
                                        sharex=True)
        fig.subplots_adjust(hspace=0.05)

        colors = plt.cm.Dark2(np.linspace(0, 1, len(self.objects)))

        # --- top panel: P-L relation ---
        for i in range(len(self.objects)):
            ax.scatter(np.log10(self.period_posteriors[i]), self.magnitude_posteriors[i],
                    alpha=0.1, s=5, color=colors[i])

            med_p = np.median(self.period_posteriors[i])
            med_m = np.median(self.magnitude_posteriors[i])
            ax.errorbar(np.log10(med_p), med_m, fmt='D', color=colors[i],
                        markersize=6, markeredgecolor='black', markeredgewidth=0.8,
                        markerfacecolor='none', label=self.names[i])

        # fit line with uncertainty band
        all_median_periods = [np.median(p) for p in self.period_posteriors]
        period_range = np.linspace(min(all_median_periods) * 0.9, max(all_median_periods) * 1.1, 100)
        log_p = np.log10(period_range)

        draw = np.random.randint(0, len(self.flat_samples), size=200)
        models = np.array([self.pl_model(period_range, self.flat_samples[d, 0],
                                        self.flat_samples[d, 1]) for d in draw])

        med_model = np.median(models, axis=0)
        spread = np.std(models, axis=0)

        ax.plot(log_p, med_model, 'r-', linewidth=2,
                label=f'M = {self.a0:.2f}(log P - 1) {"+" if self.b0 >= 0 else "−"} {abs(self.b0):.2f}')
        ax.fill_between(log_p, med_model - spread, med_model + spread,
                        color='gray', alpha=0.4, label=r'$1\sigma$')
        ax.fill_between(log_p, med_model - 2 * spread, med_model + 2 * spread,
                        color='gray', alpha=0.2, label=r'$2\sigma$')

        ax.set_ylabel('Absolute Magnitude')
        ax.legend(fontsize=7, ncol=2)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title('Period-Luminosity Relation')
        ax.tick_params(labelbottom=False)

        # --- bottom panel: residuals ---
        for i in range(len(self.objects)):
            med_p = np.median(self.period_posteriors[i])
            med_m = np.median(self.magnitude_posteriors[i])
            mag_err = np.std(self.magnitude_posteriors[i])

            expected = self.pl_model(med_p, self.a0, self.b0)
            residual = med_m - expected

            ax_res.errorbar(np.log10(med_p), residual, yerr=mag_err,
                            fmt='D', color=colors[i], capsize=3,
                            markeredgecolor='black', markeredgewidth=0.8,
                            markerfacecolor='none')

            if abs(residual) > 2 * self.sigma:
                ax_res.annotate(self.names[i], (np.log10(med_p), residual),
                                fontsize=7, xytext=(5, 5), textcoords='offset points')

        ax_res.axhline(0, color='red', linestyle='--', linewidth=1)
        ax_res.set_xlabel(r'$\log_{10}$(Period) [days]')
        ax_res.set_ylabel('Residuals [mag]')
        ax_res.grid(True, alpha=0.3)

        plt.show()

    def print_results(self):
        """
        Print fitted P-L relation parameters.
        """
        print(f"\n{'='*60}")
        print(f"Period-Luminosity Relation Results")
        print(f"{'='*60}")
        print(f"Number of Cepheids: {len(self.objects)}")
        print(f"Total posterior samples: {sum(len(p) for p in self.period_posteriors)}")
        print(f"\nFitted Parameters:")
        print(f"  Slope (a):              {self.a0:.3f} +{self.a0_err[0]:.3f} -{self.a0_err[1]:.3f}")
        print(f"  Intercept (b):          {self.b0:.3f} +{self.b0_err[0]:.3f} -{self.b0_err[1]:.3f}")
        print(f"  Intrinsic Scatter (σ):  {self.sigma:.3f} +{self.sigma_err[0]:.3f} -{self.sigma_err[1]:.3f} mag")
        print(f"\nP-L Relation: M = {self.a0:.3f}(log₁₀P - 1) + {self.b0:.3f}")
        print(f"Intrinsic scatter: {self.sigma:.3f} mag")
        print(f"\nLiterature (Benedict et al. 2007): M = -2.43(log₁₀P - 1) - 4.05")
        print(f"{'='*60}\n")

if __name__ == "__main__":

    chain_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Analysis/AliceInChains"
    per_cepheid_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Analysis/CalibratedData"

    cepheid_ids = sorted(cepheid_vizier_distances.keys())

    finders = []
    distances = []

    for cep_id in cepheid_ids:
        csv_files = sorted(Path(per_cepheid_dir).glob(f"cepheid_{cep_id}_*.csv"))
        if not csv_files:
            print(f"No CSV found for Cepheid {cep_id}, skipping")
            continue

        df = pd.read_csv(csv_files[0])

        finder = Finder(
            name=df["Name"].iloc[0],
            time=df["ISOT"].dropna().astype(str).str.strip().tolist(),
            magnitude=df["m_differential"].values,
            magnitude_error=df["m_differential_err"].values,
        )

        # try sinusoid chain first, fall back to sawtooth
        name = df["Name"].iloc[0]
        sine_chain = Path(f"{chain_dir}/{name}_sinusoid_chain.npy")
        saw_chain = Path(f"{chain_dir}/{name}_sawtooth_chain.npy")

        if sine_chain.exists():
            finder.flat_samples = np.load(sine_chain)
            print(f"Loaded sinusoid chain for {name}")
        elif saw_chain.exists():
            finder.flat_samples = np.load(saw_chain)
            print(f"Loaded sawtooth chain for {name}")
        else:
            print(f"No chain found for {name}, skipping")
            continue

        finders.append(finder)
        distances.append(cepheid_vizier_distances[cep_id]["distance"])

    print(f"\nLoaded {len(finders)} Cepheids for P-L relation")

    pl = PLRelation(objects=finders, distances=distances)
    pl.resample_chains()
    pl.run_mcmc()
    pl.pl_parameter_time_series()
    pl.pl_plot_corner()
    pl.pl_plot_emcee_fit()
    pl.print_results()