"""
Distance inference to Andromeda via Cepheid CV1 TGP 25/26
@author: jp
"""

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from andromeda_plots import plot_distance_posterior, plot_on_pl_relation, plot_light_curve



# need to decide on literature source


class AndromedaDistance:
    """

    Resamples from both the Andromeda cepheid and classical cepheid P-L relation to compute the distance measurement

    Aiming for around 785kpc..

    """

    def __init__(self, cepheid_chain_path, pl_chain_path, ebv=0.06):
        self.cepheid_chain = np.load(cepheid_chain_path)
        self.pl_chain = np.load(pl_chain_path)
        self.A_V = 3.1 * ebv

    def pl_model(self, period, a, b):
        """
        Period-luminosity relation (so we can resample)
        """
        return a * (np.log10(period) - 1) + b

    def infer_distance(self, n_samples=10000):
        """
        Resample from both chains to produce a posterior on the distance to Andromeda.
        """
        # draw from Andromeda CV1 chain
        idx_ceph = np.random.randint(0, len(self.cepheid_chain), size=n_samples)
        self.period_samples = self.cepheid_chain[idx_ceph, 3]     # period
        self.m_apparent_samples = self.cepheid_chain[idx_ceph, 2]  # midline

        # draw from P-L chain
        idx_pl = np.random.randint(0, len(self.pl_chain), size=n_samples)
        a_samples = self.pl_chain[idx_pl, 0]
        b_samples = self.pl_chain[idx_pl, 1]
        sigma_samples = self.pl_chain[idx_pl, 2]

        # predict absolute magnitude
        self.M_samples = self.pl_model(self.period_samples, a_samples, b_samples)

        # propagate intrinsic scatter
        self.M_samples += np.random.normal(0, sigma_samples)

        # distance modulus
        self.mu_samples = self.m_apparent_samples - self.A_V - self.M_samples

        # convert to distance
        self.distance_pc = 10 ** (self.mu_samples / 5 + 1)
        self.distance_kpc = self.distance_pc / 1e3
        self.distance_mpc = self.distance_pc / 1e6

        # summary statistics (1-sigma)
        median_kpc = np.median(self.distance_kpc)
        upper_kpc = np.percentile(self.distance_kpc, 84) - median_kpc
        lower_kpc = median_kpc - np.percentile(self.distance_kpc, 16)

        median_mpc = np.median(self.distance_mpc)
        upper_mpc = np.percentile(self.distance_mpc, 84) - median_mpc
        lower_mpc = median_mpc - np.percentile(self.distance_mpc, 16)

        print(f"\n{'='*60}")
        print(f"Distance to Andromeda (via CV1)")
        print(f"{'='*60}")
        print(f"  {median_kpc:.1f} +{upper_kpc:.1f} -{lower_kpc:.1f} kpc")
        print(f"  {median_mpc:.3f} +{upper_mpc:.3f} -{lower_mpc:.3f} Mpc")
        print(f"  Literature: ~785kpc ")
        print(f"{'='*60}\n")

        return self.distance_kpc

    def plot_distance_posterior(self):
        """
        Histogram of the distance posterior with literature comparison.
        """
        fig, ax = plt.subplots()

        median = np.median(self.distance_kpc)
        lower = np.percentile(self.distance_kpc, 16)
        upper = np.percentile(self.distance_kpc, 84)

        ax.hist(self.distance_kpc, bins=50, density=True,
                color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.7)

        ax.axvline(median, color='red', linestyle='-', linewidth=2,
                   label=f'Median: {median:.0f} kpc')
        ax.axvline(lower, color='red', linestyle='--', linewidth=1)
        ax.axvline(upper, color='red', linestyle='--', linewidth=1)

        ax.axvline(770, color='green', linestyle='-', linewidth=2,
                   label='Literature: ~785 kpc')

        ax.set_xlabel('Distance [kpc]')
        ax.set_ylabel('Probability Density')
        ax.set_title('Posterior Distance to Andromeda Galaxy')
        ax.legend()
        plt.show()

    def plot_on_pl_relation(self, pl_flat_samples=None):
        """
        Show Andromeda CV1 on the P-L relation.
        """
        fig, ax = plt.subplots()

        # P-L fit line
        period_range = np.linspace(1, 80, 100)
        log_p = np.log10(period_range)

        if pl_flat_samples is None:
            pl_flat_samples = self.pl_chain

        draw = np.random.randint(0, len(pl_flat_samples), size=200)
        models = np.array([self.pl_model(period_range, pl_flat_samples[d, 0],
                                          pl_flat_samples[d, 1]) for d in draw])
        med_model = np.median(models, axis=0)
        spread = np.std(models, axis=0)

        ax.plot(log_p, med_model, 'r-', linewidth=2, label='P-L Relation')
        ax.fill_between(log_p, med_model - spread, med_model + spread,
                        color='gray', alpha=0.4)
        ax.fill_between(log_p, med_model - 2 * spread, med_model + 2 * spread,
                        color='gray', alpha=0.2)

        # Andromeda CV1
        med_logp = np.median(np.log10(self.period_samples))
        logp_err = np.std(np.log10(self.period_samples))
        med_M = np.median(self.M_samples)
        M_err = np.std(self.M_samples)

        ax.errorbar(med_logp, med_M, xerr=logp_err, yerr=M_err,
                    fmt='*', color='red', markersize=15,
                    markeredgecolor='black', markeredgewidth=0.8,
                    capsize=3, label='Andromeda CV1', zorder=5)

        ax.set_xlabel(r'$\log_{10}$(Period) [days]')
        ax.set_ylabel('Absolute Magnitude')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Andromeda CV1 on the Period-Luminosity Relation')
        plt.show()


if __name__ == "__main__":
    chain_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Analysis"

    cepheid_chain = f"{chain_dir}/AliceInChains/Andromeda_sinusoid_chain.npy"
    pl_chain = f"{chain_dir}/PLRelationChain/PL_relation_chain.npy"

    andromeda = AndromedaDistance(cepheid_chain, pl_chain, ebv=0.06)
    andromeda.infer_distance()
    plot_distance_posterior(andromeda.distance_kpc, savefig=False)
    plot_on_pl_relation(andromeda.period_samples, andromeda.M_samples,
                        andromeda.pl_chain, savefig=False)