"""
Publication-quality plotting functions for Andromeda CV1 distance inference.
Style matched to the P-L relation plots.
@author: jp

This is entirely made by Claude, alas I do not have time to fancify plots myself.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import gaussian_kde
import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False


def plot_distance_posterior(distance_kpc, literature_kpc=785, savefig=False):
    """
    Publication-quality posterior distance histogram with KDE overlay.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    median = np.median(distance_kpc)
    lower = np.percentile(distance_kpc, 16)
    upper = np.percentile(distance_kpc, 84)

    # histogram
    n, bins, patches = ax.hist(distance_kpc, bins=60, density=True,
                                color='#4878A8', edgecolor='white',
                                linewidth=0.4, alpha=0.75, zorder=2)

    # KDE overlay
    kde = gaussian_kde(distance_kpc)
    x_kde = np.linspace(np.percentile(distance_kpc, 0.5),
                        np.percentile(distance_kpc, 99.5), 300)
    ax.plot(x_kde, kde(x_kde), color='#1B3A5C', linewidth=2, zorder=3)

    # median and 1-sigma
    ax.axvline(median, color='#C44E52', linestyle='-', linewidth=2.0,
               label=rf'Median: {median:.0f}$^{{+{upper - median:.0f}}}_{{-{median - lower:.0f}}}$ kpc',
               zorder=4)
    ax.axvspan(lower, upper, color='#C44E52', alpha=0.12, zorder=1,
               label=r'68\% credible interval')

    # literature
    ax.axvline(literature_kpc, color='#55A868', linestyle='--', linewidth=2.0,
               label=f'Literature: {literature_kpc} kpc', zorder=4)

    ax.set_xlabel('Distance [kpc]', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Posterior Distance to M31 via Cepheid CV1', fontsize=13)
    ax.legend(fontsize=9, frameon=True, facecolor='white',
              edgecolor='black', framealpha=0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)

    if savefig:
        fig.savefig('andromeda_distance_posterior.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_on_pl_relation(period_samples, M_samples, pl_chain,
                        mw_finders=None, mw_distances=None,
                        savefig=False):
    """
    Publication-quality P-L relation with Andromeda CV1 overlaid
    and Milky Way Cepheids shown individually.
    """
    fig, (ax_pl, ax_res) = plt.subplots(2, 1, figsize=(8, 7),
                                         gridspec_kw={'height_ratios': [3, 1]},
                                         sharex=True)
    fig.subplots_adjust(hspace=0.05)

    # --- P-L fit from chain ---
    period_range = np.linspace(2, 80, 200)
    log_p = np.log10(period_range)

    draw = np.random.randint(0, len(pl_chain), size=300)
    models = np.array([pl_chain[d, 0] * (log_p - 1) + pl_chain[d, 1]
                        for d in draw])
    med_model = np.median(models, axis=0)
    spread = np.std(models, axis=0)

    # confidence bands
    ax_pl.fill_between(log_p, med_model - 2 * spread, med_model + 2 * spread,
                        color='#D4D4D4', alpha=0.6, label=r'$2\sigma$', zorder=1)
    ax_pl.fill_between(log_p, med_model - spread, med_model + spread,
                        color='#A0A0A0', alpha=0.5, label=r'$1\sigma$', zorder=2)
    ax_pl.plot(log_p, med_model, color='#C44E52', linewidth=2,
               label='P-L Fit', zorder=3)

    # --- Milky Way Cepheids (if provided) ---
    if mw_finders is not None and mw_distances is not None:
        from general_functions import Astro_Functions
        colors = plt.cm.tab20(np.linspace(0, 1, len(mw_finders)))

        a_med = np.median(pl_chain[:, 0])
        b_med = np.median(pl_chain[:, 1])

        for i, obj in enumerate(mw_finders):
            med_p = np.median(obj.flat_samples[:, -1])
            med_m_app = np.median(obj.flat_samples[:, 2])
            med_M = Astro_Functions.apparent_to_absolute(med_m_app, mw_distances[i])
            M_err = np.std(Astro_Functions.apparent_to_absolute(
                obj.flat_samples[:, 2], mw_distances[i]))

            ax_pl.errorbar(np.log10(med_p), med_M, yerr=M_err,
                           fmt='D', color=colors[i], markersize=6,
                           markeredgecolor='black', markeredgewidth=0.6,
                           capsize=3, zorder=4, label=obj.name)

            # residual
            expected = a_med * (np.log10(med_p) - 1) + b_med
            ax_res.errorbar(np.log10(med_p), med_M - expected, yerr=M_err,
                            fmt='D', color=colors[i], markersize=6,
                            markeredgecolor='black', markeredgewidth=0.6,
                            capsize=3, zorder=4)

    # --- Andromeda CV1 ---
    med_logp = np.median(np.log10(period_samples))
    logp_err = np.std(np.log10(period_samples))
    med_M = np.median(M_samples)
    M_err = np.std(M_samples)

    ax_pl.errorbar(med_logp, med_M, xerr=logp_err, yerr=M_err,
                   fmt='*', color='#C44E52', markersize=18,
                   markeredgecolor='black', markeredgewidth=1.0,
                   capsize=4, label='M31 CV1', zorder=6)

    # CV1 residual
    a_med = np.median(pl_chain[:, 0])
    b_med = np.median(pl_chain[:, 1])
    expected_cv1 = a_med * (med_logp - 1) + b_med
    ax_res.errorbar(med_logp, med_M - expected_cv1, xerr=logp_err, yerr=M_err,
                    fmt='*', color='#C44E52', markersize=18,
                    markeredgecolor='black', markeredgewidth=1.0,
                    capsize=4, zorder=6)

    # --- Formatting ---
    ax_pl.set_ylabel('Absolute Magnitude [mag]', fontsize=12)
    ax_pl.invert_yaxis()
    ax_pl.legend(fontsize=7, ncol=2, frameon=True, facecolor='white',
                 edgecolor='black', framealpha=0.8, loc='upper right')
    ax_pl.grid(True, alpha=0.2, zorder=0)
    ax_pl.tick_params(which='both', direction='in', top=True, right=True)
    ax_pl.xaxis.set_minor_locator(AutoMinorLocator())
    ax_pl.yaxis.set_minor_locator(AutoMinorLocator())
    ax_pl.set_title('Period-Luminosity Relation with M31 CV1', fontsize=13)

    ax_res.axhline(0, color='#C44E52', linestyle='--', linewidth=1.5)
    ax_res.set_xlabel(r'$\log_{10}$(Period) [days]', fontsize=12)
    ax_res.set_ylabel('Residuals [mag]', fontsize=12)
    ax_res.grid(True, alpha=0.2, zorder=0)
    ax_res.tick_params(which='both', direction='in', top=True, right=True)
    ax_res.xaxis.set_minor_locator(AutoMinorLocator())
    ax_res.yaxis.set_minor_locator(AutoMinorLocator())

    if savefig:
        fig.savefig('andromeda_pl_relation.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_light_curve(mjd, V_mag, V_err, period=None, savefig=False):
    """
    Publication-quality light curve. If period is given, also shows phased version.
    """
    if period is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- Time series ---
    ax1.errorbar(mjd, V_mag, yerr=V_err, fmt='o', color='#4878A8',
                 markeredgecolor='black', markeredgewidth=0.6,
                 capsize=4, markersize=6, label='M31 CV1', zorder=3)
    ax1.set_xlabel('Time [MJD]', fontsize=12)
    ax1.set_ylabel('Apparent V Magnitude [mag]', fontsize=12)
    ax1.set_title('M31 CV1 Light Curve', fontsize=13)
    ax1.invert_yaxis()
    ax1.legend(fontsize=9, frameon=True, facecolor='white',
               edgecolor='black', framealpha=0.8)
    ax1.tick_params(which='both', direction='in', top=True, right=True)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(True, alpha=0.2, zorder=0)

    # --- Phased light curve ---
    if period is not None:
        phase = ((mjd - mjd[0]) % period) / period
        # plot two cycles for clarity
        for offset in [0, 1]:
            ax2.errorbar(phase + offset, V_mag, yerr=V_err, fmt='o',
                         color='#C44E52' if offset == 0 else '#C44E52',
                         markeredgecolor='black', markeredgewidth=0.6,
                         capsize=4, markersize=6, alpha=0.9 if offset == 0 else 0.35,
                         zorder=3)

        ax2.set_xlabel('Phase', fontsize=12)
        ax2.set_ylabel('Apparent V Magnitude [mag]', fontsize=12)
        ax2.set_title(f'Phased Light Curve (P = {period:.1f} d)', fontsize=13)
        ax2.invert_yaxis()
        ax2.set_xlim(-0.05, 2.05)
        ax2.tick_params(which='both', direction='in', top=True, right=True)
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.grid(True, alpha=0.2, zorder=0)

    plt.tight_layout()
    if savefig:
        fig.savefig('andromeda_light_curve.pdf', dpi=300, bbox_inches='tight')
    plt.show()