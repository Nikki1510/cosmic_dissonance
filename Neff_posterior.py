import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
import corner
import seaborn as sns


# -------------------------------------------------------------------------------------
# This code generates the marginalised posterior on N_eff,
# by creating KDEs over Planck (LCDM+Neff) data and over local data (SN+BAO+H0LiCOW),
# and then running an MCMC with H0, r_d and Neff as free parameters.
# The plots generated are a cornerplot with H0, r_d and N_eff,
# and a plot showing the marginalised posterior on N_eff.

# NB: Now the MCMC part is commented out, and the file Neff_results.txt is loaded instead.
# -------------------------------------------------------------------------------------


def gaussian(x, mu, sig):
    return 1 / (sig * (2*np.pi)**0.5) * np.exp(- (x - mu)**2 / (2 * sig * sig))


def Ln_Prior(theta):
    # Unpack parameters
    H_0, r_s, N_eff = theta
    # Check if parameters are in allowed range
    if 20 < H_0 < 100 and 100 < r_s < 200 and 0 < N_eff < 20:
        return 0.0
    return - np.inf


def Ln_likelihood(theta):
    # Unpack parameters
    H_0, r_s, N_eff = theta
    likelihood_L = np.log(sum(gaussian(H_0, lenses[:, 0], 0.7) * gaussian(r_s, lenses[:, 1], 1.4)))
    likelihood_P = np.log(sum(gaussian(H_0, planck[:, 0], 0.7) * gaussian(r_s, planck[:, 1], 1.4) *
                              gaussian(N_eff, planck[:, 2], 0.03)))
    # Optional: include SH0ES measurement
    # r_SHOES = H_0 - 74.03  # ± 1.42
    # LLH_SHOES = - 0.5 * r_SHOES ** 2 / 1.42 ** 2

    LLH = likelihood_L + likelihood_P  # + LLH_SHOES

    return LLH


def Ln_Posterior(theta):

    return Ln_Prior(theta) + Ln_likelihood(theta)


def MCMC(nwalkers=100, nburn=150, nruns=400, save=False):

    ndim = 3

    # Initialise the starting positions of the walkers
    H0_init = np.random.uniform(60, 80, nwalkers)
    rs_init = np.random.uniform(120, 150, nwalkers)
    Neff_init = np.random.uniform(2, 4, nwalkers)

    p0 = np.array([H0_init, rs_init, Neff_init]).T

    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, Ln_Posterior)

    # Run the MCMC.
    print("Start with the MCMC")
    sampler.run_mcmc(p0, nruns, progress=True)

    # Collect the results
    chain = sampler.get_chain(discard=0, thin=1, flat=False)
    flatchain = sampler.get_chain(discard=nburn, thin=1, flat=True)

    print(" ")
    H0_list = flatchain[:, 0]
    rs_list = flatchain[:, 1]
    Neff_list = flatchain[:, 2]
    Results = np.array([H0_list, rs_list, Neff_list]).T

    print("H0 = ", np.around(np.mean(H0_list), 2))
    print("r_d = ", np.around(np.mean(rs_list), 2))
    print("Neff = ", np.around(np.mean(Neff_list), 2))

    # Save the results
    if save:
        np.savetxt("Output/Neff_results.txt", Results)

    return Results


def cornerplot(results, save=False):
    #Range = [(65, 80), (130, 150), (2.5, 3.8)]
    corner.corner(results, labels=[r"$H_0$", r"$r_d$", r"$N_{eff}$"])

    if save:
        plt.savefig("Figures/Neff_H0_rs_cornerplot.pdf", bbox_inches='tight')


def posterior_plot(results, save=False):

    kwargs = dict(hist_kws={'alpha': .4}, kde_kws={'linewidth': 2})

    mean = np.mean(results[:, 2])
    median = np.median(results[:, 2])
    std = np.std(results[:, 2])
    height = 0.9
    width = 0.05

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sns.distplot(results[:, 2], color="dodgerblue", label="Compact", **kwargs)
    plt.xlabel(r"$N_{eff}$", fontsize=15, labelpad=10)
    plt.ylabel("Posterior", fontsize=15, labelpad=10)
    plt.text(width, height, "Mean = {:.4}".format(mean), transform=ax.transAxes)
    plt.text(width, height - 0.05, "Median = {:.4}".format(median), transform=ax.transAxes)
    plt.text(width, height - 0.1, "Std = {:.3}".format(std), transform=ax.transAxes)

    if save:
        plt.savefig("Figures/Neff_posterior.pdf", bbox_inches='tight')


# -------------------------------------------------------------------------------------

# Load the data
planck = np.loadtxt("Chains/chain_LCDM+Neff.txt")
lenses = pd.read_csv("Chains/model4_SN+BAO_lenses=True_SH0ES=False_TRGB=False_curvature=False.txt",
                     sep=" ", header=0, dtype=np.float64).values

# Run the MCMC to get the results
#Neff_results = MCMC(save=True)

# Or, load the results if already saved
Neff_results = np.loadtxt("Chains/Neff_results.txt")

# Make a cornerplot with H0, r_d and N_eff
cornerplot(Neff_results, save=True)

# Make a plot showing the marginalised posterior for N_eff
posterior_plot(Neff_results, save=True)

plt.show()