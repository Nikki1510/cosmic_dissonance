import matplotlib.pyplot as plt
import corner
import pickle
import seaborn as sns
from matplotlib import ticker
from SN_BAO import *
from Lenses import *
from Functions import r_fid

# --------------------------------------------------------------------------------------
"""
This script can only be ran once you've already generated data with the Run.py script.
The goal of this script is to visually check the output of the generated data.
It generates 3 different figures:
- A corner plot with r_s, H0, q0, j0 (and O_k if applicable)
- A figure with all the lensing observables + model predictions
- A figure with observables from SN & BAO + model predictions
"""
# --------------------------------------------------------------------------------------


def plot_corner(model, freeparams=4, lenses=True, SHOES=False, TRGB=False, curvature=False, save=False):
    """
    Function that creates a corner plot from the results.
    :param model: Choose 1, 2, 3 (all polynomial parametrizations) or 4 (LCDM).
    :param lenses: Calibration from lensed quasars are included in results (True) or not (False).
    :param SHOES: Calibration from Cepheids by SH0ES are included in results (True) or not (False).
    :param TRGB: Calibration from TRGB by CCHP are included in results (True) or not (False).
    :param curvature: O_k is a free parameter (True) or fixed to 0.0 (False).
    :param freeparams: If choosing model 1, 2 or 3, how many free parameters does the polynomial parametrization
    have? Choose between 3 and 4.
    :param save: Saves the file (True) or not (False).
    :return: A corner plot figure with the correlations of r_s, H0, q0, j0 (and O_k if applicable).
    """

    # Load the MCMC chains
    if model == 4:
        results = np.loadtxt("Chains/model4_SN+BAO_lenses=" + str(lenses) + "_SH0ES=" + str(SHOES) + "_TRGB=" +
                             str(TRGB) + "_curvature=" + str(curvature) + ".txt")
    else:
        results = np.loadtxt("Chains/" + str(freeparams) + "fp_SN+BAO_lenses=" + str(lenses) + "_SH0ES=" +
                str(SHOES) + "_TRGB=" + str(TRGB) + "_model" + str(model) + "_curvature=" + str(curvature) + ".txt")

    # Plotting parameters and labels
    labels = [r"$H_0$", r"$r_s$", r"$q_0$", r"$j_0$", r"$\Omega_k$"]
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams["font.size"] = 18
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8

    # Make the corner plot
    corner.corner(results, labels=labels)

    # If save=True: save the files
    if save:
        if model == 4:
            plt.savefig("Figures/Cornerplot_model4_Lenses_params:_lenses=" + str(lenses) + "_SH0ES=" + str(SHOES) +
                        "_TRGB=" + str(TRGB) + "_model" + str(model) + "_curvature=" + str(curvature) + ".pdf",
                        bbox_inches='tight')
        else:
            plt.savefig("Figures/Cornerplot_" + str(freeparams) + "fp_Lenses_params:_lenses=" + str(lenses) + "_SH0ES="
                        + str(SHOES) + "_TRGB=" + str(TRGB) + "_model" + str(model) + "_curvature=" + str(curvature)
                        + ".pdf", bbox_inches='tight')


def plot_lenses(model, freeparams=4, lenses=True, SHOES=False, TRGB=False, curvature=False, save=False):
    """
    :param model: Choose 1, 2, 3 (all polynomial parametrizations) or 4 (LCDM).
    :param lenses: Calibration from lensed quasars are included in results (True) or not (False).
    :param SHOES: Calibration from Cepheids by SH0ES are included in results (True) or not (False).
    :param TRGB: Calibration from TRGB by CCHP are included in results (True) or not (False).
    :param curvature: O_k is a free parameter (True) or fixed to 0.0 (False).
    :param freeparams: If choosing model 1, 2 or 3, how many free parameters does the polynomial parametrization
    have? Choose between 3 and 4.
    :param save: Saves the file (True) or not (False).
    :return: An overview of the lensing data (time-delay distances and angular diameter distances),
    and a plot of the results predicted by the expansion on top of it to check its accuracy.
    """

    # Load the lenses list
    if model == 4:
        with open("Testing/model4_Lenses_params:_lenses=" + str(lenses) + "_SH0ES=" + str(SHOES) + "_TRGB=" +
                    str(TRGB) + "_curvature=" + str(curvature) + ".pickle", 'rb') as p2:
            lenses_list = pickle.load(p2)
    else:
        with open("Testing/" + str(freeparams) + "fp_Lenses_params:_lenses=" + str(lenses) + "_SH0ES=" + str(SHOES) +
                "_TRGB=" + str(TRGB) + "_model" + str(model) + "_curvature=" + str(curvature) + ".pickle", 'rb') as p2:
            lenses_list = pickle.load(p2)


    # Take random samples of the lensing chains, to reduce the time for plotting the Kernal Density Estimator (KDE).
    Slength = 15000         # length of the random samples
    data_lenses = [np.array(random.sample(list(data_RXJ1131), Slength)),
                   np.array(random.sample(list(data_PG1115), Slength)),
                   np.flip(np.array(random.sample(list(data_J1206), Slength))),
                   0, data_WFI2033[:,0], data_HE0435]

    # Define the x ranges to plot the likelihood functions on.
    N = 200                 # number of points for the x-interval
    xrange_W = np.linspace(2500, 7500, N)
    xrange_H = np.linspace(1000, 5000, N)
    xrange_B_ddt = np.linspace(500, 6000, N)
    xrange_B_dd = np.linspace(500, 6000, N)
    # Define the corresponding y values by calculating the likelihoods
    # (either with a KDE or with a skewed lognormal, both from the file Lenses.py)
    yrange_B_ddt, yrange_B_dd = [], []
    yrange_W, yrange_H = [], []
    for d in range(N):
        yrange_W.append(np.exp(KDE_LLH_WFI2033(xrange_W[d])))
        yrange_H.append(np.exp(KDE_LLH_HE0435(xrange_H[d])))
        yrange_B_ddt.append(np.exp(B1608_ddt.sklogn_analytical_likelihood(xrange_B_ddt[d])))
        yrange_B_dd.append(np.exp(B1608_dd.sklogn_analytical_likelihood(xrange_B_dd[d])))

    # Define the x and y limits for the plots
    xrange = [0, 0, 0, 0, xrange_W, xrange_H]
    yrange = [0, 0, 0, 0, yrange_W, yrange_H]
    xlim = [(1800, 2400), (1100, 2000), (4500, 8000)]
    ylim = [(400, 1250), (300, 1100), (700, 3200)]
    # Labels for the lenses. Order of the lenses: RXJ1131, PG1115, J1206, B1608, WFI2033, HE0435.
    labels_lenses = ["RXJ1131", "PG1115", "J1206", "B1608", "WFI2033", "HE0435"]

    # Plotting parameters
    plt.rcParams["font.size"] = 10
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    # Start plotting. Make 6 panels, one for each lens. The first 3 show 2D KDEs for ddt and da, the 4th shows
    # separate 1D likelihoods for ddt and da, and the 5th and 6th show the 1D KDEs for ddt only.
    fig, axes = plt.subplots(3, 2, figsize=(9, 10))
    ax = axes.flatten()
    fig.subplots_adjust(hspace=0.3, wspace=0.26, top=0.91, bottom=0.09, left=0.12, right=0.95)
    fig.suptitle("Lenses check", fontsize=20)

    # --------------------------------------------------------------------------------------------------
    # Panel 1, 2 and 3: Lenses RXJ1131, PG1115 and J1206 (2D KDE of ddt and da)
    for l in range(3):
        # Print name of the lens
        ax[l].text(.95, .85, labels_lenses[l], horizontalalignment='right', transform=ax[l].transAxes, fontsize=13)
        # Plot 2D KDE with seaborn
        sns.kdeplot(data_lenses[l][:,1], data_lenses[l][:,0], n_levels=6, ax=ax[l], shade=True, shade_lowest=False)
        # Plot the predicted points
        ax[l].errorbar(np.mean(lenses_list[l][0]), np.mean(lenses_list[l][1]), xerr=np.std(lenses_list[l][0]),
                     yerr=np.std(lenses_list[l][1]), marker='s', color="black", ms=9, zorder=2)
        # Adjust number of ticks
        M = 4
        yticks = ticker.MaxNLocator(M)
        ax[l].yaxis.set_major_locator(yticks)
        # Set x and y limits & labels
        ax[l].set_xlim(xlim[l])
        ax[l].set_ylim(ylim[l])
        ax[l].set_xlabel(r"$D_{\Delta t}$ [Mpc]")
        ax[l].set_ylabel(r"$D_A$ [Mpc]")

    # --------------------------------------------------------------------------------------------------
    # Panel 4: Lens B1608 (2 lognormals for ddt and da)
    # Print name of the lens
    ax[3].text(.95, .85, labels_lenses[3], horizontalalignment='right', transform=ax[3].transAxes, fontsize=13)
    offset = max(yrange_B_ddt)+ 0.0002
    # Set y plotting range
    ymax = (max(yrange_B_ddt) + max(yrange_B_dd) + 0.0003)*100
    ymin = -0.0001*100

    # - Plot da -
    # Calculate the mean and standard deviation of the predicted da point
    B_dd = np.mean(lenses_list[3][1])
    B_dd_std = np.std(lenses_list[3][1])
    # Plot KDE
    ax[3].plot(xrange_B_dd, (np.array(yrange_B_dd) + offset)*100, label=r"$D_{A}$", color="C2")
    # Plot prediction mean value as a vertical line
    ax[3].plot([B_dd, B_dd], [offset*100, ymax], color="black")
    # Plot prediction standard deviation as a gray area around the line
    ax[3].fill_between([B_dd - B_dd_std, B_dd + B_dd_std], ymax, offset*100, color="gray", alpha=0.3)

    # - Plot ddt -
    # Calculate the mean and standard deviation of the predicted ddt point
    B_ddt = np.mean(lenses_list[3][0])
    B_ddt_std = np.std(lenses_list[3][0])
    # Plot KDE
    ax[3].plot(xrange_B_ddt, np.array(yrange_B_ddt)*100, label=r"$D_{\Delta t}$", color="C1")
    # Plot prediction mean value as a vertical line
    ax[3].plot([B_ddt, B_ddt], [ymin, offset * 100], color="black")
    # Plot prediction standard deviation as a gray area around the line
    ax[3].fill_between([B_ddt - B_ddt_std, B_ddt + B_ddt_std], offset * 100, ymin, color="gray", alpha=0.3)

    # Set x, y labels, limits and create a legend
    ax[3].set_xlabel("Distance [Mpc]")
    ax[3].set_ylabel(r"Log-normal ($\times 10^{-2}$)")
    ax[3].set_ylim(ymin, ymax)
    ax[3].legend(loc='upper center')

    # --------------------------------------------------------------------------------------------------
    # Panel 5 and 6: Lenses WFI2033 and HE0435 (KDE for ddt)
    for l in (4, 5):
        # Set parameter for y plotting range
        ymax = max(yrange[l])+0.0002
        # Calculate mean and standard deviations of the predicted ddt values
        ddt_mean = np.mean(lenses_list[l][0])
        ddt_std = np.std(lenses_list[l][0])
        # Print name of the lens
        ax[l].text(.95, .85, labels_lenses[l], horizontalalignment='right', transform=ax[l].transAxes, fontsize=13)
        # Plot KDE
        ax[l].plot(xrange[l], np.array(yrange[l])*100)
        # Plot prediction mean value as a vertical line
        ax[l].axvline(x=ddt_mean, color="black")
        # Plot prediction standard deviation as a gray area around the line
        ax[l].fill_between([ddt_mean - ddt_std, ddt_mean + ddt_std], ymax*100, -ymax*3, color="gray", alpha=0.3)
        # Set x, y labels and limits
        ax[l].set_xlabel(r"$D_{\Delta t}$ [Mpc]")
        ax[l].set_ylabel(r"KDE ($\times 10^{-2}$)")
        ax[l].set_ylim(-ymax*3, ymax*100)
    # --------------------------------------------------------------------------------------------------
    # If save=True: save the files
    if save:
        if model == 4:
            plt.savefig("Figures/Lenses_model4_Lenses_params:_lenses=" + str(lenses) + "_SH0ES=" + str(SHOES) +
                        "_TRGB=" + str(TRGB) + "_model" + str(model) + "_curvature=" + str(curvature) + ".pdf",
                        bbox_inches='tight')
        else:
            plt.savefig("Figures/Lenses_" + str(freeparams) + "fp_Lenses_params:_lenses=" + str(lenses) + "_SH0ES="
                        + str(SHOES) + "_TRGB=" + str(TRGB) + "_model" + str(model) + "_curvature=" + str(curvature)
                        + ".pdf", bbox_inches='tight')


def plot_SN_BAO(model, freeparams=4, lenses=True, SHOES=False, TRGB=False, curvature=False, save=False):
    """
    :param model: Choose 1, 2, 3 (all polynomial parametrizations) or 4 (LCDM).
    :param lenses: Calibration from lensed quasars are included in results (True) or not (False).
    :param SHOES: Calibration from Cepheids by SH0ES are included in results (True) or not (False).
    :param TRGB: Calibration from TRGB by CCHP are included in results (True) or not (False).
    :param curvature: O_k is a free parameter (True) or fixed to 0.0 (False).
    :param freeparams: If choosing model 1, 2 or 3, how many free parameters does the polynomial parametrization
    have? Choose between 3 and 4.
    :param save: Saves the file (True) or not (False).
    :return: An overview of the SN and BAO data (H, Dm and mu), and a plot of the results predicted by the expansion
    on top to check its accuracy.
    """

    # Load the data
    if model == 4:
        with open("Testing/model4_H_Dm_mu_params:_lenses=" + str(lenses) + "_SH0ES=" + str(SHOES) + "_TRGB=" +
                    str(TRGB) + "_curvature=" + str(curvature) + ".pickle", 'rb') as p1:
            loaded_obj = pickle.load(p1)
    else:
        with open("Testing/" + str(freeparams) + "fp_H_Dm_mu_params:_lenses=" + str(lenses) + "_SH0ES=" + str(SHOES) +
                "_TRGB=" + str(TRGB) + "_model" + str(model) + "_curvature=" + str(curvature) + ".pickle", 'rb') as p1:
            loaded_obj = pickle.load(p1)

    z_list, rs_list, H_mean, H_std, Dm_mean, Dm_std, mu_mean, mu_std = loaded_obj
    r_s = np.mean(rs_list)

    # Correct BAO measurements from fiducial r_s to r_s predicted by data
    H_BAO_cor = H_BAO * r_fid / r_s
    Dm_BAO_cor = Dm_BAO * r_s / r_fid

    # Determine upper and lower bounds for the results + standard deviations (shaded regions in plot)
    H_low, H_up = H_mean - H_std, H_mean + H_std
    Dm_low, Dm_up = Dm_mean - Dm_std, Dm_mean + Dm_std
    mu_low, mu_up = mu_mean - mu_std, mu_mean + mu_std

    # Y labels and colours used in the 3 panels
    y_labels = [r"$H(z)$", r"$D_M$", r"$\mu$"]
    colors = ["C0", "C1", "C2"]
    # Midpoints (means), lower and upper points to plot
    means = [H_mean, Dm_mean, mu_mean]
    lower = [H_low, Dm_low, mu_low]
    upper = [H_up, Dm_up, mu_up]
    # Redshift ranges for the 3 panels
    data_z = [z_BAO, z_BAO, z_pan]
    # BAO data points + errors (SN data points are taken from the SN_BAO.py file)
    data_points = [H_BAO_cor, Dm_BAO_cor, mu_pan]
    data_errors = [E_H, E_Dm, E_pan]
    # Markers, sizes and legend labels
    markers = ['*', '*', '.']
    sizes = [12, 12, 5]
    legend_labels=["BAO (BOSS)", "BAO (BOSS)", "SN (Pantheon)"]

    # Plotting parameters
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams["font.size"] = 24
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    # Start plotting: 1st panel shows predictions of H(z), 2nd of comoving distance, and 3rd of distance modulus
    fig1, ax1 = plt.subplots(3, 1, sharex=True, figsize=(12,8))
    fig1.subplots_adjust(hspace=0)
    fig1.suptitle("SN & BAO check", fontsize=20)

    for f in range(3):
        # Plot the mean and standard deviation of the predictions
        ax1[f].plot(z_list, means[f], color=colors[f])
        ax1[f].fill_between(z_list, upper[f], lower[f], color=colors[f], alpha=0.4)
        # Plot the data points (BAO or SN)
        ax1[f].errorbar(data_z[f], data_points[f], yerr=data_errors[f], ms=sizes[f], marker=markers[f], color="black",
                        linestyle='none', label=legend_labels[f])
        # Set the x limits, y labels and legend
        ax1[f].set_xlim(0, 1.7)
        ax1[f].set_ylabel(y_labels[f], labelpad=10)
        ax1[f].legend(loc='lower right', fontsize=12)
        # Set the number of ticks
        M = 3
        yticks = ticker.MaxNLocator(M)
        ax1[f].yaxis.set_major_locator(yticks)

    # Set the x label
    ax1[f].set_xlabel(r"$z$")

    # If save=True: save the files
    if save:
        if model == 4:
            plt.savefig("Figures/H_Dm_mu_model4_Lenses_params:_lenses=" + str(lenses) + "_SH0ES=" + str(SHOES) +
                        "_TRGB=" + str(TRGB) + "_model" + str(model) + "_curvature=" + str(curvature) + ".pdf",
                        bbox_inches='tight')
        else:
            plt.savefig("Figures/H_Dm_mu_" + str(freeparams) + "fp_Lenses_params:_lenses=" + str(lenses) + "_SH0ES="
                        + str(SHOES) + "_TRGB=" + str(TRGB) + "_model" + str(model) + "_curvature=" + str(curvature)
                        + ".pdf", bbox_inches='tight')


# --------------------------------------------------------------------------------------


# Plot the test results

# Corner plot with r_s, H0, q0, j0 (O_k)
plot_corner(model=3, freeparams=4, lenses=False, SHOES=True, TRGB=False, curvature=False, save=True)

# Figure with all the data from the 6 H0LiCOW lenses
plot_lenses(model=3, freeparams=4, lenses=False, SHOES=True, TRGB=False, curvature=False, save=True)

# Figure with H(z), transverse comoving distance and distance modulus (+ data points from SN and BAO)
plot_SN_BAO(model=3, freeparams=4, lenses=False, SHOES=True, TRGB=False, curvature=False, save=True)

plt.show()