import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.optimize as so
import matplotlib.patches as mpatches

# --------------------------------------------------------------------------
# This is the code to generate figure 3 from the Cosmic Dissonance paper.
# --------------------------------------------------------------------------

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['legend.labelspacing'] = 0.6
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.borderpad'] = 0.5


def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level


def density_contour(xdata, ydata, nbins, color, max_level, ax=None, linestyle='solid', linewidth=2, fill=False, **contour_kwargs):
    """
    Code adapted from https://gist.github.com/adrn/3993992
    Create a density contour plot.
    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins : Number of bins along x and y dimension
    color : Color of the contour
    max_level : Determines if the plot shows 1, 2 or 5 sigma contours
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    linestyle : Linestyle for contours
    linewidth : Linewidth of contour lines
    fill : if True, contour is filled
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """

    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins,nbins), normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))

    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
    five_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.999999426697))
    if max_level == 5:
        levels = [five_sigma]
    elif max_level == 2:
        levels = [two_sigma, one_sigma]
    elif max_level == 1:
        levels = [one_sigma]

    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    if fill == True:
        if max_level == 2:
            levels = [two_sigma, 0.1]
        elif max_level == 1:
            levels = [one_sigma, 0.1]
        if ax == None:
            contour = plt.contourf(X, Y, Z, levels=levels, origin="lower", colors=color, **contour_kwargs)
        else:
            contour = ax.contourf(X, Y, Z, levels=levels, origin="lower", colors=color, **contour_kwargs)
        return contour

    if ax == None:
        contour = plt.contour(X, Y, Z, levels=levels, origin="lower", colors=color, linestyles=linestyle,
                              linewidths=linewidth, **contour_kwargs)
    else:
        contour = ax.contour(X, Y, Z, levels=levels, origin="lower", colors=color, linestyles=linestyle,
                             linewidths=linewidth, **contour_kwargs)

    return contour


def figure3(data_SHOES, data_CCHP, save=False):
    """
    Creates coloured contours of the LCDM extensions, with 2 solid gray shapes for SH0ES and CCHP data.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    color_wCDM = "#23a0c2"
    color_planck = "black"
    color_Neff = "#ffab03"
    color_EDE = "#27ad23"
    color_PEDE = "#6c3dd1"

    # Plot the Planck contours
    #density_contour(wCDM_H0, wCDM_rs, 19, color=color_wCDM, max_level=2, ax=ax, linestyle='dotted', linewidth=[2.5, 3])
    density_contour(Neff_H0, Neff_rs, 18, color=color_Neff, max_level=2, ax=ax, linestyle='dashed', linewidth=[2.5, 3])
    #density_contour(EDE_H0, EDE_rs, 20, color=color_EDE, max_level=2, ax=ax, linewidth=[2.5, 3])
    #density_contour(PEDE_H0, PEDE_rs, 18, color=color_PEDE, max_level=2, ax=ax, linewidth=[2.5, 3])
    density_contour(planck[:, 0], planck[:, 1], 18, color=color_planck, max_level=2, ax=ax, linewidth=[2.5, 3])

    # Plot the local data contours
    density_contour(data_CCHP[:, 0], data_CCHP[:, 1], 18, color="#d4d4d4", max_level=2, ax=ax, fill=True)
    density_contour(data_SHOES[:, 0], data_SHOES[:, 1], 18, color="#a3a3a3", max_level=2, ax=ax, fill=True)

    handle1 = mlines.Line2D([], [], color=color_planck, linewidth=3, label=r"$\Lambda$CDM")
    handle2 = mlines.Line2D([], [], color=color_Neff, linewidth=3, linestyle='dashed', label=r"$\Lambda$CDM + $\rm{N}_{\rm{eff}}$")
    handle3 = mlines.Line2D([], [], color=color_EDE, linewidth=3, label="Early DE")
    handle4 = mlines.Line2D([], [], color=color_wCDM, linewidth=3, linestyle='dotted', label=r"$w$CDM")
    handle5 = mlines.Line2D([], [], color=color_PEDE, linewidth=3, label="PEDE")
    handle6 = mpatches.Patch(color="#d4d4d4", label='CCHP + H0LiCOW')
    handle7 = mpatches.Patch(color="#a3a3a3", label='SH0ES + H0LiCOW')
    #plt.legend(handles=[handle1, handle2, handle3, handle4, handle5, handle6, handle7], fontsize=18, loc="lower left")
    plt.legend(handles=[handle1, handle2, handle6, handle7], fontsize=18, loc="lower left")

    plt.xlim(62, 78)
    plt.ylim(126.5, 154)
    plt.xlabel(r"$H_{\rm{0}} \; [\rm{km} \; \rm{s}^{-1} \rm{Mpc}^{-1}]$", fontsize=30, labelpad=8)
    plt.ylabel(r"$r_{\rm{d}} \; [\rm{Mpc}]$", fontsize=30, labelpad=10)
    plt.yticks(size=19)
    plt.xticks(size=19)

    if save:
        plt.savefig("Figures/Figure_3_cosmic_dissonance.pdf", bbox_inches='tight')


# --------------------------------------------------------------------------

# Load Planck + extensions data
planck = np.loadtxt("Chains/chain_planck.txt")

# Extensions:
planck_Neff = np.loadtxt("Chains/chain_LCDM+Neff.txt")
#planck_EDE = np.loadtxt("Chains/chain_EDE.txt")
#planck_wCDM = np.loadtxt("Chains/chain_wCDM.txt")
#planck_PEDE = np.loadtxt("Chains/chain_PEDE.txt")

Neff_H0 = planck_Neff[:, 0]
#EDE_H0 = planck_EDE[:, 3] * 100
#wCDM_H0 = planck_wCDM[:, 0]
#PEDE_H0 = planck_PEDE[:, 3] * 100

Neff_rs = planck_Neff[:, 1]
#EDE_rs = planck_EDE[:, 7]
#wCDM_rs = planck_wCDM[:, 1]
#PEDE_rs = planck_PEDE[:, 5]


# Load local data
chain_SHOES = pd.read_csv("Chains/SN+BAO_lenses=True_SH0ES=True_TRGB=False_model3_curvature=False.txt",
                     sep=" ", header=0, dtype=np.float64).values
chain_CCHP = pd.read_csv("Chains/SN+BAO_lenses=True_SH0ES=False_TRGB=True_model3_curvature=False.txt",
                     sep=" ", header=0, dtype=np.float64).values

# Generate figure 3
figure3(chain_SHOES, chain_CCHP, save=True)

plt.show()