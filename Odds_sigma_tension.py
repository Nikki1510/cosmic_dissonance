import numpy as np
import pandas as pd
from scipy import integrate
from scipy import special

# ------------------------------------------------------------------------------------------
# This code computes the odds and number sigma tension between two 2-dimensional data sets.
# See section 3.1 of the Cosmic Dissonance paper for more details on the Gaussian odds indicator.
# ------------------------------------------------------------------------------------------


def gaussian_2d(x, y, mu, cov):
    X = np.array([[x], [y]])
    # Pre-factor (not necessary here):
    # (2*np.pi)**(-1) * np.linalg.det(cov) ** -0.5
    return np.e ** (-0.5 * np.dot(np.dot((X - mu).T, np.linalg.inv(cov)), (X - mu)))[0][0]


def odds(data1, data2):
    """
    Return the odds and the number sigma tension.
    """
    xdata1 = data1[:, 0]
    ydata1 = data1[:, 1]
    xdata2 = data2[:, 0]
    ydata2 = data2[:, 1]

    # Compute covariance matrix and mean of the data
    cov1 = np.cov(xdata1, ydata1)
    cov2 = np.cov(xdata2, ydata2)
    mean1 = np.array([[np.mean(xdata1)], [np.mean(ydata1)]])
    mean2 = np.array([[np.mean(xdata2)], [np.mean(ydata2)]])

    # - - - - Find odds - - - -
    # Calculate integral of the 2D gaussians
    f_orig = lambda y, x: gaussian_2d(x, y, mean1, cov1) * gaussian_2d(x, y, mean2, cov2)
    f_shifted = lambda y, x: gaussian_2d(x, y, mean1, cov1) * gaussian_2d(x, y, mean1, cov2)
    integral_orig = integrate.dblquad(f_orig, 60, 90, lambda x: 120, lambda x: 155)
    integral_shifted = integrate.dblquad(f_shifted, 60, 90, lambda x: 120, lambda x: 155)
    tau = integral_shifted[0] / integral_orig[0]

    # - - - - Convert to number sigma tension - - - -
    # Use regularized incomplete gamma function of Scipy to calculate the enclosed fraction
    fraction = special.gammainc(1, np.log(tau))
    # and then inverse error function to calculate number of sigma
    tension = special.erfinv(fraction) * 2 ** 0.5

    return np.log(tau), tension


# ------------------------------------------------------------------------------------------

# Load Planck (+ extensions) data
planck = np.loadtxt("Chains/chain_planck.txt")
# Load local data set
local = pd.read_csv("Chains/model4_SN+BAO_lenses=True_SH0ES=False_TRGB=False_curvature=False.txt", sep=" ",
                    header=0, dtype=np.float64).values

# Print some results
print("Local data set: H0LiCOW calibration, model 4 (LCDM), flat universe")
print("CMB-based data set: Planck flat LCDM")

tau_tension, sigma_tension = odds(local, planck)
print(" ")
print("Tension between local and CMB-based:")
print("Ln(tau) = ", np.around(tau_tension, 2))
print("Tension = ", np.around(sigma_tension, 2), "sigma")

