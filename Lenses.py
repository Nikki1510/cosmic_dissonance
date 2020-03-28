#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import pandas as pd
import scipy.stats as stats
import random

# ----------------------------------------------------------------------------------
"""
The skewed lognormal functions in this python script used to calculate the likelihoods of the lensed quasars
are all obtained from the public H0LiCOW jupyter notebook: 
http://shsuyu.github.io/H0LiCOW/site/notebooks/H0_from_lenses.html
"""
# ----------------------------------------------------------------------------------

class StrongLensSystem(object):
    """
    This is a parent class, common to all lens modeling code outputs.
    It stores the "physical" parameters of the lens (name, redshifts, ...)
    """

    def __init__(self, name, zlens, zsource, longname=None):
        self.name = name
        self.zlens = zlens
        self.zsource = zsource
        self.longname = longname

    def __str__(self):
        return "%s\n\tzl = %f\n\tzs = %f" % (self.name, self.zlens, self.zsource)


class GLEELens(StrongLensSystem):
    """
    This class takes the output of GLEE (Ddt distribution) from which it evaluates the likelihood of a Ddt
    (in Mpc) predicted in a given cosmology.

    The default likelihood follows a skewed log-normal distribution. No other likelihoods have been implemented so far.
    """

    def __init__(self, name, zlens, zsource,
                 loglikelihood_type="sklogn_analytical",
                 mu=None, sigma=None, lam=None, explim=100.,
                 longname=None):

        StrongLensSystem.__init__(self, name=name, zlens=zlens, zsource=zsource, longname=longname)
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.zlens = zlens
        self.zsource = zsource
        self.explim = explim
        self.loglikelihood_type = loglikelihood_type
        self.init_loglikelihood()

    def sklogn_analytical_likelihood(self, ddt):
        """
        Evaluates the likelihood of a time-delay distance ddt (in Mpc) against the model predictions,
        using a skewed log-normal distribution.
        """
        # skewed lognormal distribution with boundaries
        if (ddt < self.lam) or ((-self.mu + math.log(ddt - self.lam)) ** 2 / (2. * self.sigma ** 2) > self.explim):
            return -np.inf
        else:
            llh = math.exp(-((-self.mu + math.log(ddt - self.lam)) ** 2 / (2. * self.sigma ** 2))) / (
                        math.sqrt(2 * math.pi) * (ddt - self.lam) * self.sigma)

            if np.isnan(llh):
                return -np.inf
            else:
                return np.log(llh)

    def init_loglikelihood(self):
        if self.loglikelihood_type == "sklogn_analytical":
            self.loglikelihood = self.sklogn_analytical_likelihood
        else:
            assert ValueError("unknown keyword: %s" % self.loglikelihood_type)
        # if you want to implement other likelihood estimators, do it here


# ----------------------------------------------------------------------------------

"""Create the lenses objects"""

HE0435_ddt = GLEELens(name="HE0435", longname="HE0435 (Wong+2017)", zlens=0.4546, zsource=1.693,
                     mu=7.57930024e+00, sigma=1.03124167e-01, lam=6.53901645e+02,
                     loglikelihood_type="sklogn_analytical"
                    )

RXJ1131 = GLEELens(name="RXJ1131", longname="RXJ1131 (Suyu+2014)", zlens=0.295, zsource=0.654,
                      mu=6.4682, sigma=0.20560, lam=1388.8,
                      loglikelihood_type="sklogn_analytical"
                     )

B1608_ddt = GLEELens(name="B1608", longname="B1608 (Suyu+2010)", zlens=0.6304, zsource=1.394,
                    mu=7.0531390, sigma=0.2282395, lam=4000.0,
                    loglikelihood_type="sklogn_analytical"
                   )

B1608_dd = GLEELens(name="B1608", longname="B1608 (Suyu+2010)", zlens=0.6304, zsource=1.394,
                    mu=6.79671, sigma=0.1836, lam=334.2,
                    loglikelihood_type="sklogn_analytical"
                   )

# ----------------------------------------------------------------------------------
# For the other lenses that don't have skweded lognormal functions yet: construct KDEs of their posterior chains.

# Redshifts of lenses + sources
J1206_zlens = 0.745
J1206_zsource = 1.789

PG1115_zlens = 0.311
PG1115_zsource = 1.722

WFI2033_zlens = 0.6575
WFI2033_zsource = 1.662

RXJ1131_zlens = 0.295
RXJ1131_zsource = 0.654

# Make lists for z_l and z_s, in the following order: RXJ1131, PG1115, J1206, B1608, WFI2033, HE0435
z_l = [RXJ1131_zlens, PG1115_zlens, J1206_zlens, B1608_ddt.zlens, WFI2033_zlens, HE0435_ddt.zlens]
z_s = [RXJ1131_zsource, PG1115_zsource, J1206_zsource, B1608_ddt.zsource, WFI2033_zsource, HE0435_ddt.zsource]

# Load the posterior points
data_J1206 = pd.read_csv("Data/lens_J1206_ddt_dd.txt", sep=",", header=0, dtype=np.float64).values
data_PG1115 = pd.read_csv("Data/PG1115_AO+HST_Dd_Ddt.dat", sep=" ", header=0, dtype=np.float64).values[:,0:2]
data_WFI2033 = pd.read_csv("Data/wfi2033_dt_bic.dat", sep=",", header=0, dtype=np.float64).values
data_RXJ1131 = pd.read_csv("Data/RXJ1131_AO+HST_Dd_Ddt.dat", sep=" ", header=0, dtype=np.float64).values[:,0:2]
data_HE0435 = pd.read_csv("Data/HE0435_Ddt_AO+HST.dat", sep=" ", header=0, dtype=np.float64).values

# For some lenses: take random samples from the posterior chains, in order to reduce the computation time.
# The samples should be large enough to accurately represent the full chains.
sample_length = 650000

data_P = np.array(random.sample(list(data_PG1115), sample_length)).T
data_W = np.array(random.sample(list(data_WFI2033), sample_length))

# Make kernels
kernel_J = stats.gaussian_kde(data_J1206.T, bw_method=0.3)
kernel_P = stats.gaussian_kde(data_P, bw_method=0.3)
kernel_R = stats.gaussian_kde(data_RXJ1131.T, bw_method=0.3)
kernel_W = stats.gaussian_kde(data_W[:, 0], bw_method=0.3, weights=data_W[:, 1])
kernel_H = stats.gaussian_kde(data_HE0435.T, bw_method=0.3)


def KDE_LLH_J1206(ddt, dd):
    """
    Input: ddt (time delay distance) and dd (angular diameter distance).
    Output: Ln-likelihood.
    """
    return np.log(kernel_J.evaluate([ddt, dd])[0])

def KDE_LLH_PG1115(ddt, dd):
    """
    Input: ddt (time delay distance) and dd (angular diameter distance).
    Output: Ln-likelihood.
    """
    return np.log(kernel_P.evaluate([dd, ddt])[0])

def KDE_LLH_RXJ1131(ddt, dd):
    """
    Input: ddt (time delay distance) and dd (angular diameter distance).
    Output: Ln-likelihood.
    """
    return np.log(kernel_R.evaluate([dd, ddt])[0])

def KDE_LLH_WFI2033(ddt):
    """
    Input: ddt (time delay distance) and dd (angular diameter distance).
    Output: Ln-likelihood.
    """
    return np.log(kernel_W.evaluate(ddt)[0])

def KDE_LLH_HE0435(ddt):
    """
    Input: ddt (time delay distance) and dd (angular diameter distance).
    Output: Ln-likelihood.
    """
    return np.log(kernel_H.evaluate(ddt)[0])

