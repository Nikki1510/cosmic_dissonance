import numpy as np


# Load the relevant data

# SN
z_pan = np.loadtxt("Data/pantheon_data.txt", unpack=True)[1]
mu_pan = np.loadtxt("Data/pantheon_data.txt", unpack=True)[4]
cov_pan = (np.loadtxt("Data/pantheon_covariancematrix.txt")).reshape(40, 40)
stat_pan = np.loadtxt("Data/pantheon_data.txt", unpack=True)[5]
cov_pan += np.diag(stat_pan ** 2)
invcov_pan = np.linalg.inv(cov_pan)
E_pan = (np.diagonal(cov_pan))**0.5


# BAO
z_BAO, Dm_BAO, E_Dm_stat, E_Dm_sys, H_BAO, E_H_stat, E_H_sys = np.loadtxt("Data/BAOdata.txt", unpack=True)
E_H = (E_H_sys**2 + E_H_stat**2)**0.5
E_Dm = (E_Dm_sys**2 + E_Dm_stat**2)**0.5
cov_BAO = (np.loadtxt("Data/BAO_covariancematrix_2.txt")).reshape(6, 6)
invcov_BAO = np.linalg.inv(cov_BAO)


# Put data in right format for the MCMC
SN_data = np.array([z_pan, mu_pan])
SN_invcov = invcov_pan
BAO_data = np.array([z_BAO, Dm_BAO, H_BAO])
BAO_invcov = invcov_BAO