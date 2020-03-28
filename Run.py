from Model4 import *
from Model123_3freeparams import *
from Model123_4freeparams import *

"""
This program allows you to run model 1, 2, 3 and 4 from our paper 'Cosmic Dissonance'.
In our work, model 1 has three free parameters (3fp) and model 2 and 3 have four (4fp).

Description of the parameters of the Markov Chain Monte Carlo (MCMC) classes:
- curvature: choose whether O_k is equal to 0.0 (curvature=False), or a free parameter (curvature=True).
- lenses: choose to include the lenses from H0LiCOW in the inference (lenses=True), or not (lenses=False).
- SHOES: choose to include a prior from SH0ES in the inference (SHOES=True), or not (SHOES=False).
- TRGB: choose to include the TRGB measurement from the CCHP (TRGB=True), or not (TRGB=False).
- testing: save a file to compare the best-fit model with the SN, BAO and lensed quasars data points.
    One file contains predictions for H(z), D_m and distance modulus as a function of redshift (in the range z = 0-2).
    Another contains predictions for time-delay distances and angular diameter distances at the redshifts of the lenses.
    These files can be ran with the "Fig_test.py" code, which plots comparison graphs between predictions and data.
    
Comment: don't choose both SHOES=True and TRGB=True, due to partial overlap in their galaxy samples their 
individual measurements should not be combined. Choose one of the two (or neither).

Have fun!
"""

Nburn = 500
Nrun = 500


# Run model 1 (flat Universe):
MCMC_model123_3fp(model=1, curvature=False, lenses=False, SHOES=True, TRGB=False, save=True, testing=True,
                  nburn=Nburn, nruns=Nrun).run()

# Run model 2 (flat Universe)
MCMC_model123_4fp(model=2, curvature=False, lenses=False, SHOES=True, TRGB=False, save=True, testing=True,
                  nburn=Nburn, nruns=Nrun).run()

# Run model 3 (flat Universe)
MCMC_model123_4fp(model=3, curvature=False, lenses=False, SHOES=True, TRGB=False, save=True, testing=True,
                  nburn=Nburn, nruns=Nrun).run()

# Run model 4 (LCDM, flat Universe):
MCMC_model4(curvature=False, lenses=False, SHOES=True, TRGB=False, save=True, testing=True, nburn=Nburn, nruns=Nrun).run()

