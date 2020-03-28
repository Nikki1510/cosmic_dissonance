import time
import emcee
import pickle
from uncertainties import ufloat
from Posterior_model123_3freeparams import *


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


class MCMC_model123_3fp:

    def __init__(self, model, curvature=False, lenses=False, SHOES=False, TRGB=False, save=True, testing=True,
                 nwalkers=100, nburn=2000, nruns=2000):
        """
        Initializing the important characteristics of the expansion and which data sets to use.
        :param model: choose which polynomial parametrization model to use: input can either be 1, 2 or 3.
        For specifics about the models, please see our paper: https://arxiv.org/pdf/1909.07986.pdf
        :param curvature: choose whether O_k is equal to 0.0 (curvature=False), or a free parameter (curvature=True).
        :param lenses: choose to include the lenses from H0LiCOW in the inference (lenses=True), or not (lenses=False).
        :param SHOES: choose to include a prior from SH0ES in the inference (SHOES=True), or not (SHOES=False).
        :param TRGB: choose to include the TRGB measurement from the CCHP (TRGB=True), or not (TRGB=False).
        :param save: saves the output and chains of the results.
        :param testing: saves a file to compare the best-fit model with the SN, BAO and lensed quasars data points.
        :param nwalkers: Specify number of walkers in the MCMC
        :param nburn: Specify number of burn-in runs. These runs will not contribute to the final results.
        :param nruns: Specify number of runs of the MCMC.
        """
        self.model = model
        self.curvature = curvature
        self.lenses = lenses
        self.SHOES = SHOES
        self.TRGB = TRGB
        self.save = save
        self.testing = testing
        self.nwalkers = nwalkers
        self.nburn = nburn
        self.nruns = nruns

        # Number of free parameters = nparam (3) + 1 (M from SN) + 1 (Rs from BAO)
        self.ndim = 5
        # If curvature=True: add another free parameter
        if curvature: self.ndim += 1

    def get_params(self):
        """
        Get the chains of the expansion parameters and put them in a best-fit parameter list.
        """

        self.M = ufloat(np.mean(self.flatchain[:, 0]), np.std(self.flatchain[:, 0]))
        self.r_s = ufloat(np.mean(self.flatchain[:, 1]), np.std(self.flatchain[:, 1]))
        print("r_s = {:.3f}".format(self.r_s))
        self.rs_list = self.flatchain[:, 1]
        theta = [self.M.n, self.r_s.n]

        self.c0 = ufloat(np.mean(self.flatchain[:, 2]), np.std(self.flatchain[:, 2]))
        self.c1 = ufloat(np.mean(self.flatchain[:, 3]), np.std(self.flatchain[:, 3]))
        self.c2 = ufloat(np.mean(self.flatchain[:, 4]), np.std(self.flatchain[:, 4]))
        theta.append(self.c0.n)
        theta.append(self.c1.n)
        theta.append(self.c2.n)

        if self.curvature:
            Ok_list = self.flatchain[:, -1]
            self.O_k = ufloat(np.mean(self.flatchain[:, -1]), np.std(self.flatchain[:, -1]))
            print("O_k = {:.3f}".format(self.O_k))
            theta.append(self.O_k.n)
        else:
            Ok_list = np.zeros_like(self.rs_list)

        # -------------------------

        if self.model == 1:
            H0rs = self.flatchain[:, 2] * self.flatchain[:, 1]
            H0_list = self.flatchain[:, 2]
            q0_list = self.flatchain[:, 3] / H0_list - 1
            j0_list = 2 * self.flatchain[:, 4] / H0_list + q0_list ** 2

        elif self.model == 2:
            H0rs = c * np.log(10) / self.flatchain[:, 2] * self.flatchain[:, 1]
            H0_list = c * np.log(10) / self.flatchain[:, 2]
            q0_list = 2 - 2 * H0_list * self.flatchain[:, 3] / (c * np.log(10)**2)
            j0_list = 3 - 2 * q0_list + 3 * q0_list**2 - 6 * H0_list * self.flatchain[:, 4] / (c * np.log(10)**3) + Ok_list

        else:
            H0rs = c / self.flatchain[:, 2] * self.flatchain[:, 1]
            H0_list = c / self.flatchain[:, 2]
            q0_list = 1 - 2 * H0_list * self.flatchain[:, 3] / c
            j0_list = - 6 * H0_list * self.flatchain[:, 4] / c + 2 - 2 * q0_list + 3 * q0_list ** 2 + Ok_list

        # -------------------------

        self.H0_list = H0_list
        self.Ok_list = Ok_list
        self.H0 = ufloat(np.mean(H0_list), np.std(H0_list))
        self.H0rs = ufloat(np.mean(H0rs), np.std(H0rs))
        self.q0 = ufloat(np.mean(q0_list), np.std(q0_list))
        self.j0 = ufloat(np.mean(j0_list), np.std(j0_list))

        print("H0 r_s = {:.3f}".format(self.H0rs))
        print("H0 = {:.3f}".format(self.H0))
        print("q0 = {:.3f}".format(self.q0))
        print("j0 = {:.3f}".format(self.j0))
        print("M = {:.3f}".format(self.M))

        if self.curvature:
            results = np.array([H0_list, self.rs_list, q0_list, j0_list, Ok_list]).T
        else:
            results = np.array([H0_list, self.rs_list, q0_list, j0_list]).T

        return theta, results

    def test_expansion(self):

        # Create lists with values for H, Dm and mu (+ uncertainties), from redshift 0-2.
        nsamples = self.nwalkers * self.nruns
        z_range = np.linspace(0, 2, 500)
        z_list = np.concatenate((z_range, z_l, z_s))
        z_list.sort()

        # Initialize empty matrices
        H_matrix = np.zeros((nsamples, len(z_list)))
        Dm_matrix = np.zeros((nsamples, len(z_list)))
        mu_matrix = np.zeros((nsamples, len(z_list)))
        lenses_list = []

        # ------------------------------------------------------------------------------------------------
        if self.model == 1:

            # Make function for H(z)
            def H_func(z_, c0, c1, c2):
                return c0 + c1 * z_ + c2 * z_ ** 2

            for n in range(nsamples):
                # Calculate distances and Hubble parameter
                H = H_func(z_list, self.flatchain[n, 2], self.flatchain[n, 3], self.flatchain[n, 4])
                Dm = ComovingDistance(1 / H, self.H0_list[n], z_list, self.Ok_list[n])
                Dl = Dm * (1 + z_list)
                mu = self.flatchain[n, 0] + 5 * np.log10(Dl)
                # Fill matrices
                H_matrix[n] = H
                Dm_matrix[n] = Dm
                mu_matrix[n] = mu

        # ------------------------------------------------------------------------------------------------
        elif self.model == 2:

            # Make function for Dl(z)
            def Dl_func(z_, c0, c1, c2):
                return c0 * z_ + c1 * z_ ** 2 + c2 * z_ ** 3

            for n in range(nsamples):
                # Calculate distances and Hubble parameter
                Z = np.log10(1 + z_list)
                Dl = Dl_func(Z, self.flatchain[n, 2], self.flatchain[n, 3], self.flatchain[n, 4])
                Dm = Dl / (1 + z_list)
                Dm_Dz = np.diff(Dm) / np.diff(z_list)
                Dm_Dz = np.insert(Dm_Dz, 0, Dm_Dz[0])
                H = c / Dm_Dz * (1 + (self.H0_list[n] / c) ** 2 * self.Ok_list[n] * Dm ** 2) ** 0.5
                mu = self.flatchain[n, 0] + 5 * np.log10(Dl)
                # Fill matrices
                H_matrix[n] = H
                Dm_matrix[n] = Dm
                mu_matrix[n] = mu

        # ------------------------------------------------------------------------------------------------
        else:

            # Make function for Dm(z)
            def Dm_func(z_, c0, c1, c2):
                return c0 * z_ + c1 * z_ ** 2 + c2 * z_ ** 3

            for n in range(nsamples):
                # Calculate distances and Hubble parameter
                Z = z_list / (1 + z_list)
                Dm = Dm_func(Z, self.flatchain[n, 2], self.flatchain[n, 3], self.flatchain[n, 4])
                Dl = Dm * (1 + z_list)
                Dm_Dz = np.diff(Dm) / np.diff(z_list)
                Dm_Dz = np.insert(Dm_Dz, 0, Dm_Dz[0])
                H = c / Dm_Dz * (1 + (self.H0_list[n] / c) ** 2 * self.Ok_list[n] * Dm ** 2) ** 0.5
                mu = self.flatchain[n, 0] + 5 * np.log10(Dl)
                # Fill matrices
                H_matrix[n] = H
                Dm_matrix[n] = Dm
                mu_matrix[n] = mu
        # ------------------------------------------------------------------------------------------------

        # Calculate arrays with mean and std of H, Dm and mu
        H_mean = np.mean(H_matrix, axis=0)
        H_std = np.std(H_matrix, axis=0)
        Dm_mean = np.mean(Dm_matrix, axis=0)
        Dm_std = np.std(Dm_matrix, axis=0)
        mu_mean = np.mean(mu_matrix, axis=0)
        mu_std = np.std(mu_matrix, axis=0)


        # Calculate time delay distance and angular diameter distance for every lens.
        for L in range(6):
            lenses_list.append(TimeDelayDistance_chains(z_l[L], z_s[L], (Dm_matrix[:, z_list == z_l[L]]).ravel(),
                             (Dm_matrix[:, z_list == z_s[L]]).ravel(), self.Ok_list, self.H0_list))

        # Save files
        with open("Testing/3fp_H_Dm_mu_params:_lenses=" + str(self.lenses) + "_SH0ES=" + str(self.SHOES) + "_TRGB=" +
                str(self.TRGB) + "_model" + str(self.model) + "_curvature=" + str(self.curvature) + ".pickle", 'wb') as p1:
            pickle.dump(np.array([z_list, self.rs_list, H_mean, H_std, Dm_mean, Dm_std, mu_mean, mu_std]), p1)
        with open("Testing/3fp_Lenses_params:_lenses=" + str(self.lenses) + "_SH0ES=" + str(self.SHOES) + "_TRGB=" +
                str(self.TRGB) + "_model" + str(self.model) + "_curvature=" + str(self.curvature) + ".pickle", 'wb') as p2:
            pickle.dump(lenses_list, p2)

        # ------------------------------------------------------------------------------------------------------

    def output_file(self):
        """
        If save = True: write all the print statements to an output file.
        """

        # Clean the file and write the header
        print("Model = " + str(self.model), file=open("Output/3fp_SN+BAO_lenses=" + str(self.lenses) + "_SH0ES=" + str(self.SHOES)
            + "_TRGB=" + str(self.TRGB) + "_model" + str(self.model) + "_curvature=" + str(self.curvature) + ".txt", "w"))

        # Define the file to write output to
        f = open("Output/3fp_SN+BAO_lenses=" + str(self.lenses) + "_SH0ES=" + str(self.SHOES) + "_TRGB=" + str(self.TRGB) +
                      "_model" + str(self.model) + "_curvature=" + str(self.curvature) + ".txt", "a")

        # Continue writing the header
        print("Curvature = ", self.curvature, file=f)
        print("Data sets: SN, BAO, lenses = " + str(self.lenses) + ", SH0ES = " + str(self.SHOES) + ", "
        "TRGB = " + str(self.TRGB), file=f)
        print("Nwalkers = ", self.nwalkers,", nburn = ", self.nburn,", nruns = ", self.nruns, file=f)
        print("Free parameters: 3", file=f)

        # Write the output
        print(" ", file=f)
        if self.curvature:
            print("O_k = {:.3f}".format(self.O_k), file=f)
        print("r_s = {:.3f}".format(self.r_s), file=f)
        print("H0 r_s = {:.3f}".format(self.H0rs), file=f)
        print("H0 = {:.3f}".format(self.H0), file=f)
        print("q0 = {:.3f}".format(self.q0), file=f)
        print("j0 = {:.3f}".format(self.j0), file=f)
        print("M = {:.3f}".format(self.M), file=f)
        print("LLH = ", self.LLH, file=f)
        print("BIC = ", self.BIC, file=f)
        print(" ", file=f)
        print("Time elapsed: {:.2f}".format(self.duration / 60), "min", file=f)

        # Close the file
        f.close()

    def run(self):
        """
        Run emcee.
        """
        print("Start MCMC, model = ", self.model)

        # Start the timer
        start = time.time()

        # For theta: Define a random starting position for each of the walkers [nwalkers, ndim].
        M_init = np.random.uniform(0, 30, self.nwalkers)
        rs_init = np.random.uniform(100, 150, self.nwalkers)
        Ok_init = np.random.uniform(-0.45, 0.45, self.nwalkers)

        # For expansion in Hubble parameter:
        if self.model == 1:
            param_init1 = np.random.uniform(25, 100, self.nwalkers)
            param_init2 = np.random.uniform(0, 100, self.nwalkers)
            param_init3 = np.random.uniform(0, 100, self.nwalkers)

        # For expansion in distance: need larger starting values for c0, c1 and c2.
        elif self.model == 2:
            param_init1 = np.random.uniform(9000, 11000, self.nwalkers)
            param_init2 = np.random.uniform(25000, 30000, self.nwalkers)
            param_init3 = np.random.uniform(20000, 40000, self.nwalkers)

        else:
            param_init1 = np.random.uniform(4000, 4500, self.nwalkers)
            param_init2 = np.random.uniform(3000, 3500, self.nwalkers)
            param_init3 = np.random.uniform(2000, 2500, self.nwalkers)

        # Define starting positions
        if self.curvature:
            p0 = np.array([M_init, rs_init, param_init1, param_init2, param_init3, Ok_init]).T
        else:
            p0 = np.array([M_init, rs_init, param_init1, param_init2, param_init3]).T

        # Set up the sampler
        Ln_Post = Ln_Posterior
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, Ln_Posterior,
                                        args=[self.model, self.curvature, self.lenses, self.SHOES, self.TRGB])

        # Run the burn-in phase.
        print("Burn-in phase")
        pos = sampler.run_mcmc(p0, self.nburn)[0]
        sampler.reset()

        # Now run the MCMC with improved initial positions.
        print("Start with the MCMC")
        print(" ")
        pos2 = sampler.run_mcmc(pos, self.nruns)[0]

        # Collect the results
        self.chain = sampler.chain
        self.flatchain = sampler.flatchain

        # Get the expansion parameters and predicted arrays of H, Da and mu.
        Theta, Results = MCMC_model123_3fp.get_params(self)

        # Get the BIC value
        self.LLH = Ln_Post(Theta, self.model, self.curvature, self.lenses, self.SHOES, self.TRGB)
        K = self.ndim
        N = len(SN_data[0]) + 2 * len(BAO_data[0])
        if self.lenses: N += 10
        if self.SHOES: N += 1
        if self.TRGB: N += 1
        self.BIC = np.log(N) * K - 2 * self.LLH
        print("LLH = ", self.LLH)
        print("BIC = ", self.BIC)

        # Stop the timer
        end = time.time()
        self.duration = end - start
        print(" ")
        print("Time elapsed: {:.2f}".format(self.duration/60), "min")

        # Perform tests to compare the expansion to the data points
        if self.testing:
            MCMC_model123_3fp.test_expansion(self)

        # Save the results
        if self.save:
            MCMC_model123_3fp.output_file(self)
            np.savetxt("Chains/3fp_SN+BAO_lenses=" + str(self.lenses) + "_SH0ES=" + str(self.SHOES) + "_TRGB=" +
                str(self.TRGB) + "_model" + str(self.model) + "_curvature=" + str(self.curvature) + ".txt", Results)


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

