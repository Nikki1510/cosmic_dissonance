from Functions import *
from Lenses import *
from SN_BAO import *


# ----------------------------------------------------------------------------------

# --- Prior ---

def Ln_Prior(theta):
    """
    :param theta: The guesses for the free parameters.
    :return: 0.0 (becomes 1) if the condition is satisfied, - inf (becomes 0) otherwise.
    """
    # r_s between 0 and 200
    if not 80 < theta[1] < 200:
        return - np.inf

    # O_k between -1 and 1
    if len(theta) == 6:             # If curvature = True
        if not -1 < theta[-1] < 1:
            return - np.inf

    return 0.0

# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------

# --- Likelihood ---


def Ln_Likelihood(theta, model, curvature, lenses, SHOES, TRGB):
    """
    Function to calculate the total likelihood.
    """

    # Unpack the free parameters

    if curvature:
        M, r_s, c0, c1, c2, O_k = theta
    else:
        M, r_s, c0, c1, c2 = theta
        O_k = 0.0

    # ---------------------------------

    # Make array of combined redshifts
    z_array = np.linspace(0, 1.8, 1000)
    z_lenses = np.array([HE0435_ddt.zlens, HE0435_ddt.zsource, RXJ1131.zlens, RXJ1131.zsource, B1608_ddt.zlens,
                         B1608_ddt.zsource, J1206_zlens, J1206_zsource, PG1115_zlens, PG1115_zsource, WFI2033_zlens,
                         WFI2033_zsource, 2.34, 2.35])
    z_total = np.concatenate((SN_data[0], BAO_data[0], z_lenses, z_array))
    z_total.sort()

    # ---------------------------------

    # Obtain H and Dm

    # Model 1
    if model == 1:
        expansion_H = c0 + c1 * z_total + c2 * z_total ** 2
        expansion_Dm = ComovingDistance(1 / expansion_H, c0, z_total, O_k)
        expansion_Dl = expansion_Dm * (1 + z_total)

    # Model 2
    elif model == 2:
        Z = np.log10(1 + z_total)
        expansion_Dl = c0 * Z + c1 * Z ** 2 + c2 * Z ** 3
        expansion_Dm = expansion_Dl / (1 + z_total)
        Dm_dz = np.diff(expansion_Dm) / np.diff(z_total)
        Dm_dz = np.insert(Dm_dz, 0, Dm_dz[0])
        c0 = c * np.log(10) / c0
        expansion_H = c / Dm_dz * (1 + (c0 / c) ** 2 * O_k * expansion_Dm ** 2) ** 0.5

    # Model 3
    else:
        Z = z_total / (1 + z_total)
        Z_dz = 1 / (1 + z_total) - z_total / (1 + z_total) ** 2
        Dm_dz = (c0 + 2 * c1 * Z + 3 * c2 * Z ** 2) * Z_dz
        expansion_Dm = c0 * Z + c1 * Z ** 2 + c2 * Z ** 3
        expansion_Dl = expansion_Dm * (1 + z_total)
        c0 = c / c0
        expansion_H = c / Dm_dz * (1 + (c0 / c) ** 2 * O_k * expansion_Dm ** 2) ** 0.5

    # ---------------------------------

    # - - - - - BAO - - - - -

    # Get H for BAO
    BAO_expansion_H = expansion_H[np.in1d(z_total, BAO_data[0])]
    expansion_H_fiducial = BAO_expansion_H * r_s / r_fid

    # Get Dm for BAO
    BAO_expansion_Dm = expansion_Dm[np.in1d(z_total, BAO_data[0])]
    expansion_Dm_fiducial = BAO_expansion_Dm * r_fid / r_s

    # Calculate the ln-likelihood
    r_BAO = np.concatenate((BAO_data[1] - expansion_Dm_fiducial, BAO_data[2] - expansion_H_fiducial))
    LLH_BAO = compute_LLH(r_BAO, BAO_invcov)

    # - - - - - SN - - - - -

    # Get the distance modulus for SN
    SN_expansion_Dl = expansion_Dl[np.in1d(z_total, SN_data[0])]
    SN_expansion_mu = M + 5 * np.log10(SN_expansion_Dl)

    # Calculate the ln-likelihood
    r_SN = SN_data[1] - SN_expansion_mu
    LLH_SN = compute_LLH(r_SN, SN_invcov)

    # - - - - - LENSES - - - - -

    if lenses:
        # Calculate ddt
        ddt_HE0435 = TimeDelayDistance(HE0435_ddt.zlens, HE0435_ddt.zsource, z_total, expansion_Dm, O_k, c0)
        ddt_RXJ1131 = TimeDelayDistance(RXJ1131_zlens, RXJ1131_zsource, z_total, expansion_Dm, O_k, c0)
        ddt_B1608 = TimeDelayDistance(B1608_ddt.zlens, B1608_ddt.zsource, z_total, expansion_Dm, O_k, c0)
        ddt_J1206 = TimeDelayDistance(J1206_zlens, J1206_zsource, z_total, expansion_Dm, O_k, c0)
        ddt_PG1115 = TimeDelayDistance(PG1115_zlens, PG1115_zsource, z_total, expansion_Dm, O_k, c0)
        ddt_WFI2033 = TimeDelayDistance(WFI2033_zlens, WFI2033_zsource, z_total, expansion_Dm, O_k, c0)

        # For 4 lenses: calculate dd
        dd_J1206 = expansion_Dm[z_total == J1206_zlens][0] / (1 + J1206_zlens)
        dd_PG1115 = expansion_Dm[z_total == PG1115_zlens][0] / (1 + PG1115_zlens)
        dd_RXJ1131 = expansion_Dm[z_total == RXJ1131_zlens][0] / (1 + RXJ1131_zlens)
        dd_B1608 = expansion_Dm[z_total == B1608_ddt.zlens][0] / (1 + B1608_ddt.zlens)

        LLH_lenses = KDE_LLH_RXJ1131(ddt_RXJ1131, dd_RXJ1131) + KDE_LLH_PG1115(ddt_PG1115, dd_PG1115) + \
                     KDE_LLH_J1206(ddt_J1206, dd_J1206) + KDE_LLH_WFI2033(ddt_WFI2033) + \
                     B1608_ddt.sklogn_analytical_likelihood(ddt_B1608) + B1608_dd.sklogn_analytical_likelihood(dd_B1608) + \
                     KDE_LLH_HE0435(ddt_HE0435)

    # - - - - - SH0ES / TRGB - - - - -

    r_SHOES = c0 - 74.03  # ± 1.42
    LLH_SHOES = - 0.5 * r_SHOES ** 2 / 1.42 ** 2

    r_TRGB = c0 - 69.8  # ±2.0
    LLH_TRGB = - 0.5 * r_TRGB ** 2 / 2.0 ** 2

    # - - - - - - - - - - - - - - - - -

    total_LLH = LLH_SN + LLH_BAO

    if lenses:
        total_LLH += LLH_lenses
    if SHOES:
        total_LLH += LLH_SHOES
    if TRGB:
        total_LLH += LLH_TRGB

    if np.isnan(total_LLH):
        return - np.inf

    return total_LLH


# ----------------------------------------------------------------------------------

# --- Posterior ---


def Ln_Posterior(theta, model, curvature, lenses, SHOES, TRGB):
    """
    :param theta: Contains the estimated values for the free parameters.
    :return: The posterior for the guessed parameters theta, given the data.
    """
    return Ln_Prior(theta) + Ln_Likelihood(theta, model, curvature, lenses, SHOES, TRGB)
