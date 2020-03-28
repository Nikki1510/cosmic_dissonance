from scipy import integrate
import numpy as np

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

# ---- Constants ----

# speed of light in km / s
c = 299792.458

# Sound horizon used in fiducial cosmology (BOSS)
r_fid = 147.78


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

# ---- Functions ----


def ComovingDistance(H_inv, H_0, redshift_, O_k_):
    """
    Function that calculates the comoving distance array from a Hubble parameter array.
    -- Important: the input redshift array should start at 0! --
    """

    Dm = integrate.cumtrapz(H_inv, redshift_, initial=0) * c

    if O_k_ > 0.0:
        Dm = c / H_0 / O_k_ ** 0.5 * np.sinh(O_k_ ** 0.5 * Dm * H_0 / c)
    elif O_k_ < 0.0:
        Dm = c / H_0 / (np.abs(O_k_)) ** 0.5 * np.sin((np.abs(O_k_)) ** 0.5 * Dm * H_0 / c)

    return Dm


def TimeDelayDistance(z_l, z_s, z_array, Dm_array, O_k, H0):
    """
    Function that calculates the time delay distance from the lens and source redshifts and comoving distances.
    """
    Da_array = Dm_array / (1 + z_array)
    Da_l = Da_array[z_array == z_l][0]
    Da_s = Da_array[z_array == z_s][0]
    Dm_l = Dm_array[z_array == z_l][0]
    Dm_s = Dm_array[z_array == z_s][0]
    Da_ls = 1 / (1 + z_s) * ( Dm_s * (1 + O_k * (Dm_l * H0 / c)**2 )**0.5 - Dm_l * (1 + O_k * (Dm_s * H0 / c)**2 )**0.5)
    ddt = (1 + z_l) * Da_l * Da_s / Da_ls
    return ddt


def TimeDelayDistance_chains(z_l, z_s, Dm_l_List, Dm_s_List, O_k_List, H0_List):
    """
    For use in the testing function. Returns chains of ddt and da measurements.
    """
    Da_l_List = Dm_l_List / (1 + z_l)
    Da_s_List = Dm_s_List / (1 + z_s)
    Da_ls_List = []
    for i in range(len(Da_l_List)):
        Da_ls_List.append(1 / (1 + z_s) * (Dm_s_List[i] * (1 + O_k_List[i] * (Dm_l_List[i] * H0_List[i] / c)**2 )**0.5 -
                                            Dm_l_List[i] * (1 + O_k_List[i] * (Dm_s_List[i] * H0_List[i] / c)**2 )**0.5))

    ddt_List = (1 + z_l) * Da_l_List * Da_s_List / Da_ls_List
    return [ddt_List, Da_l_List]


def compute_LLH(R, sigma):
    """
    Function that calculates the likelihood for any probes.
    :param R: Difference between the predicted and observed value.
    :param sigma: Uncertainty. Can be either in the form of an array with error values or an inverse covariance matrix.
    :return: The likelihood
    """
    # Check if sigma is the inverse covariance matrix or an error array
    if sigma.ndim == 1:
        LLH = - 0.5 * sum(R**2 / sigma ** 2)
    else:
        LLH = - 0.5 * np.dot(np.dot(R.conj().T, sigma), R)
    return LLH
