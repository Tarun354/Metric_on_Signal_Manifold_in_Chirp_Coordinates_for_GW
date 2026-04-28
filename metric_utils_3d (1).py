import numpy as np
import pycbc
import matplotlib.pyplot as plt
from pycbc.psd import aLIGOAPlusDesignSensitivityT1800042
from pycbc.waveform import get_fd_waveform, get_waveform_filter_length_in_time as chirplen
from pycbc.filter.matchedfilter import overlap, match
from tqdm import tqdm
from pycbc.conversions import mass1_from_tau0_tau3, mass2_from_tau0_tau3, tau0_from_mass1_mass2, tau3_from_mass1_mass2,eta_from_mass1_mass2
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
from pycbc.types.frequencyseries import FrequencySeries
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from pycbc.conversions import mass1_from_mtotal_eta, mass2_from_mtotal_eta, chi_eff
from pycbc.conversions import tau0_from_mtotal_eta, tau3_from_mtotal_eta,  eta_from_mass1_mass2
from pycbc.waveform.spa_tmplt import findchirp_chirptime, spa_length_in_time
from pycbc.conversions import mchirp_from_mass1_mass2, mass1_from_tau0_tau3, mass2_from_tau0_tau3, tau0_from_mass1_mass2, tau3_from_mass1_mass2
import lal


# In[3]:


# mass1_from_mtotal_eta(mtotal, eta), mass2_from_mtotal_eta(mtotal, eta), tau0_from_mtotal_eta(mtotal, eta, f_lower),
# tau3_from_mtotal_eta(mtotal, eta, f_lower)


# In[4]:


def get_wf(params, Nf):
    try:
        if params["mass1"] <=0 or params["mass2"]<= 0:
            return None
        
        wf, _ = get_fd_waveform(template = None, **params)
        
        if len(wf) != Nf:
            wf.resize(Nf)
        return wf
    
    except Exception as e:
        return None


# neccacary functions
def velocity_to_frequency(v, M):
    """ Calculate the gravitational-wave frequency from the
    total mass and invariant velocity.

    Parameters
    ----------
    v : float
        Invariant velocity
    M : float
        Binary total mass

    Returns
    -------
    f : float
        Gravitational-wave frequency
    """
    return v**(3.0) / (M * lal.MTSUN_SI * lal.PI)

def f_schwarzchild_isco(M):
    """
    Innermost stable circular orbit (ISCO) for a test particle
    orbiting a Schwarzschild black hole

    Parameters
    ----------
    M : float or numpy.array
        Total mass in solar mass units

    Returns
    -------
    f : float or numpy.array
        Frequency in Hz
    """

    return velocity_to_frequency((1.0/6.0)**(0.5), M)


def m1_m2f_theta03(theta0, theta3, flow):
    m1 = mass1_from_tau0_tau3(theta0/(2*np.pi*flow), theta3/(2*np.pi*flow), flow)
    m2 = mass2_from_tau0_tau3(theta0/(2*np.pi*flow), theta3/(2*np.pi*flow), flow)
    return m1, m2


# In[6]:


def theta03(m1, m2, flow):
    theta0 = 2*np.pi * flow * tau0_from_mass1_mass2(m1, m2, flow)
    theta3 = 2*np.pi * flow * tau3_from_mass1_mass2(m1, m2, flow)
    return theta0, theta3


# In[7]:


def theta3s_m_s(m1, m2, s1z, s2z, flow, eps=1e-12):
    eta = eta_from_mass1_mass2(m1, m2)
    delta = (m1 - m2)/(m1+m2)
    chi_s = 0.5*(s1z+s2z)
    chi_a = 0.5*(s1z-s2z)
    chi_r = chi_s + delta*chi_a - (76/113)*eta*chi_s 
    theta3 = 2*np.pi * flow * tau3_from_mass1_mass2(m1, m2, flow)
    theta3s = (113/(48*np.pi)) * theta3 * chi_r
    # Avoid divide by zero
    chi_r = np.where(np.isclose(chi_r, 0.0), eps, chi_r)
    theta3 = np.where(np.isclose(theta3, 0.0), eps, theta3)
    theta3s = np.where(np.isclose(theta3s, 0.0), eps, theta3s)
    return theta3s


# In[8]:


# theta3s_m_s(m1, m2, s1z, s2z, flow, eps=1e-12)


# In[9]:


def convert_to_chi1chi2(theta0, theta3, theta3s, flow, condition='', eps=1e-12):

    if np.isclose(theta3, 0.0):
        theta3 = eps

    tau0 = theta0 / (2 * np.pi * flow)
    tau3 = theta3 / (2 * np.pi * flow)

    mass1 = mass1_from_tau0_tau3(tau0, tau3, flow)
    mass2 = mass2_from_tau0_tau3(tau0, tau3, flow)

    eta = eta_from_mass1_mass2(mass1, mass2)
    delta = (mass1 - mass2) / (mass1 + mass2)

    chi_r = (48 * np.pi / 113) * (theta3s / theta3)

    if condition == 'equal_spins':
        spin1z = chi_r / (1 - (76 / 113) * eta)
        spin2z = spin1z

    elif condition == 'zero_secondary':
        spin1z = chi_r / (0.5 + 0.5*delta - (76/226)*eta)
        spin2z = 0

    else:
        raise ValueError("condition must be 'equal_spins' or 'zero_secondary'")

    return np.array([spin1z, spin2z])


# $$
# \Gamma_{ij} = \left( \frac{\partial h}{\partial \theta_i} \,\middle|\, \frac{\partial h}{\partial \theta_j} \right)
# $$
# 
# with the inner product defined as  
# 
# $$
# (a|b) = 4 \, \Re \int_{f_{\text{low}}}^{f_{\text{high}}} \frac{\tilde{a}(f) \, \tilde{b}^*(f)}{S_n(f)} \, df \, ,
# $$
# 

# ##### Using finite difference formula to calculate the deravatives 
# $$
# f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} + \mathcal{O}(h^2)
# $$

# In[10]:

def metric_3d(params, Nf, psd, steps, keys=['dtheta0', 'dtheta3', 'dtheta3s', 'tc', 'phi0']):

    if steps is None:
        raise ValueError("`steps` must be provided to metric_3d")

    hp = get_wf(params, Nf)
    f = hp.sample_frequencies

    flow = params['f_lower']
    m1 = params['mass1']
    m2 = params['mass2']
    s1z = params['spin1z']
    s2z = params['spin2z']
    M = m1 + m2
    fisco = f_schwarzchild_isco(M)
    fhigh = fisco
    
    if np.isclose(s1z, s2z):
        condition = 'equal_spins'
    else:
        condition = 'zero_secondary'
        s2z = 0.
        
    delta_f = params['delta_f']

    A = np.abs(hp)
    norm_hp = overlap(hp, hp, psd, flow, fhigh, normalized=False)

    theta0, theta3 = theta03(m1, m2, flow)
    theta3s = theta3s_m_s(m1, m2, s1z, s2z, flow, eps=1e-12)

    derivs_phi = {}
    derivs_amp = {}

    for key in keys:
        dp = steps.get(key)

        if key == 'dtheta0':
            theta0_plus = theta0 + dp
            theta0_minus = theta0 - dp

            m1_plus, m2_plus = m1_m2f_theta03(theta0_plus, theta3, flow)
            m1_minus, m2_minus = m1_m2f_theta03(theta0_minus, theta3, flow)

            # <-- PASS the computed `condition` here
            s1z_plus, s2z_plus = convert_to_chi1chi2(theta0_plus, theta3, theta3s, flow, condition=condition)
            s1z_minus, s2z_minus = convert_to_chi1chi2(theta0_minus, theta3, theta3s, flow, condition=condition)

            valid_plus = (
                not np.iscomplex(m1_plus) and not np.iscomplex(m2_plus)
                and m1_plus > 0 and m2_plus > 0
                and np.isfinite(s1z_plus) and np.isfinite(s2z_plus)
                and abs(s1z_plus) < 1 and abs(s2z_plus) < 1
                and not np.iscomplex(s1z_plus) and not np.iscomplex(s2z_plus)
            )

            valid_minus = (
                not np.iscomplex(m1_minus) and not np.iscomplex(m2_minus)
                and m1_minus > 0 and m2_minus > 0
                and np.isfinite(s1z_minus) and np.isfinite(s2z_minus)
                and abs(s1z_minus) < 1 and abs(s2z_minus) < 1
                and not np.iscomplex(s1z_minus) and not np.iscomplex(s2z_minus)
            )

            if valid_plus and valid_minus:
                use_scheme = 'central'
                params_plus = params.copy()
                params_minus = params.copy()

                params_plus.update({
                    'mass1': float(m1_plus), 'mass2': float(m2_plus),
                    'spin1z': float(s1z_plus), 'spin2z': float(s2z_plus)
                })

                params_minus.update({
                    'mass1': float(m1_minus), 'mass2': float(m2_minus),
                    'spin1z': float(s1z_minus), 'spin2z': float(s2z_minus)
                })

            elif valid_plus:
                use_scheme = 'forward'
                params_plus = params.copy()
                params_plus.update({
                    'mass1': float(m1_plus), 'mass2': float(m2_plus),
                    'spin1z': float(s1z_plus), 'spin2z': float(s2z_plus)
                })

            elif valid_minus:
                use_scheme = 'backward'
                params_minus = params.copy()
                params_minus.update({
                    'mass1': float(m1_minus), 'mass2': float(m2_minus),
                    'spin1z': float(s1z_minus), 'spin2z': float(s2z_minus)
                })

        elif key == 'dtheta3':
            theta3_plus = theta3 + dp
            theta3_minus = theta3 - dp

            m1_plus, m2_plus = m1_m2f_theta03(theta0, theta3_plus, flow)
            m1_minus, m2_minus = m1_m2f_theta03(theta0, theta3_minus, flow)

            # <-- PASS the computed `condition` here
            s1z_plus, s2z_plus = convert_to_chi1chi2(theta0, theta3_plus, theta3s, flow, condition=condition)
            s1z_minus, s2z_minus = convert_to_chi1chi2(theta0, theta3_minus, theta3s, flow, condition=condition)

            valid_plus = (
                not np.iscomplex(m1_plus) and not np.iscomplex(m2_plus)
                and m1_plus > 0 and m2_plus > 0
                and np.isfinite(s1z_plus) and np.isfinite(s2z_plus)
                and abs(s1z_plus) < 1 and abs(s2z_plus) < 1
                and not np.iscomplex(s1z_plus) and not np.iscomplex(s2z_plus)
            )

            valid_minus = (
                not np.iscomplex(m1_minus) and not np.iscomplex(m2_minus)
                and m1_minus > 0 and m2_minus > 0
                and np.isfinite(s1z_minus) and np.isfinite(s2z_minus)
                and abs(s1z_minus) < 1 and abs(s2z_minus) < 1
                and not np.iscomplex(s1z_minus) and not np.iscomplex(s2z_minus)
            )

            if valid_plus and valid_minus:
                use_scheme = 'central'
                params_plus = params.copy()
                params_minus = params.copy()

                params_plus.update({
                    'mass1': float(m1_plus), 'mass2': float(m2_plus),
                    'spin1z': float(s1z_plus), 'spin2z': float(s2z_plus)
                })

                params_minus.update({
                    'mass1': float(m1_minus), 'mass2': float(m2_minus),
                    'spin1z': float(s1z_minus), 'spin2z': float(s2z_minus)
                })

            elif valid_plus:
                use_scheme = 'forward'
                params_plus = params.copy()
                params_plus.update({
                    'mass1': float(m1_plus), 'mass2': float(m2_plus),
                    'spin1z': float(s1z_plus), 'spin2z': float(s2z_plus)
                })

            elif valid_minus:
                use_scheme = 'backward'
                params_minus = params.copy()
                params_minus.update({
                    'mass1': float(m1_minus), 'mass2': float(m2_minus),
                    'spin1z': float(s1z_minus), 'spin2z': float(s2z_minus)
                })

        elif key == 'dtheta3s':
            theta3s_plus = theta3s + dp
            theta3s_minus = theta3s - dp

            # <-- PASS the computed `condition` here
            s1z_plus, s2z_plus = convert_to_chi1chi2(theta0, theta3, theta3s_plus, flow, condition=condition)
            s1z_minus, s2z_minus = convert_to_chi1chi2(theta0, theta3, theta3s_minus, flow, condition=condition)
            valid_plus = (
                np.isfinite(s1z_plus) and np.isfinite(s2z_plus)
                and abs(s1z_plus) < 1 and abs(s2z_plus) < 1
                and not np.iscomplex(s1z_plus) and not np.iscomplex(s2z_plus)
            )

            valid_minus = (
                np.isfinite(s1z_minus) and np.isfinite(s2z_minus)
#                 and abs(s1z_minus) < 1 and abs(s2z_minus) < 1
                and not np.iscomplex(s1z_minus) and not np.iscomplex(s2z_minus)
            )

            if valid_plus and valid_minus:
                use_scheme = 'central'
                params_plus = params.copy()
                params_minus = params.copy()

                params_plus.update({
                    'spin1z': float(s1z_plus), 'spin2z': float(s2z_plus)
                })

                params_minus.update({
                    'spin1z': float(s1z_minus), 'spin2z': float(s2z_minus)
                })

            elif valid_plus:
                use_scheme = 'forward'
                params_plus = params.copy()
                params_plus.update({
                    'spin1z': float(s1z_plus), 'spin2z': float(s2z_plus)
                })

            elif valid_minus:
                use_scheme = 'backward'
                params_minus = params.copy()
                params_minus.update({
                    'spin1z': float(s1z_minus), 'spin2z': float(s2z_minus)
                })

        elif key == 'tc':
            derivs_phi[key] = FrequencySeries(2 * np.pi * f, delta_f)
            derivs_amp[key] = FrequencySeries(np.zeros(Nf), delta_f)
            continue

        elif key == 'phi0':
            derivs_phi[key] = FrequencySeries(np.ones(Nf), delta_f)
            derivs_amp[key] = FrequencySeries(np.zeros(Nf), delta_f)
            continue

        if use_scheme == 'central':
            h_plus = get_wf(params_plus, Nf)
            h_minus = get_wf(params_minus, Nf)

            phi_plus = np.unwrap(np.angle(h_plus))
            phi_minus = np.unwrap(np.angle(h_minus))

            amp_plus = np.abs(h_plus)
            amp_minus = np.abs(h_minus)

            derivs_phi[key] = (phi_plus - phi_minus) / (2 * dp)
            derivs_amp[key] = (amp_plus - amp_minus) / (2 * dp)

        elif use_scheme == 'forward':
            h_plus = get_wf(params_plus, Nf)
            h0 = get_wf(params, Nf)

            phi_plus = np.unwrap(np.angle(h_plus))
            phi0 = np.unwrap(np.angle(h0))

            amp_plus = np.abs(h_plus)
            amp0 = np.abs(h0)

            derivs_phi[key] = (phi_plus - phi0) / dp
            derivs_amp[key] = (amp_plus - amp0) / dp

        elif use_scheme == 'backward':
            h_minus = get_wf(params_minus, Nf)
            h0 = get_wf(params, Nf)

            phi_minus = np.unwrap(np.angle(h_minus))
            phi0 = np.unwrap(np.angle(h0))

            amp_minus = np.abs(h_minus)
            amp0 = np.abs(h0)

            derivs_phi[key] = (phi0 - phi_minus) / dp
            derivs_amp[key] = (amp0 - amp_minus) / dp

    n = list(keys)
    k = len(n)

    F = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            key_i = n[i]
            key_j = n[j]

            t1 = overlap(A * derivs_phi[key_i],
                         A * derivs_phi[key_j],
                         psd, flow, fhigh, normalized=False)

            t2 = overlap(derivs_amp[key_i],
                         derivs_amp[key_j],
                         psd, flow, fhigh, normalized=False)

            F[i, j] = (t1 + t2) / (2 * norm_hp)

    G1 = F[:3, :3]
    G2 = F[:3, 3:]
    G3 = F[3:, 3:]
    G4 = F[3:, :3]

    G3_inv = np.linalg.inv(G3)
    g = G1 - G2 @ G3_inv @ G4

    return g