import numpy as np
import pycbc
import matplotlib.pyplot as plt
from pycbc.psd import aLIGOAPlusDesignSensitivityT1800042
from pycbc.waveform import get_fd_waveform, get_waveform_filter_length_in_time as chirplen
from pycbc.filter.matchedfilter import overlap, match
from tqdm import tqdm
from pycbc.conversions import mass1_from_tau0_tau3, mass2_from_tau0_tau3, tau0_from_mass1_mass2, tau3_from_mass1_mass2
from pycbc.conversions import eta_from_mass1_mass2
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
from pycbc.types.frequencyseries import FrequencySeries
from scipy.interpolate import griddata
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import pandas as pd
import lal
from scipy.stats import qmc
from sklearn.neighbors import BallTree

## waveform genration function
def get_wf(params, Nf):
    try:
        # sanity check
        if params['mass1'] <= 0 or params['mass2'] <= 0:
            return None

        wf, _ = get_fd_waveform(template=None, **params)

        if len(wf) != Nf:  # resize if needed
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


## m1, m2 from theta0, theta3
def m1_m2f_theta03(theta0, theta3, flow):
    
    tau0 = theta0 / (2 * np.pi * flow)
    tau3 = theta3 / (2 * np.pi * flow)
    
    m1 = mass1_from_tau0_tau3(tau0, tau3, flow)
    m2 = mass2_from_tau0_tau3(tau0, tau3, flow)
    
    return m1, m2

## theta0, theta3 from m1, m2
def theta03(m1, m2, flow):
    tau0 = tau0_from_mass1_mass2(m1, m2, flow)
    tau3 = tau3_from_mass1_mass2(m1, m2, flow)
    theta0 = 2 * np.pi * flow * tau0
    theta3 = 2* np.pi * flow * tau3
    return theta0, theta3

## metric on signal manifold of 2d in theta0, theta3 cordinates from fisher information matrix 
def metric_2d(params, Nf, psd, steps, keys=['dtheta0', 'dtheta3', 'tc', 'phi0']):

    m1 = params['mass1']
    m2 = params['mass2']
    M = m1 + m2
    fisco = f_schwarzchild_isco(M)
    fhigh = fisco
    hp = get_wf(params, Nf)
    
    f = hp.sample_frequencies  

    flow = params['f_lower']

    A = np.abs(hp)  # Amplitude
    norm_hp = overlap(hp, hp, psd, flow, fhigh, normalized=False)

    delta_f = params['delta_f']

    theta0, theta3 = theta03(m1, m2, flow)

    derivs_phi = {}
    derivs_amp = {}

    for key in keys:
        dp = steps.get(key)

        if key == 'dtheta0':
            # candidate points
            theta_plus = theta0 + dp
            theta_minus = theta0 - dp

            m1_plus, m2_plus = m1_m2f_theta03(theta_plus, theta3, flow)
            m1_minus, m2_minus = m1_m2f_theta03(theta_minus, theta3, flow)

            valid_plus = (not np.iscomplex(m1_plus) and not np.iscomplex(m2_plus)
                          and m1_plus > 0 and m2_plus > 0)
            valid_minus = (not np.iscomplex(m1_minus) and not np.iscomplex(m2_minus)
                           and m1_minus > 0 and m2_minus > 0)

            if valid_plus and valid_minus:
                # central difference (preferred)
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus.update({'mass1': float(m1_plus), 'mass2': float(m2_plus)})
                params_minus.update({'mass1': float(m1_minus), 'mass2': float(m2_minus)})
                use_scheme = 'central'
            elif valid_plus:
                # forward difference: (f(x+dp) - f(x)) / dp
                params_plus = params.copy()
                params_plus.update({'mass1': float(m1_plus), 'mass2': float(m2_plus)})
                params_minus = params.copy()  # central point
                use_scheme = 'forward'
            elif valid_minus:
                # backward difference: (f(x) - f(x-dp)) / dp
                params_minus = params.copy()
                params_minus.update({'mass1': float(m1_minus), 'mass2': float(m2_minus)})
                params_plus = params.copy()  # central point
                use_scheme = 'backward'
            else:
                return None  # can't compute derivative at boundary

        elif key == 'dtheta3':
            theta_plus = theta3 + dp
            theta_minus = theta3 - dp

            m1_plus, m2_plus = m1_m2f_theta03(theta0, theta_plus, flow)
            m1_minus, m2_minus = m1_m2f_theta03(theta0, theta_minus, flow)

            valid_plus = (not np.iscomplex(m1_plus) and not np.iscomplex(m2_plus)
                          and m1_plus > 0 and m2_plus > 0)
            valid_minus = (not np.iscomplex(m1_minus) and not np.iscomplex(m2_minus)
                           and m1_minus > 0 and m2_minus > 0)

            if valid_plus and valid_minus:
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus.update({'mass1': float(m1_plus), 'mass2': float(m2_plus)})
                params_minus.update({'mass1': float(m1_minus), 'mass2': float(m2_minus)})
                use_scheme = 'central'
            elif valid_plus:
                params_plus = params.copy()
                params_plus.update({'mass1': float(m1_plus), 'mass2': float(m2_plus)})
                params_minus = params.copy()
                use_scheme = 'forward'
            elif valid_minus:
                params_minus = params.copy()
                params_minus.update({'mass1': float(m1_minus), 'mass2': float(m2_minus)})
                params_plus = params.copy()
                use_scheme = 'backward'
            else:
                return None

        elif key == 'tc':
            derivs_phi[key] = FrequencySeries(2 * np.pi * f, delta_f)
            derivs_amp[key] = FrequencySeries(np.zeros(Nf), delta_f)
            continue

        elif key == 'phi0':
            derivs_phi[key] = FrequencySeries(np.ones(Nf), delta_f)
            derivs_amp[key] = FrequencySeries(np.zeros(Nf), delta_f)
            continue

        # Now compute waveforms needed depending on scheme
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
            # f(x+dp) - f(x)
            h_plus = get_wf(params_plus, Nf)
            h0 = get_wf(params, Nf)  # central

            phi_plus = np.unwrap(np.angle(h_plus))
            phi0 = np.unwrap(np.angle(h0))

            amp_plus = np.abs(h_plus)
            amp0 = np.abs(h0)

            derivs_phi[key] = (phi_plus - phi0) / dp
            derivs_amp[key] = (amp_plus - amp0) / dp

        elif use_scheme == 'backward':
            # f(x) - f(x-dp)
            h_minus = get_wf(params_minus, Nf)
            h0 = get_wf(params, Nf)  # central

            phi_minus = np.unwrap(np.angle(h_minus))
            phi0 = np.unwrap(np.angle(h0))

            amp_minus = np.abs(h_minus)
            amp0 = np.abs(h0)

            derivs_phi[key] = (phi0 - phi_minus) / dp
            derivs_amp[key] = (amp0 - amp_minus) / dp

        else:
            # Should not happen, but guard
            return None

    # if any derivative is missing → bail
    if len(derivs_phi) < len(keys) or len(derivs_amp) < len(keys):
        return None

    # build Fisher matrix
    n = list(keys)
    k = len(n)

    F = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            key_i = n[i]
            key_j = n[j]
            t1 = overlap(A * derivs_phi[key_i], A * derivs_phi[key_j], psd, flow, fhigh, normalized=False)
            t2 = overlap(derivs_amp[key_i], derivs_amp[key_j], psd, flow, fhigh, normalized=False)
            F[i, j] = (t1 + t2) / (2 * norm_hp)

    G1 = F[0:2, 0:2]  # intrinsic- intrinsic block
    G2 = F[0:2, 2:4]  # intrinsic - extrinsic block
    G3 = F[2:4, 2:4]  # extrinsic - extrinsic block
    G4 = F[2:4, 0:2]  # extrinsic - intrinsic block
    G3_inv = np.linalg.inv(G3)
    g = G1 - G2 @ G3_inv @ G4

    return g