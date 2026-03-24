import numpy as np
from scipy.optimize import curve_fit

def t1_inversion_recovery(t, M0, A, T1):
    """
    Standard T1 Inversion Recovery equation:
    M(t) = M0 * (1 - A * exp(-t / T1))
    """
    return M0 * (1 - A * np.exp(-t / T1))

def fit_t1_data(times, intensities):
    """
    Fits the trajectory to the T1 equation.
    Returns: (M0, A, T1)
    """
    # Initial guesses: 
    # M0 = max intensity, A = 2 (for perfect inversion), T1 = 1.0s
    p0 = [np.max(np.abs(intensities)), 2.0, 1.0]
    
    try:
        popt, pcov = curve_fit(t1_inversion_recovery, times, intensities, p0=p0)
        return popt
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None