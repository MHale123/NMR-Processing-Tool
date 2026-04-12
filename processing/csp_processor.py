import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress


def fit_gaussian(ppm, spectrum, center_guess, window_ppm=0.3):
    """
    Fits a Gaussian to the peak nearest to center_guess within ±window_ppm.

    Returns the fitted peak centre in ppm, or center_guess on failure.
    """
    # Restrict to window around the guess
    if ppm[0] > ppm[-1]:
        # Inverted ppm axis
        mask = (ppm <= center_guess + window_ppm) & (ppm >= center_guess - window_ppm)
    else:
        mask = (ppm >= center_guess - window_ppm) & (ppm <= center_guess + window_ppm)

    ppm_w = ppm[mask]
    spec_w = spectrum[mask]

    if len(ppm_w) < 5:
        return center_guess, None

    def gaussian(x, amp, mu, sigma, baseline):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + baseline

    try:
        amp0      = float(spec_w[np.argmax(np.abs(spec_w))])
        sigma0    = abs(ppm_w[-1] - ppm_w[0]) / 6.0
        baseline0 = float(np.median(spec_w))

        popt, _ = curve_fit(
            gaussian, ppm_w, spec_w,
            p0=[amp0, center_guess, sigma0, baseline0],
            maxfev=5000,
        )
        fitted_centre = float(popt[1])

        # Sanity check: fitted centre must stay within the window
        lo = min(ppm_w[0], ppm_w[-1])
        hi = max(ppm_w[0], ppm_w[-1])
        if not (lo <= fitted_centre <= hi):
            return center_guess, None

        return fitted_centre, popt

    except Exception:
        # Fall back to simple maximum if Gaussian fit fails
        return float(ppm_w[np.argmax(np.abs(spec_w))]), None


def extract_peak_centres(spectra_list, center_ppm, window_ppm=0.3):
    """
    Fits a Gaussian to the selected peak in each spectrum and returns
    the fitted centre position for each.

    Parameters
    ----------
    spectra_list : list of dicts (output of load_1d_spectrum)
    center_ppm   : float, initial guess for peak centre in ppm
    window_ppm   : float, half-width of fitting window in ppm

    Returns
    -------
    centres : 1D array of fitted peak centres (ppm), one per spectrum
    """
    centres = []
    for s in spectra_list:
        centre, _ = fit_gaussian(s['ppm'], s['spectrum'], center_ppm, window_ppm)
        centres.append(centre)
    return np.array(centres)


def compute_delta_delta(centres, ref_idx):
    """
    Computes Δδ = δ_obs - δ_ref for each spectrum.

    Parameters
    ----------
    centres : 1D array of fitted peak centres
    ref_idx : int, index of the reference spectrum (no protein)

    Returns
    -------
    delta_delta : 1D array of Δδ values (ppm)
    """
    return centres - centres[ref_idx]


def fit_kd(concentrations, delta_delta, ref_idx):
    """
    Fits the dissociation constant Kd using the linear CSP method from
    Fielding (2007) / MacManus-Spencer (2010):

        [PFAS] = (n * [BSA] / Δδ) * ΔδB_app - Kd

    Rearranged: plotting [PFAS] vs 1/Δδ gives a line whose
    negative x-intercept = Kd.

    Only non-reference, non-zero Δδ points are used.

    Parameters
    ----------
    concentrations : 1D array of PFAS concentrations (µM)
    delta_delta    : 1D array of Δδ values (ppm)
    ref_idx        : int, index of the reference (excluded from fit)

    Returns
    -------
    dict with keys:
      'kd_uM'      : Kd in µM (from negative x-intercept)
      'slope'      : slope of [PFAS] vs 1/Δδ
      'intercept'  : y-intercept
      'r_squared'  : R² of linear fit
      'inv_dd'     : 1/Δδ values used in fit
      'conc_fit'   : concentration values used in fit
      'error'      : None or error string
    """
    # Exclude reference and any zero Δδ (avoid division by zero)
    mask = np.ones(len(concentrations), dtype=bool)
    mask[ref_idx] = False
    mask[np.abs(delta_delta) < 1e-6] = False

    conc_fit = concentrations[mask]
    dd_fit   = delta_delta[mask]

    if len(conc_fit) < 3:
        return {
            'kd_uM': None, 'slope': None, 'intercept': None,
            'r_squared': None, 'inv_dd': None, 'conc_fit': None,
            'error': "Need at least 3 non-reference points for Kd fit."
        }

    inv_dd = 1.0 / dd_fit

    try:
        slope, intercept, r, _, _ = linregress(inv_dd, conc_fit)
        r2 = r ** 2

        # Kd = negative x-intercept: 0 = slope * x + intercept → x = -intercept/slope
        # Kd is expressed as positive µM value
        kd = -intercept / slope if slope != 0 else None

        return {
            'kd_uM':     abs(kd) if kd is not None else None,
            'slope':     slope,
            'intercept': intercept,
            'r_squared': r2,
            'inv_dd':    inv_dd,
            'conc_fit':  conc_fit,
            'error':     None,
        }

    except Exception as e:
        return {
            'kd_uM': None, 'slope': None, 'intercept': None,
            'r_squared': None, 'inv_dd': None, 'conc_fit': None,
            'error': str(e),
        }