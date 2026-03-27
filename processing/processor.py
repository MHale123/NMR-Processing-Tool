import numpy as np


def extract_trajectory(data, x0, x1):
    """
    Integrates (sums) the spectral intensity between column indices x0 and x1
    across every row (every delay time).

    Parameters
    ----------
    data : 2D numpy array, shape (n_delays, n_spectral_points)
    x0   : int, left column index (inclusive)
    x1   : int, right column index (exclusive)

    Returns
    -------
    traj : 1D numpy array of length n_delays
    """
    if data.ndim == 2:
        return np.sum(data[:, x0:x1], axis=1).astype(float)
    else:
        return np.array([np.sum(data[x0:x1])], dtype=float)


def detect_polarity(trajectory):
    """
    Determines whether the inversion-recovery trajectory is in the expected
    orientation or has been phase-flipped by Bruker processing.

    Standard IR:  signal starts NEGATIVE (inverted), rises to POSITIVE.
    Flipped IR:   signal starts POSITIVE, falls to NEGATIVE.

    Returns 1 if standard, -1 if flipped.
    """
    # Compare the sign of the first point vs the last point.
    # In a standard IR, first < 0 and last > 0 (or at least last > first).
    first = float(trajectory[0])
    last  = float(trajectory[-1])

    if last > first:
        return 1    # standard: recovering upward
    else:
        return -1   # flipped: recovering downward


def check_data_sufficiency(delays, trajectory, t1_estimate):
    """
    Checks whether the delay range is long enough to reliably determine T1.

    A reliable T1 measurement requires:
      - The curve to have crossed zero (t_max > T1 * ln(2))
      - Ideally t_max > 3 * T1 so the plateau is approached

    Returns a list of warning strings (empty list = no warnings).
    """
    warnings = []
    t_max = float(delays[-1])
    t_zero_crossing = t1_estimate * np.log(2)

    if t_max < t_zero_crossing:
        warnings.append(
            "WARNING: Your longest delay ({:.1f} s) is shorter than the\n"
            "estimated zero-crossing time ({:.1f} s = T1 x ln2).\n"
            "The signal has not yet crossed zero, so T1 is poorly constrained.\n"
            "Acquire longer delays (suggest up to {:.0f} s) for a reliable fit.".format(
                t_max, t_zero_crossing, t1_estimate * 5
            )
        )
    elif t_max < 3 * t1_estimate:
        warnings.append(
            "NOTE: Your longest delay ({:.1f} s) is less than 3xT1 ({:.1f} s).\n"
            "The curve has not fully plateaued. The T1 estimate may be\n"
            "slightly underestimated. Consider acquiring up to {:.0f} s.".format(
                t_max, 3 * t1_estimate, t1_estimate * 5
            )
        )

    if len(delays) < 6:
        warnings.append(
            "NOTE: Only {} delay points measured. At least 8-10 points\n"
            "spread across the recovery curve give more reliable T1 fits.".format(len(delays))
        )

    return warnings


def fit_t1(delays, trajectory):
    """
    Fits the T1 inversion-recovery curve.

    Handles both standard and phase-inverted data automatically.

    Standard Bruker IR model:
        I(t) = A * (1 - 2 * exp(-t / T1)) + C

    Parameters
    ----------
    delays     : 1D array of delay times in seconds
    trajectory : 1D array of integrated peak intensities

    Returns
    -------
    dict with keys:
      't1'         : fitted T1 in seconds
      'amplitude'  : A (equilibrium intensity, always returned as positive)
      'offset'     : C (baseline offset)
      'polarity'   : 1 (standard) or -1 (phase-flipped data)
      'fitted'     : array of fitted values at the input delay times
      'residuals'  : trajectory - fitted
      'r_squared'  : goodness of fit (0-1)
      'warnings'   : list of warning strings about data quality
      'error'      : None on success, or an error message string on failure
    """
    from scipy.optimize import curve_fit

    def ir_model(t, A, T1, C):
        return A * (1.0 - 2.0 * np.exp(-t / T1)) + C

    # ------------------------------------------------------------------
    # Step 1: detect and correct polarity
    # ------------------------------------------------------------------
    polarity = detect_polarity(trajectory)
    traj_fit = trajectory * polarity   # now always in standard orientation

    # ------------------------------------------------------------------
    # Step 2: robust initial guesses on the corrected trajectory
    # ------------------------------------------------------------------
    # A_guess: the equilibrium value — use the last point if it's positive,
    # otherwise use the maximum absolute value.
    if traj_fit[-1] > 0:
        A_guess = float(traj_fit[-1])
    else:
        A_guess = float(np.max(np.abs(traj_fit)))

    # T1_guess: find zero crossing in the corrected trajectory
    sign_changes = np.where(np.diff(np.sign(traj_fit)))[0]
    if len(sign_changes) > 0:
        i       = sign_changes[0]
        y0, y1  = traj_fit[i], traj_fit[i + 1]
        t0, t1p = delays[i], delays[i + 1]
        t_cross = t0 - y0 * (t1p - t0) / (y1 - y0)
        T1_guess = max(t_cross / np.log(2), 1e-3)
    else:
        # No zero crossing in range — T1 is probably >> longest delay
        T1_guess = float(delays[-1]) * 2.0

    C_guess = 0.0

    # ------------------------------------------------------------------
    # Step 3: try fitting with multiple T1 seeds
    # ------------------------------------------------------------------
    t1_seeds = [
        T1_guess,
        T1_guess * 0.3,
        T1_guess * 0.5,
        T1_guess * 2.0,
        T1_guess * 4.0,
        float(delays[-1]),
    ]
    best_result = None
    best_r2     = -np.inf

    for t1_seed in t1_seeds:
        if t1_seed <= 0:
            continue
        try:
            popt, _ = curve_fit(
                ir_model,
                delays,
                traj_fit,
                p0=[A_guess, t1_seed, C_guess],
                maxfev=20_000,
                bounds=([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
            )
            A_fit, T1_fit, C_fit = popt
            fitted_corrected = ir_model(delays, *popt)

            ss_res = np.sum((traj_fit - fitted_corrected) ** 2)
            ss_tot = np.sum((traj_fit - np.mean(traj_fit)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            if r2 > best_r2:
                best_r2 = r2
                # Return fitted values in ORIGINAL polarity so plots match data
                fitted_original = fitted_corrected * polarity
                best_result = {
                    "t1":        T1_fit,
                    "amplitude": A_fit,
                    "offset":    C_fit * polarity,
                    "polarity":  polarity,
                    "fitted":    fitted_original,
                    "residuals": trajectory - fitted_original,
                    "r_squared": r2,
                    "warnings":  [],
                    "error":     None,
                }
        except Exception:
            continue

    if best_result is not None:
        best_result["warnings"] = check_data_sufficiency(
            delays, trajectory, best_result["t1"]
        )
        return best_result

    return {
        "t1":        None,
        "amplitude": None,
        "offset":    None,
        "polarity":  polarity,
        "fitted":    None,
        "residuals": None,
        "r_squared": None,
        "warnings":  [],
        "error": (
            "Curve fitting failed with all initial guesses.\n\n"
            "Common causes:\n"
            "  - Integration window is not on a real peak\n"
            "  - All delay points have the same sign (curve never crosses zero\n"
            "    AND T1 >> longest delay, making the shape indistinguishable\n"
            "    from a flat line to the optimizer)\n"
            "  - Too few delay points (fewer than 4 non-zero values)\n\n"
            "Try adjusting the integration window or acquiring more delays."
        ),
    }