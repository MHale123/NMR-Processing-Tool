import csv
import os
from datetime import datetime


def export_results(filepath, result, delays, trajectory, integration_window, experiment_path):
    """
    Exports T1 fit results to a plain CSV file.

    Parameters
    ----------
    filepath           : str, full path to write (should end in .csv)
    result             : dict returned by fit_t1()
    delays             : 1D array of delay times used in the fit
    trajectory         : 1D array of integrated intensities used in the fit
    integration_window : dict with keys 'left', 'right', 'unit', 'width'
    experiment_path    : str, path to the Bruker experiment folder
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        # --- Header block ---
        writer.writerow(["NMR T1 Processor — Export"])
        writer.writerow(["Exported", timestamp])
        writer.writerow(["Experiment", experiment_path])
        writer.writerow([])

        # --- Fit results ---
        writer.writerow(["FIT RESULTS"])
        if result.get("error") is None:
            t1     = result["t1"]
            t_zero = t1 * 0.6931471805599453   # T1 * ln(2)
            writer.writerow(["T1 (s)",          "{:.6f}".format(t1)])
            writer.writerow(["R2",         "{:.6f}".format(result["r_squared"])])
            writer.writerow(["t0 (s)",     "{:.6f}".format(t_zero)])
            writer.writerow(["A (a.u.)",        "{:.2f}".format(result["amplitude"])])
            writer.writerow(["C (a.u.)",        "{:.2f}".format(result["offset"])])
            writer.writerow(["Polarity",        "standard" if result["polarity"] == 1 else "phase-flipped"])
            writer.writerow(["Fit model",       "A * (1 - 2 * exp(-t / T1)) + C"])
        else:
            writer.writerow(["Fit status", "FAILED"])
            writer.writerow(["Error", result["error"]])
        writer.writerow([])

        # --- Integration window ---
        writer.writerow(["INTEGRATION WINDOW"])
        writer.writerow(["Left edge",  "{:.4f} {}".format(integration_window["left"],  integration_window["unit"])])
        writer.writerow(["Right edge", "{:.4f} {}".format(integration_window["right"], integration_window["unit"])])
        writer.writerow(["Width",      "{:.4f} {}".format(integration_window["width"], integration_window["unit"])])
        writer.writerow([])

        # --- Warnings ---
        if result.get("warnings"):
            writer.writerow(["WARNINGS"])
            for w in result["warnings"]:
                writer.writerow([w])
            writer.writerow([])

        # --- Raw data table ---
        writer.writerow(["RAW DATA"])
        header = ["Delay (s)", "Intensity (a.u.)"]
        if result.get("fitted") is not None:
            header += ["Fitted (a.u.)", "Residual (a.u.)"]
        writer.writerow(header)

        fitted    = result.get("fitted")
        residuals = result.get("residuals")

        for i, (d, y) in enumerate(zip(delays, trajectory)):
            row = ["{:.6f}".format(d), "{:.2f}".format(y)]
            if fitted is not None and i < len(fitted):
                row += ["{:.2f}".format(fitted[i]), "{:.2f}".format(residuals[i])]
            writer.writerow(row)