# processing/process.py
import numpy as np

def extract_trajectory(data, x0, x1):
    """Integrates peak between indices x0 and x1 across all spectra."""
    if len(data.shape) == 2:
        # Sums along the horizontal axis for every row (spectrum)
        return np.sum(data[:, x0:x1], axis=1)
    else:
        return np.array([np.sum(data[x0:x1])])