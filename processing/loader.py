import nmrglue as ng
import os
import numpy as np


def _find_2rr(subfolder):
    """
    Returns the path to the processed data folder containing '2rr', or None.
    Bruker software uses either 'pdata/1/' (TopSpin standard) or 'data/1/'
    (older / exported datasets). Check both.
    """
    for candidate in ('pdata', 'data'):
        path = os.path.join(subfolder, candidate, '1', '2rr')
        if os.path.exists(path):
            return os.path.join(subfolder, candidate, '1')
    return None


def find_t1_experiment(dataset_root):
    """
    Given a Bruker dataset root folder (the one you open in TopSpin that contains
    numbered subfolders like 1/, 2/, 3/...), this function scans all subfolders and
    returns the path of the one most likely to be a T1 inversion-recovery experiment.

    A T1 pseudo-2D experiment is identified by:
      - having an 'acqus' file (required for any Bruker experiment)
      - having a 'vdlist' file (variable delay list — the inversion times)
      - having either a 'ser' file (raw pseudo-2D) or a processed 2rr matrix
        (in either pdata/1/ or data/1/)

    Returns the path to the best candidate subfolder, or None if not found.
    """
    candidates = []

    for entry in sorted(os.listdir(dataset_root)):
        subfolder = os.path.join(dataset_root, entry)
        if not os.path.isdir(subfolder):
            continue

        has_acqus  = os.path.exists(os.path.join(subfolder, 'acqus'))
        has_vdlist = os.path.exists(os.path.join(subfolder, 'vdlist'))
        has_ser    = os.path.exists(os.path.join(subfolder, 'ser'))
        pdata_path = _find_2rr(subfolder)          # None if not found
        has_2rr    = pdata_path is not None

        if has_acqus and has_vdlist and (has_ser or has_2rr):
            # Score: prefer processed (2rr) over raw (ser), then lower experiment number
            score = (0 if has_2rr else 1, int(entry) if entry.isdigit() else 999)
            candidates.append((score, subfolder))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


class BrukerLoader:
    """
    Loads a Bruker T1 inversion-recovery experiment.

    Usage:
        # Option A: point directly at the numbered experiment folder (e.g. .../Test/.../3)
        loader = BrukerLoader("/path/to/dataset/3")

        # Option B: point at the dataset root and let it find the T1 experiment
        loader = BrukerLoader("/path/to/dataset", auto_find=True)
    """

    def __init__(self, path, auto_find=False):
        if auto_find:
            found = find_t1_experiment(path)
            if found is None:
                raise FileNotFoundError(
                    f"Could not find a T1 inversion-recovery experiment under:\n{path}\n\n"
                    "Expected to find a subfolder containing 'acqus' + 'vdlist' + ('ser' or pdata/1/2rr)."
                )
            self.exp_path = found
        else:
            self.exp_path = path

        self._dic   = None   # acqus/acqu2s parameter dictionary
        self._data  = None   # numpy array of spectral data
        self._vdlist = None  # inversion delay times
        self._loaded = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_acqus(self):
        """Read only the parameter dictionary (cheap — no data loaded)."""
        if self._dic is None:
            self._dic, _ = ng.bruker.read(self.exp_path)
        return self._dic

    def _load_vdlist(self):
        """Read the variable delay list (inversion times in seconds)."""
        vdlist_path = os.path.join(self.exp_path, 'vdlist')
        delays = []
        with open(vdlist_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Values can have a unit suffix: 's', 'm', 'u' — convert to seconds
                if line.endswith('s') and not line.endswith('ms') and not line.endswith('us'):
                    delays.append(float(line[:-1]))
                elif line.endswith('ms'):
                    delays.append(float(line[:-2]) * 1e-3)
                elif line.endswith('us'):
                    delays.append(float(line[:-2]) * 1e-6)
                elif line.endswith('m') and not line.endswith('nm'):
                    delays.append(float(line[:-1]) * 1e-3)
                else:
                    delays.append(float(line))   # bare number → assume seconds
        return np.array(delays)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_experiment_path(self):
        """Returns the resolved experiment folder path."""
        return self.exp_path

    def get_metadata(self):
        """
        Returns a dict of key experimental parameters.
        Safe to call even if the data has not been loaded yet.
        """
        dic = self._read_acqus()
        acqus = dic.get('acqus', dic)  # some nmrglue versions flatten the dict

        d_array = acqus.get('D', [0] * 20)

        meta = {
            "exp_path":   self.exp_path,
            "pulprog":    acqus.get('PULPROG', 'unknown'),
            "nuc1":       acqus.get('NUC1', 'unknown'),
            "sfo1_mhz":   acqus.get('SFO1', 0.0),
            "sw_hz":      acqus.get('SW_h', 0.0),
            "ns":         acqus.get('NS', 0),
            "td":         acqus.get('TD', 0),
            "d1_s":       d_array[1]  if len(d_array) > 1  else None,
            "d16_s":      d_array[16] if len(d_array) > 16 else None,
            "has_vdlist": os.path.exists(os.path.join(self.exp_path, 'vdlist')),
            "has_ser":    os.path.exists(os.path.join(self.exp_path, 'ser')),
            "has_2rr":    _find_2rr(self.exp_path) is not None,
        }

        if meta["has_vdlist"]:
            meta["vdlist"] = self._load_vdlist()
            meta["n_delays"] = len(meta["vdlist"])

        return meta

    def load_processed_data(self, proc_no=1):
        """
        Loads the processed 2D data matrix from the pdata (or data) folder.

        Checks both 'pdata/<proc_no>/' and 'data/<proc_no>/' since older or
        exported Bruker datasets sometimes use the latter path.

        Returns (dic, data) where:
          - dic  : parameter dictionary from nmrglue
          - data : 2D numpy array, shape (n_delays, n_spectral_points)
                   rows = each inversion delay, columns = chemical shift axis

        Raises FileNotFoundError if neither processed data folder exists.
        """
        # Try both path variants
        pdata_path = None
        for candidate in ('pdata', 'data'):
            p = os.path.join(self.exp_path, candidate, str(proc_no))
            if os.path.exists(p):
                pdata_path = p
                break

        if pdata_path is None:
            raise FileNotFoundError(
                f"Processed data folder not found under:\n  {self.exp_path}\n\n"
                "Looked for both 'pdata/1/' and 'data/1/'.\n"
                "The data may not have been processed yet in TopSpin. "
                "Try running 'xf2' in TopSpin first, or load the raw FID instead."
            )

        dic, data = ng.bruker.read_pdata(pdata_path)
        self._dic  = dic
        self._data = data
        return dic, data

    def load_raw_data(self):
        """
        Loads raw (unprocessed) FID data from the 'ser' file.

        Returns (dic, data) where data is the raw time-domain array.
        Useful as a fallback when processed data is not available.
        """
        ser_path = os.path.join(self.exp_path, 'ser')
        if not os.path.exists(ser_path):
            raise FileNotFoundError(
                f"Raw FID file 'ser' not found in:\n  {self.exp_path}"
            )
        dic, data = ng.bruker.read(self.exp_path)
        self._dic  = dic
        self._data = data
        return dic, data

    def get_delays(self):
        """
        Returns the array of inversion delay times (in seconds) from vdlist.
        Raises FileNotFoundError if vdlist is missing.
        """
        vdlist_path = os.path.join(self.exp_path, 'vdlist')
        if not os.path.exists(vdlist_path):
            raise FileNotFoundError(
                f"'vdlist' not found in:\n  {self.exp_path}\n\n"
                "This file contains the inversion-recovery delay times. "
                "It should be present for any T1 experiment."
            )
        return self._load_vdlist()

    def get_ppm_axis(self, proc_no=1):
        """
        Returns the chemical shift (ppm) axis for the direct (F2) dimension.

        Builds the axis manually from the 'procs' file parameters rather than
        relying on ng.bruker.make_uc, which requires the full acquisition dic
        and can silently return None for pseudo-2D datasets.

        Parameters used (all from pdata/<proc_no>/procs):
          SW_p  — spectral width in Hz
          SF    — spectrometer frequency in MHz (used as reference for ppm)
          OFFSET — ppm value of the left edge of the spectrum
          SI    — number of real points in the processed spectrum

        Returns a 1D numpy array of length SI, or None on failure.
        """
        pdata_path = None
        for candidate in ('pdata', 'data'):
            p = os.path.join(self.exp_path, candidate, str(proc_no))
            if os.path.exists(p):
                pdata_path = p
                break

        if pdata_path is None:
            return None

        procs_path = os.path.join(pdata_path, 'procs')
        if not os.path.exists(procs_path):
            return None

        try:
            # Parse procs as key-value pairs
            params = {}
            with open(procs_path, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('##$'):
                        if '=' in line:
                            key, _, val = line[3:].partition('=')
                            params[key.strip()] = val.strip()

            sw_p   = float(params['SW_p'])   # spectral width in Hz
            sf     = float(params['SF'])     # spectrometer freq in MHz
            offset = float(params['OFFSET']) # ppm of left edge
            si     = int(params['SI'])       # number of real points

            # ppm axis: left edge = OFFSET, right edge = OFFSET - SW_p/SF
            # (ppm decreases left → right in NMR convention)
            sw_ppm = sw_p / sf
            ppm    = np.linspace(offset, offset - sw_ppm, si)
            return ppm

        except Exception:
            # Fall back to nmrglue make_uc if manual parsing fails
            try:
                dic, data = ng.bruker.read_pdata(pdata_path)
                uc = ng.bruker.make_uc(dic, data, dim=1)
                return uc.ppm_scale()
            except Exception:
                return None


def load_1d_spectrum(exp_path, proc_no=1):
    """
    Loads a processed Bruker 1D spectrum from pdata/<proc_no>/1r.

    This is used by the CSP module for loading individual concentration
    points. Each folder is a separate 1D acquisition.

    Parameters
    ----------
    exp_path : str, path to the numbered Bruker experiment folder
    proc_no  : int, processing number (default 1)

    Returns
    -------
    dict with keys:
      'ppm'      : 1D numpy array of chemical shift values
      'spectrum' : 1D numpy array of real intensities
      'nucleus'  : str, e.g. '19F' or '1H'
      'sfo1_mhz' : float, spectrometer frequency
      'sw_hz'    : float, spectral width in Hz
      'pulprog'  : str, pulse program name
      'path'     : str, the experiment path
    or raises FileNotFoundError / ValueError on failure.
    """
    # Find processed data folder
    pdata_path = None
    for candidate in ('pdata', 'data'):
        p = os.path.join(exp_path, candidate, str(proc_no))
        if os.path.exists(p):
            pdata_path = p
            break

    if pdata_path is None:
        raise FileNotFoundError(
            "No processed data folder found in:\n  {}\n"
            "Looked for pdata/{}/ and data/{}/".format(exp_path, proc_no, proc_no)
        )

    one_r = os.path.join(pdata_path, '1r')
    if not os.path.exists(one_r):
        raise FileNotFoundError(
            "No '1r' file found in:\n  {}\n"
            "This folder does not contain a processed 1D spectrum.".format(pdata_path)
        )

    # ------------------------------------------------------------------
    # Read processed 1D data — real (1r) and imaginary (1i) parts.
    # Having both parts allows us to apply zero- and first-order phase
    # correction in software, which fixes the curved baselines that arise
    # when the on-spectrometer phase correction was imperfect.
    # ------------------------------------------------------------------
    dic, data_r = ng.bruker.read_pdata(pdata_path)
    spectrum_r  = np.array(data_r, dtype=float).ravel()

    # Try to load imaginary part (may not exist for all datasets)
    one_i = os.path.join(pdata_path, '1i')
    spectrum_i = None
    if os.path.exists(one_i):
        try:
            _, data_i  = ng.bruker.read_pdata(
                pdata_path, bin_files=['1i'])
            spectrum_i = np.array(data_i, dtype=float).ravel()
            if len(spectrum_i) != len(spectrum_r):
                spectrum_i = None
        except Exception:
            spectrum_i = None

    # Read PHC0 / PHC1 from procs for phase correction
    phc0, phc1 = 0.0, 0.0
    procs_path_phase = os.path.join(pdata_path, 'procs')
    if os.path.exists(procs_path_phase):
        try:
            params_ph = {}
            with open(procs_path_phase, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('##$') and '=' in line:
                        key, _, val = line[3:].partition('=')
                        params_ph[key.strip()] = val.strip()
            phc0 = float(params_ph.get('PHC0', 0.0))
            phc1 = float(params_ph.get('PHC1', 0.0))
        except Exception:
            pass

    # Apply phase correction if we have imaginary data.
    # Phase model: spec_phased = real( (R + iI) * exp(i*phi(x)) )
    # where phi(x) = (phc0 + phc1*(x - x_pivot)) in radians.
    # x_pivot = right edge of spectrum (standard Bruker convention).
    if spectrum_i is not None and (phc0 != 0.0 or phc1 != 0.0):
        n      = len(spectrum_r)
        # Normalised frequency axis 0→1 left to right (Bruker convention)
        x_norm = np.linspace(0.0, 1.0, n)
        phi    = np.deg2rad(phc0 + phc1 * x_norm)
        cplx   = (spectrum_r + 1j * spectrum_i) * np.exp(1j * phi)
        spectrum = cplx.real
    else:
        spectrum = spectrum_r

    # Build ppm axis from procs (same logic as BrukerLoader.get_ppm_axis)
    procs_path = os.path.join(pdata_path, 'procs')
    ppm = None
    if os.path.exists(procs_path):
        try:
            params = {}
            with open(procs_path, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('##$') and '=' in line:
                        key, _, val = line[3:].partition('=')
                        params[key.strip()] = val.strip()
            sw_p   = float(params['SW_p'])
            sf     = float(params['SF'])
            offset = float(params['OFFSET'])
            si     = int(params['SI'])
            sw_ppm = sw_p / sf
            ppm    = np.linspace(offset, offset - sw_ppm, si)
        except Exception:
            ppm = None

    if ppm is None or len(ppm) != len(spectrum):
        ppm = np.arange(len(spectrum), dtype=float)

    # Read acqus for metadata
    acqus_path = os.path.join(exp_path, 'acqus')
    nucleus  = 'unknown'
    sfo1     = 0.0
    sw_hz    = 0.0
    pulprog  = 'unknown'

    if os.path.exists(acqus_path):
        try:
            params = {}
            with open(acqus_path, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('##$') and '=' in line:
                        key, _, val = line[3:].partition('=')
                        params[key.strip()] = val.strip()
            nucleus = params.get('NUC1', 'unknown').strip('<>').strip()
            sfo1    = float(params.get('SFO1', 0.0))
            sw_hz   = float(params.get('SW_h', 0.0))
            pulprog = params.get('PULPROG', 'unknown').strip('<>').strip()
        except Exception:
            pass

    return {
        'ppm':      ppm,
        'spectrum': spectrum,
        'spectrum_r': spectrum_r,
        'spectrum_i': spectrum_i,   # None if 1i not available
        'phc0':     phc0,
        'phc1':     phc1,
        'nucleus':  nucleus,
        'sfo1_mhz': sfo1,
        'sw_hz':    sw_hz,
        'pulprog':  pulprog,
        'path':     exp_path,
    }