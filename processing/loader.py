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