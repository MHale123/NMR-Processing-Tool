import nmrglue as ng
import os
import numpy as np

class BrukerLoader:
    def __init__(self, exp_path):
        """
        exp_path: Path to the specific experiment folder (e.g., '.../Test/Experiment/1')
        """
        self.exp_path = exp_path

    def get_metadata(self):
        """Extracts T1-specific parameters: Pulse program and delays."""
        dic, _ = ng.bruker.read(self.exp_path)
        acqus = dic['acqus']
        
        # Pulling the D array (delays)
        # In your IR experiment, D1 is recycle, D16 is the variable delay
        metadata = {
            "pulprog": acqus.get('PULPROG', 'unknown'),
            "d1": acqus['D'][1],
            "d16": acqus['D'][16],
            "nuc": acqus.get('NUC1', 'unknown'),
            "sfo1": acqus.get('SFO1', 0.0) # Frequency for PPM conversion
        }
        return metadata

    def load_processed_data(self, proc_no=1):
        """Loads the processed 2D matrix (pdata/1/2rr or 1r)."""
        pdata_path = os.path.join(self.exp_path, 'pdata', str(proc_no))
        if not os.path.exists(pdata_path):
            return None, None
        
        # read_pdata returns (dictionary, data_array)
        return ng.bruker.read_pdata(pdata_path)