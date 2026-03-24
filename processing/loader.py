import nmrglue as ng
import os
import numpy as np

class BrukerLoader:
    def __init__(self, exp_path):
        self.exp_path = exp_path

    def load_processed_data(self, proc_no=1):
        """
        Reads the processed 2D data (the stack of T1 experiments).
        Returns the dictionary and the 2D data matrix.
        """
        pdata_path = os.path.join(self.exp_path, 'pdata', str(proc_no))
        if not os.path.exists(pdata_path):
            raise FileNotFoundError(f"Could not find pdata at {pdata_path}")
            
        # ng.bruker.read_pdata handles the Bruker-specific processing files
        dic, data = ng.bruker.read_pdata(pdata_path)
        return dic, data

    def get_vd_list(self):
        """
        Attempts to find the variable delay list.
        In your case, we will start by pulling D16 as a baseline.
        """
        dic, _ = ng.bruker.read(self.exp_path)
        # We'll eventually need to handle the full list of delays
        # for all points in the recovery curve.
        return dic['acqus']['D'][16]