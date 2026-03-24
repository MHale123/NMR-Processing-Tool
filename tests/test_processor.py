import os
import sys
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt

# This ensures the script can see the 'processing' folder in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.loader import BrukerLoader
from processing.processor import extract_trajectory

def test_integration():
    # 1. Setup the path to your '1' folder on the Desktop
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Test")
    
    # We'll use our discovery logic to find the '1' folder automatically
    target_dir = None
    for root, dirs, files in os.walk(desktop_path):
        if 'acqu' in files:
            target_dir = root
            break
            
    if not target_dir:
        print("❌ Could not find a folder with an 'acqu' file in Desktop/Test")
        return

    print(f"--- Loading Data from: {target_dir} ---")
    
    # 2. Use the loader
    loader = BrukerLoader(target_dir)
    dic, data = loader.load_processed_data()
    
    if data is not None:
        print(f"Data Matrix Shape: {data.shape}")
        
        # 3. Integrate a sample window
        # For 19F PFOA, the peaks are usually sharp. 
        # We'll integrate a middle-of-the-road chunk of the spectrum.
        mid = data.shape[1] // 2
        traj = extract_trajectory(data, mid - 50, mid + 50)
        
        print("\nExtracted Intensities (Trajectory):")
        print(traj)
        
        # 4. Plot to verify the T1 curve shape
        plt.figure(figsize=(8, 5))
        plt.plot(traj, 'ro-', label='Raw Integrals')
        plt.title(f"T1 Recovery Check: {os.path.basename(os.path.dirname(target_dir))}")
        plt.xlabel("Experiment Index")
        plt.ylabel("Integrated Intensity")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("❌ Data is None. Ensure you have 'pdata' (1r or 2rr files) in the folder.")

if __name__ == "__main__":
    test_integration()