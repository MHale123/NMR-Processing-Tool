import os
import nmrglue as ng

def find_bruker_folder(base_path):
    """
    Automatically searches for a folder containing an 'acqu' file 
    within the provided directory.
    """
    for root, dirs, files in os.walk(base_path):
        if 'acqu' in files:
            return root
    return None

def run_desktop_test():
    # 1. Dynamically find your Desktop 'Test' folder
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Test")
    
    print(f"--- Checking for Data in: {desktop_path} ---")
    
    # 2. Find the actual experiment folder (the one containing 'acqu')
    target_dir = find_bruker_folder(desktop_path)
    
    if not target_dir:
        print("❌ FAILED: Could not find any folder with an 'acqu' file inside 'Desktop/Test'.")
        print("Double check that you moved the folders correctly.")
        return

    print(f"✅ Found NMR Data at: {target_dir}")

    try:
        # 3. Load the metadata
        dic, data = ng.bruker.read(target_dir)
        acqus = dic['acqus']
        
        # Pull values from your specific experiment [cite: 1, 20]
        pulprog = acqus.get('PULPROG', 'Unknown')
        d1 = acqus['D'][1]   # Recycle delay 
        d16 = acqus['D'][16] # T1 delay 
        
        print(f"\n--- Metadata Results ---")
        print(f"Pulse Program: {pulprog}")
        print(f"D1 (Recycle Delay): {d1} s")
        print(f"D16 (Variable Delay): {d16} s")
        
        # Verify against your uploaded acqu file parameters 
        if 't1ir' in pulprog.lower() and d1 == 1.0:
            print("\n🎉 SUCCESS: The loader is communicating with your data!")
        
    except Exception as e:
        print(f"❌ Error reading the binary files: {e}")

if __name__ == "__main__":
    run_desktop_test()