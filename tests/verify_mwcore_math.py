
import sys
import os
import numpy as np

# 1. Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
mwcore_root = os.path.dirname(current_dir)
legacy_project_root = os.path.join(os.path.dirname(mwcore_root), 'radar_legacy')

sys.path.append(mwcore_root)
sys.path.append(legacy_project_root)

# 2. Imports
# Import Legacy from reading project
try:
    from legacy import pc_generation as legacy
    print("Legacy module loaded successfully.")
except ImportError as e:
    print(f"Failed to import legacy module: {e}")
    sys.exit(1)

# Import New Reader from mwCore
from mwcore.registry import READERS

def verify_mwcore_vs_legacy(bin_file, num_frames=5):
    print(f"\nVerifying mwCore vs Legacy on: {bin_file}")
    
    # --- Legacy Setup ---
    legacy_cfg = legacy.PointCloudProcessCFG()
    legacy_reader = legacy.RawDataReader(bin_file)
    
    # --- mwCore Setup ---
    # Build via Registry to ensure full integration works
    mwcore_reader = READERS.build(dict(
        type='RawBinReader',
        file_path=bin_file
    ))
    
    pass_count = 0
    fail_count = 0
    
    for i in range(num_frames):
        # A. Legacy Output
        l_bin_frame = legacy_reader.getNextFrame(legacy_cfg.frameConfig)
        l_np_frame = legacy.bin2np_frame(l_bin_frame)
        legacy_pc = legacy.frame2pointcloud(l_np_frame, legacy_cfg)
        
        # B. mwCore Output
        # mwCore reader.read() increments internal counter automatically
        mwcore_pc = mwcore_reader.read()
        
        # C. Compare
        if legacy_pc.shape != mwcore_pc.shape:
             print(f"Frame {i} FAIL: Shape mismatch. Legacy {legacy_pc.shape}, mwCore {mwcore_pc.shape}")
             fail_count += 1
             continue
             
        if legacy_pc.size == 0:
            print(f"Frame {i} PASS: Both Empty")
            pass_count += 1
            continue
            
        if i == 0 and legacy_pc.shape[1] > 0:
            print("\nVisual Inspection (First 3 points):")
            print("--- LEGACY ---")
            print(legacy_pc.T[:3, :])
            print("--- MWCORE ---")
            print(mwcore_pc.T[:3, :])
            print("------------------\n")

        diff = np.abs(legacy_pc - mwcore_pc)
        max_diff = np.max(diff)
        
        if max_diff < 1e-5:
            print(f"Frame {i} PASS: Max diff {max_diff}")
            pass_count += 1
        else:
            print(f"Frame {i} FAIL: Max diff {max_diff}")
            fail_count += 1
            
    print("-" * 30)
    print(f"Total: {pass_count} PASS, {fail_count} FAIL")

if __name__ == "__main__":
    # Use the sample bin file in mwCore/tests/data
    bin_file = os.path.join(current_dir, 'data', 'sample.bin')
    if not os.path.exists(bin_file):
        print(f"Error: Test file not found at {bin_file}")
        sys.exit(1)
        
    verify_mwcore_vs_legacy(bin_file)
