"""
Helper script to guide the user in downloading the dataset.
"""
import os
import sys
from pathlib import Path

def main():
    print("="*80)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print()
    print("This project requires the 'Fully-neuromorphic vision and control' dataset.")
    print("Due to size and licensing, we cannot host it directly in this repo.")
    print()
    print("Please follow these steps:")
    print("1. Visit the dataset URL:")
    print("   https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/QTFHQX")
    print()
    print("2. Download the 'sr_dataset_gt' folder (or the zip file containing it).")
    print()
    print("3. Extract/Place the data so that the folder structure looks like this:")
    print("   Drone_snn_control/")
    print("   ├── sr_dataset_gt/")
    print("   │   ├── sr_dataset_train/")
    print("   │   └── sr_dataset_test/")
    print("   ├── preprocessed/  (will be created by preprocess_data.py)")
    print("   ├── fpga/")
    print("   └── ...")
    print()
    
    # Check if data already exists
    current_dir = Path.cwd()
    data_path = current_dir / "sr_dataset_gt"
    
    if data_path.exists():
        print(f"✓ GOOD NEWS: It looks like 'sr_dataset_gt' already exists at:")
        print(f"  {data_path}")
        print("  You are ready to run 'python preprocess_data.py'")
    else:
        print(f"⚠ ACTION REQUIRED: 'sr_dataset_gt' folder NOT found in:")
        print(f"  {current_dir}")
    
    print()
    print("="*80)

if __name__ == "__main__":
    main()
