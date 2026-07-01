"""
Data download helper for Animals-10 dataset
Run this once to download the dataset
"""

import os
import zipfile
from pathlib import Path
import requests
import shutil

def download_animals_dataset():
    """Guide user to download animals-10 dataset"""
    
    data_dir = Path('data/animals')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if list(data_dir.glob('*')):
        classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
        print(f"✓ Dataset already exists in {data_dir}")
        print(f"  Found {len(classes)} categories: {', '.join(classes)}")
        return True
    
    print("\n" + "="*60)
    print("ANIMALS-10 DATASET DOWNLOAD")
    print("="*60)
    print("\nPlease follow these steps:")
    print("\n1. Go to: https://www.kaggle.com/datasets/alessiocorrado99/animals10")
    print("2. Click 'Download'")
    print("3. Extract the downloaded ZIP file")
    print(f"4. Move the extracted 'animals' folder to: {data_dir}")
    print("\nExpected folder structure:")
    print("data/animals/")
    print("  ├── dog/")
    print("  ├── cat/")
    print("  ├── horse/")
    print("  ├── spider/")
    print("  ├── butterfly/")
    print("  ├── chicken/")
    print("  ├── sheep/")
    print("  ├── cow/")
    print("  ├── squirrel/")
    print("  └── elephant/")
    
    input("\nPress Enter once you've downloaded and extracted the dataset...")
    
    if list(data_dir.glob('*')):
        classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
        print(f"\n✓ Dataset found! Found {len(classes)} categories")
        return True
    else:
        print("\n❌ Dataset not found. Please check the path and try again.")
        return False

if __name__ == "__main__":
    download_animals_dataset()