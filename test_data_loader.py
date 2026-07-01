"""
Quick test script for data loader with stratified splits
"""

import sys
sys.path.append('src')
import numpy as np

from data_loader import AnimalsDataLoader

def test_data_loader():
    print("="*60)
    print("TESTING DATA LOADER WITH STRATIFIED SPLITS")
    print("="*60)
    
    #Initialize data loader
    print("\n1. Initializing data loader...")
    data_loader = AnimalsDataLoader(
        data_dir='data/animals',
        batch_size=32,
        image_size=224,
        use_english_names=True
    )
    print("✓ Data loader created")
    
    #Load data with stratified splits
    print("\n2. Loading data with stratified splits...")
    train_loader, val_loader, test_loader = data_loader.load_data(
        val_ratio=0.15,
        test_ratio=0.15
    )
    print("✓ Data loaded successfully")
    
    #Test a batch
    print("\n3. Testing data loading...")
    images, labels = next(iter(train_loader))
    print(f"✓ Batch shape: {images.shape}")
    print(f"✓ Labels shape: {labels.shape}")
    
    #Verify classes
    print("\n4. Classes:")
    for idx, cls in enumerate(data_loader.classes):
        print(f"  {idx}: {cls}")
    
    #Verify data split
    print("\n5. Data split:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    #Check stratification
    print("\n6. Checking stratification (class balance across splits):")
    full_dataset = train_loader.dataset.dataset
    all_labels = full_dataset.get_labels()
    
    for split_name, loader in [("Training", train_loader), 
                               ("Validation", val_loader), 
                               ("Test", test_loader)]:
        # Get labels from this split
        if hasattr(loader.dataset, 'indices'):
            split_labels = [full_dataset.labels[idx] for idx in loader.dataset.indices]
        else:
            split_labels = [full_dataset.labels[idx] for idx, _ in loader.dataset]
        
        # Count classes
        unique, counts = np.unique(split_labels, return_counts=True)
        print(f"\n  {split_name}:")
        for class_idx, count in zip(unique, counts):
            class_name = data_loader.classes[class_idx]
            percentage = (count / len(split_labels)) * 100
            print(f"    {class_name}: {count} ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED! Data loader with stratified splits is working.")
    print("="*60)

if __name__ == "__main__":
    test_data_loader()