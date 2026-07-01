"""
Data loading and preprocessing module for Animals-10 dataset
With stratified train/val/test splits
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Tuple, Dict, Optional, List
import os
from collections import defaultdict

class AnimalsDataset(Dataset):
    """Custom Dataset for Animals-10 with support for Italian class names"""
    
    # Map Italian to English class names
    CLASS_MAPPING = {
        'cane': 'dog',
        'cavallo': 'horse',
        'elefante': 'elephant',
        'farfalla': 'butterfly',
        'gallina': 'chicken',
        'gatto': 'cat',
        'mucca': 'cow',
        'pecora': 'sheep',
        'ragno': 'spider',
        'scoiattolo': 'squirrel'
    }
    
    def __init__(self, root_dir: str, transform=None, use_english_names=True):
        """
        Args:
            root_dir: Path to dataset (e.g., 'data/animals/')
            transform: Optional transforms to apply
            use_english_names: If True, convert Italian folder names to English
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_english_names = use_english_names
        
        # Get all class folders (could be Italian or English)
        raw_classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        # Determine if classes are Italian or English
        is_italian = any(cls in self.CLASS_MAPPING for cls in raw_classes)
        
        if is_italian and use_english_names:
            # Map Italian to English for display
            self.classes = [self.CLASS_MAPPING[cls] for cls in raw_classes]
            self.class_to_idx = {self.CLASS_MAPPING[cls]: idx for idx, cls in enumerate(raw_classes)}
            self.italian_to_english = {cls: self.CLASS_MAPPING[cls] for cls in raw_classes}
            print(f"✓ Detected Italian class names. Mapping to English:")
            for italian, english in self.italian_to_english.items():
                print(f"  {italian} -> {english}")
        else:
            # Use classes as-is
            self.classes = raw_classes
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.italian_to_english = None
        
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Collect all image paths and labels
        self.images = []
        self.labels = []
        
        for idx, class_name in enumerate(raw_classes):
            class_dir = self.root_dir / class_name
            class_idx = idx  # Use original index
            
            # Get all image files (support various extensions)
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + \
                         list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.JPG')) + \
                         list(class_dir.glob('*.PNG')) + list(class_dir.glob('*.JPEG'))
            
            for img_path in image_files:
                self.images.append(img_path)
                self.labels.append(class_idx)
        
        print(f"\n✓ Loaded {len(self.images)} images from {len(self.classes)} classes")
        if is_italian and use_english_names:
            print(f"  Display names: {', '.join(self.classes)}")
        else:
            print(f"  Classes: {', '.join(self.classes)}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Get class distribution with English names if mapped"""
        distribution = {}
        for idx, class_name in enumerate(self.classes):
            count = self.labels.count(idx)
            distribution[class_name] = count
        return distribution
    
    def get_labels(self):
        """Get all labels as numpy array (useful for stratification)"""
        return np.array(self.labels)
    
    def get_original_names(self):
        """Get original folder names (Italian if applicable)"""
        if self.italian_to_english:
            return list(self.italian_to_english.keys())
        return self.classes

class AnimalsDataLoader:
    """Main data loader class for Animals-10 dataset with stratified splits"""
    
    def __init__(self, data_dir='data/animals', batch_size=64, 
                 image_size=224, num_workers=2, use_english_names=True):
        """
        Args:
            data_dir: Path to dataset
            batch_size: Batch size for training
            image_size: Size to resize images to
            num_workers: Number of workers for data loading
            use_english_names: Convert Italian folder names to English
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.use_english_names = use_english_names
        self.classes = None
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_dir}. "
                "Please download the dataset first."
            )
    
    def get_transforms(self):
        """Get basic transforms (no augmentation)"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _stratified_split(self, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Perform stratified split to maintain class balance
        
        Args:
            dataset: The full dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
        
        Returns:
            train_indices, val_indices, test_indices
        """
        # Get labels
        labels = dataset.get_labels()
        n = len(labels)
        
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        # For each class, split indices
        for class_idx, indices in class_indices.items():
            # Shuffle indices within class
            random.shuffle(indices)
            
            # Calculate split sizes
            n_class = len(indices)
            n_train = int(train_ratio * n_class)
            n_val = int(val_ratio * n_class)
            n_test = n_class - n_train - n_val
            
            # Split
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])
        
        # Shuffle final indices
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)
        
        return train_indices, val_indices, test_indices
    
    def load_data(self, val_ratio=0.15, test_ratio=0.15):
        """
        Load dataset with stratified train/val/test splits
        
        Args:
            val_ratio: Proportion for validation (default: 0.15)
            test_ratio: Proportion for testing (default: 0.15)
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Set seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        
        # Get transforms
        transform = self.get_transforms()
        
        # Create full dataset
        full_dataset = AnimalsDataset(
            self.data_dir, 
            transform=transform,
            use_english_names=self.use_english_names
        )
        self.classes = full_dataset.classes
        
        # Print class distribution
        print("\nClass Distribution:")
        distribution = full_dataset.get_class_distribution()
        for cls, count in distribution.items():
            print(f"  {cls}: {count} images")
        
        # Perform stratified split
        train_ratio = 1 - val_ratio - test_ratio
        train_indices, val_indices, test_indices = self._stratified_split(
            full_dataset, train_ratio, val_ratio, test_ratio
        )
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        print(f"\nData Split (Stratified):")
        print(f"  Training: {len(train_dataset)} images ({len(train_dataset)/len(full_dataset)*100:.1f}%)")
        print(f"  Validation: {len(val_dataset)} images ({len(val_dataset)/len(full_dataset)*100:.1f}%)")
        print(f"  Test: {len(test_dataset)} images ({len(test_dataset)/len(full_dataset)*100:.1f}%)")
        
        # Verify stratification
        print("\nClass Distribution per Split:")
        for split_name, split_dataset in [("Training", train_dataset), 
                                          ("Validation", val_dataset), 
                                          ("Test", test_dataset)]:
            # Get labels from subset
            labels = [full_dataset.labels[idx] for idx in split_dataset.indices]
            unique, counts = np.unique(labels, return_counts=True)
            print(f"\n  {split_name}:")
            for class_idx, count in zip(unique, counts):
                class_name = self.classes[class_idx]
                print(f"    {class_name}: {count} images")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def visualize_samples(self, dataset, num_samples=25, save_path='outputs/samples.png'):
        """Visualize sample images"""
        class_names = self.classes
        
        # Get a batch
        data_loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
        images, labels = next(iter(data_loader))
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
        
        # Display
        cols = 5
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        fig.suptitle('Sample Animals-10 Images', fontsize=16)
        
        for idx in range(num_samples):
            row = idx // cols
            col = idx % cols
            
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            img = images[idx].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(class_names[labels[idx]], fontsize=10)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(num_samples, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows == 1:
                axes[col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Visualization saved to {save_path}")