"""
Training utilities for CNN models
Handles training loop, early stopping, and model saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os
from tqdm import tqdm
import json
import time


class ModelTrainer:
    """
    Handles training, validation, early stopping, and model persistence
    """
    
    def __init__(self, model, device, model_name, save_dir='models_pickles'):
        """
        Args:
            model: PyTorch model to train
            device: 'cuda' or 'cpu'
            model_name: Name for saving (e.g., 'model_01_base')
            save_dir: Directory to save models
        """
        self.model = model
        self.device = device
        self.model_name = model_name
        self.save_dir = save_dir
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0
        self.best_epoch = 0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, targets in tqdm(train_loader, desc='Training', leave=False):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc='Validation', leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs=30, learning_rate=0.001,
              weight_decay=0.0, patience=5):
        """
        Full training loop with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization strength
            patience: Early stopping patience
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"Training: {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs} | LR: {learning_rate} | Weight Decay: {weight_decay}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                self.save_model()
                print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
                break
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return self.history
    
    def save_model(self):
        """Save model as pickle file"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'model_name': self.model_name,
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'device': str(self.device)
        }
        
        pickle_path = os.path.join(self.save_dir, f'{self.model_name}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save as PyTorch checkpoint
        torch_path = os.path.join(self.save_dir, f'{self.model_name}.pth')
        torch.save(model_data, torch_path)
        
        return pickle_path
    
    def load_model(self, model_path=None):
        """Load model from pickle file"""
        if model_path is None:
            model_path = os.path.join(self.save_dir, f'{self.model_name}.pkl')
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.history = model_data['history']
        self.best_val_acc = model_data['best_val_acc']
        self.best_epoch = model_data['best_epoch']
        
        print(f"✓ Loaded model: {model_data['model_name']}")
        print(f"  Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return model_data
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        accuracy = 100. * correct / total
        
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'true_labels': all_labels
        }