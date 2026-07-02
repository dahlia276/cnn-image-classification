"""
CNN Model Architectures - Models 1-4
Part of 10-model comparison for Animals-10 classification

Each model follows the brief requirements:
- Convolutional layers for feature extraction
- Pooling layers for downsampling
- Fully connected layers for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Model1_BaseCNN(nn.Module):
    """
    Model 1: Base CNN (No Regularization)
    
    Architecture:
    - 3 Convolutional layers: 32 → 64 → 128 filters
    - Max pooling after each conv layer
    - 2 Fully connected layers: 256 → 10 classes
    - No dropout, no batch norm
    
    Purpose: Baseline model to compare all other models against.
    Expected: Overfitting (high train acc, lower val acc)
    """
    
    def __init__(self, num_classes=10):
        super(Model1_BaseCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Model2_Dropout25(nn.Module):
    """
    Model 2: CNN with Dropout 0.25
    
    Architecture:
    - Same as Base CNN (3 conv, 2 FC)
    - Added Dropout (0.25) after first fully connected layer
    - Dropout randomly turns off 25% of neurons during training
    
    Purpose: Test if light regularization improves generalization.
    """
    
    def __init__(self, num_classes=10):
        super(Model2_Dropout25, self).__init__()
        
        # Convolutional layers (same as Model 1)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # FC layer with activation
        x = F.relu(self.fc1(x))
        
        # Apply dropout (25% - light regularization)
        x = self.dropout(x)
        
        # Final FC layer
        x = self.fc2(x)
        
        return x


class Model3_Dropout50(nn.Module):
    """
    Model 3: CNN with Dropout 0.5
    
    Architecture:
    - Same as Base CNN (3 conv, 2 FC)
    - Added Dropout (0.5) after first fully connected layer
    - Heavier dropout than Model 2 (50% vs 25%)
    
    Purpose: Test if heavy regularization improves generalization.
    """
    
    def __init__(self, num_classes=10):
        super(Model3_Dropout50, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer (50% drop rate - heavier regularization)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # FC layer with activation
        x = F.relu(self.fc1(x))
        
        # Apply dropout (50% - heavy regularization)
        x = self.dropout(x)
        
        # Final FC layer
        x = self.fc2(x)
        
        return x


class Model4_BatchNorm(nn.Module):
    """
    Model 4: CNN with Batch Normalization
    
    Architecture:
    - Same as Base CNN (3 conv, 2 FC)
    - Added Batch Normalization after each conv layer
    - BatchNorm normalizes outputs to have mean 0, variance 1
    - No dropout (BatchNorm has a slight regularizing effect)
    
    Purpose: Test if Batch Normalization improves training speed and
             generalization. Also tests if it can replace dropout.
    """
    
    def __init__(self, num_classes=10):
        super(Model4_BatchNorm, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Conv + BatchNorm + ReLU + Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    
class Model5_DataAug(nn.Module):
    """
    Model 5: CNN with Data Augmentation
    
    Architecture: Same as Model 1 (3 Conv + 2 FC)
    Difference: Trained with augmented data (flip, rotation, color jitter)
    
    Purpose: Test if data augmentation improves generalization
    without changing the architecture.
    """
    
    def __init__(self, num_classes=10):
        super(Model5_DataAug, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


class Model10_TransferLearning(nn.Module):
    """
    Model 10: Transfer Learning with ResNet18
    
    Uses a model pre-trained on ImageNet and fine-tunes it for 
    Animals-10 classification.
    
    Two approaches:
    1. Feature Extractor: Freeze all layers, train only the classifier
    2. Fine-tuning: Unfreeze and train all layers with a low learning rate
    """
    
    def __init__(self, num_classes=10, method='feature_extractor'):
        super(Model10_TransferLearning, self).__init__()
        
        # Load pre-trained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        
        # Method 1: Feature Extractor (freeze everything)
        if method == 'feature_extractor':
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        # Original: 1000 classes (ImageNet)
        # New: 10 classes (Animals-10)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        self.method = method
        
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✅ All layers unfrozen for fine-tuning")
    
    def freeze(self):
        """Freeze all layers except classifier"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Keep classifier trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        print("✅ All layers frozen except classifier")