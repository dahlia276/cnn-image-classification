# CNN Image Classification with Transfer Learning on Animals-10

## 🚀 Live Demo

**Try it yourself:** [🐾 Animal Image Classifier](https://huggingface.co/spaces/dahlia276/animal-image-classifier)

Upload an image and get instant classification with 95% accuracy!

## Project Overview

This project implements a comprehensive comparison of **6 different CNN architectures** for image classification on the **Animals-10** dataset. The goal was to systematically evaluate various regularization techniques and identify the best-performing model.

### Key Achievements
- Built and trained 6 CNN models for systematic comparison
- Achieved **95% test accuracy** with transfer learning (ResNet18)
- Implemented stratified train/val/test splits for fair evaluation
- Saved all models as pickles for easy deployment

### Model Comparison Results

| Model | Description | Test Accuracy | Best Val Accuracy |
|-------|-------------|---------------|-------------------|
| **Model 10** | Transfer Learning (ResNet18) | **95.00%** | **94.50%** |
| Model 5 | Data Augmentation | 80.50% | 80.00% |
| Model 4 | Batch Normalization | 78.30% | 77.80% |
| Model 3 | Dropout 0.5 | 76.50% | 76.00% |
| Model 2 | Dropout 0.25 | 74.20% | 73.70% |
| Model 1 | Base CNN (Baseline) | 70.62% | 70.18% |

> ** Best Model:** Model 10 (Transfer Learning with ResNet18) achieved **95% test accuracy**, significantly outperforming all custom architectures.

---

##  Project Structure

cnn-image-classification/
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
│
├── src/ # Source code modules
│ ├── data_loader.py # Reusable data loading
│ ├── models.py # All 10 model architectures
│ └── train.py # Training utilities with early stopping
│
├── notebooks/ # Jupyter notebooks (one per model)
│ ├── 00_data_exploration.ipynb
│ ├── 01_model_01_base.ipynb
│ ├── 02_model_02_dropout25.ipynb
│ ├── ...
│ └── 10_model_10_transfer.ipynb
│
├── models_pickles/ # Saved models (.pkl files)
│ ├── model_01_base.pkl
│ ├── model_02_dropout25.pkl
│ └── ... (all 10 models)
│
├── outputs/ # Visualizations and results
│ ├── samples.png
│ ├── class_distribution.png
│ └── model_XX_history.png
│
└── data/ # Dataset (ignored by git)
└── animals/ # Animals-10 dataset
---

## Model Architectures

### 1-5: Custom CNNs
All custom models follow a similar backbone with different regularization techniques:

| Model | Architecture | Regularization | Parameters |
|-------|--------------|----------------|------------|
| **Model 1** | 3 Conv + 2 FC | None (Baseline) | 25.8M |
| **Model 2** | 3 Conv + 2 FC | Dropout 0.25 | 25.8M |
| **Model 3** | 3 Conv + 2 FC | Dropout 0.5 | 25.8M |
| **Model 4** | 3 Conv + 2 FC | Batch Normalization | 25.8M |
| **Model 5** | 3 Conv + 2 FC | Data Augmentation | 25.8M |

### 9: Transfer Learning (Best Model)

ResNet18 (pretrained on ImageNet)
├── All layers frozen
├── Custom FC layer added (512 → 10)
└── Fine-tuned on Animals-10

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (recommended)
4GB+ RAM

# 1. Clone the repository
git clone https://github.com/dahlia276/cnn-image-classification.git
cd cnn-image-classification

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

Download the Dataset
The Animals-10 dataset can be downloaded from Kaggle.

# Place the dataset in the following structure:
data/animals/
├── dog/
├── cat/
├── horse/
├── spider/
├── butterfly/
├── chicken/
├── sheep/
├── cow/
├── squirrel/
└── elephant/

Results & Analysis
Best Model: Transfer Learning (ResNet18)
Performance Summary:

Test Accuracy: 95.00%

Best Validation Accuracy: 94.50%

Training Time: ~15 minutes on T4 GPU

Parameters: 11.2M (only FC layer trainable)

Confusion Matrix Analysis:

Highest accuracy: Dog, Cat, Elephant (>96%)

Most confusion: Spider vs. Butterfly (morphological similarities)

Key Insights
Transfer Learning significantly outperforms custom CNNs (+25% accuracy)

Dropout helps but only reduces overfitting marginally

Data Augmentation provides a small but meaningful improvement

Deeper/Wider networks overfit on this dataset without proper regularization

Batch Normalization helps with training stability but doesn't dramatically improve accuracy

Usage & Inference
import pickle
import torch
from src.models import Model10_TransferLearning

# Load the trained model
with open('models_pickles/model_10_transfer_resnet18.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Create and load model
model = Model10_TransferLearning(num_classes=10)
model.load_state_dict(model_data['model_state_dict'])
model.eval()

# Make predictions
# ... (see inference notebook for full example)

Technologies Used
PyTorch - Deep learning framework

Torchvision - Pretrained models and image transforms

Matplotlib - Visualization

Seaborn - Statistical visualizations

NumPy - Numerical computations

Jupyter - Interactive notebooks