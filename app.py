"""
Gradio Web App for Animal Image Classifier
Uses Model 10 (Transfer Learning with ResNet18)
"""

import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import pickle

# CONFIGURATION
CLASS_NAMES = [
    'dog', 'horse', 'elephant', 'butterfly', 'chicken',
    'cat', 'cow', 'sheep', 'spider', 'squirrel'
]
CLASS_NAMES_ITALIAN = [
    'cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
    'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'
]
MODEL_PATH = 'models_pickles/model_10_transfer.pkl'
MODEL_PATH_PTH = 'models_pickles/model_10_transfer.pth'
IMAGE_SIZE = 224

# LOAD MODEL
def load_model():
    """Load the trained transfer learning model"""
    
    # Create the model architecture
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    
    # Try loading from .pkl first
    if os.path.exists(MODEL_PATH):
        print(f"📥 Loading model from: {MODEL_PATH}")
        try:
            with open(MODEL_PATH, 'rb') as f:
                checkpoint = pickle.load(f)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('backbone.'):
                        new_key = key[9:]  # Remove 'backbone.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                
                model.load_state_dict(new_state_dict)
                print("✅ Model loaded successfully from pickle!")
                model.eval()
                return model
                
        except Exception as e:
            print(f"⚠️ Error loading pickle: {e}")
    if os.path.exists(MODEL_PATH_PTH):
        print(f"📥 Loading model from: {MODEL_PATH_PTH}")
        try:
            # Load with map_location='cpu' and weights_only=False for compatibility
            checkpoint = torch.load(MODEL_PATH_PTH, map_location=torch.device('cpu'), weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[9:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            model.load_state_dict(new_state_dict)
            print("✅ Model loaded successfully from .pth!")
            model.eval()
            return model
            
        except Exception as e:
            print(f"⚠️ Error loading .pth: {e}")
    
    # If model not found, create a dummy model for testing
    print("⚠️ Model not found. Using dummy model for testing...")
    return model

# Load the model
print("="*50)
print("LOADING ANIMAL CLASSIFIER MODEL")
print("="*50)
model = load_model()

# IMAGE PREPROCESSING
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    """Preprocess the input image for the model"""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        image = Image.fromarray(image).convert('RGB')
    
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension

def predict_image(image):
    """
    Predict the animal class from an uploaded image
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        Dictionary with prediction results
    """
    try:
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get the predicted class
        predicted_class = predicted.item()
        class_name = CLASS_NAMES[predicted_class]
        class_name_italian = CLASS_NAMES_ITALIAN[predicted_class]
        confidence_score = confidence.item() * 100
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        top5_results = []
        for i in range(5):
            idx = top5_indices[0][i].item()
            top5_results.append({
                'class': CLASS_NAMES[idx],
                'class_italian': CLASS_NAMES_ITALIAN[idx],
                'confidence': top5_probs[0][i].item() * 100
            })
        result = {
            'predicted_class': class_name,
            'predicted_class_italian': class_name_italian,
            'confidence': confidence_score,
            'top_5': top5_results,
            'all_classes': CLASS_NAMES,
            'all_confidences': [prob.item() * 100 for prob in probabilities[0]]
        }
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'predicted_class': 'Error',
            'confidence': 0
        }

# 5. GRADIO INTERFACE

def classify_image(image):
    """Wrapper function for Gradio"""
    result = predict_image(image)
    
    if 'error' in result:
        return f"❌ Error: {result['error']}"
    output = f"""
    🏆 **Prediction: {result['predicted_class']}** ({result['predicted_class_italian']})
    📊 **Confidence: {result['confidence']:.1f}%**
    
    ---
    ### Top 5 Predictions:
    """
    
    for i, pred in enumerate(result['top_5']):
        bar_length = int(pred['confidence'] / 5)  # Scale for display
        bar = '█' * bar_length + '░' * (20 - bar_length)
        output += f"\n{i+1}. **{pred['class']}** ({pred['class_italian']}): {pred['confidence']:.1f}% {bar}"
    
    return output

# CREATE THE GRADIO APP
custom_css = """
.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
}
h1 {
    text-align: center;
    color: #2c3e50;
}
"""

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(
        label="Upload an Animal Image",
        type="pil",
        sources=["upload", "webcam"]
    ),
    outputs=gr.Markdown(
        label="Prediction Results",
        value="Upload an image to see the prediction..."
    ),
    title="🐾 Animal Image Classifier",
    description="""
    Upload a picture of an animal and the model will classify it into one of 10 categories:
    
    🐕 Dog | 🐎 Horse | 🐘 Elephant | 🦋 Butterfly | 🐔 Chicken | 🐈 Cat | 🐄 Cow | 🐑 Sheep | 🕷️ Spider | 🐿️ Squirrel
    
    **Best Model Accuracy: 95.32%** (Transfer Learning with ResNet18)
    """,
    article="""
    ---
    ### How It Works
    1. Upload an image of an animal
    2. The model (ResNet18 pre-trained on ImageNet) processes it
    3. Results show the predicted class with confidence score
    
    **Built with:** PyTorch, ResNet18, Gradio
    **Dataset:** Animals-10 (26,179 images)
    """
)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🐾 ANIMAL IMAGE CLASSIFIER")
    print("="*50)
    print("Launching Gradio app...")
    print("="*50)
    
    iface.launch(
        server_name="0.0.0.0",
        share=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green"
        ),
        css=custom_css
    )