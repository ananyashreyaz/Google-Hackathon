from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model_training import build_model
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained PyTorch model
model = torch.load("ocr_model.pth", map_location=torch.device("cpu"))
model.eval()

# Load Label Encoder (if used)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Define image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if needed
    transforms.Resize((128, 128)),  # Resize to match training input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# API Endpoint to Handle Image Upload and Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Read and preprocess image
    file = request.files["image"]
    image = Image.open(file)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    # Convert class index to label
    class_name = label_encoder.inverse_transform([predicted_class])[0]

    return jsonify({"prediction": class_name})

# Run Flask server
if __name__ == "__main__":
    app.run(debug=True)

