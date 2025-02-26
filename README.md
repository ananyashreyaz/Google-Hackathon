# Google-Hackathon
# Google Hackathon - OCR for Doctor's Handwritten Prescriptions

This project aims to develop an Optical Character Recognition (OCR) system for recognizing medicine names from doctors' handwritten prescriptions.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Running the API](#running-the-api)
- [Testing the API](#testing-the-api)
- [Project Structure](#project-structure)

## Requirements
Ensure you have the following installed on your system:
- Python 3.8+
- Node.js (if working with the frontend)
- pip
- Virtual Environment (optional but recommended)
- PyTorch

## Installation

1. Clone the Repository
```sh
git clone <repository-url>
cd Google-Hackathon
```

2. Create a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


3. Install Dependencies
pip install torch torchvision flask numpy scikit-learn pillow


## Dataset
The dataset used in this project is **Doctor's Handwritten Prescription BD dataset**. Ensure that the dataset is properly placed in the project folder structure:
```
Doctor's Handwritten Prescription BD dataset/
├── Training/
├── Testing/
├── Validation/
```
Modify the dataset path in `model_training.py` if necessary.

## Model Training
To train the model, run:
```sh
python model_training.py
```
This will:
1. Load and preprocess the dataset
2. Train a Convolutional Neural Network (CNN)
3. Save the trained model as `ocr_model.pth`

## Running the API
Once the model is trained, you can start the Flask API to make predictions.

### 1. Start the API
```sh
python app.py
```
By default, the API will be available at `http://127.0.0.1:5000`

## Testing the API
You can test the API using **Postman** or **cURL**.

### 1. Upload an Image for Prediction
Using `cURL`:
```sh
curl -X POST -F "image=@path/to/image.png" http://127.0.0.1:5000/predict
```

Using **Postman**:
1. Open Postman
2. Select `POST` request
3. Enter `http://127.0.0.1:5000/predict`
4. In "Body" select `form-data`
5. Add a new key named `image` and upload an image file
6. Click "Send" and check the response

## Project Structure
```
Google Hackathon/
├── __pycache__/
├── .dist/
├── Doctor's Handwritten Prescription BD dataset/
│   ├── Testing/
│   ├── Training/
│   ├── Validation/
├── 136.png                    # Sample image
├── app.jsx                     # Frontend (if applicable)
├── app.py                      # Flask API
├── dataset_loader.py           # Dataset handling
├── model_training.py           # Model training script
├── ocr_model.pth               # Trained model
├── package.json                # Frontend dependencies (if applicable)
└── README.md                   # This documentation
```

## Notes
- Ensure all dependencies are installed before running.
- Modify dataset paths in `model_training.py` if required.
- Use a virtual environment to avoid conflicts.
- If `npm start` fails, check `package.json` and ensure a start script is defined.


