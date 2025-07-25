# AI Framework for Diagnosing Stages of Diabetic Retinopathy

This project presents an AI-based diagnostic tool for detecting and classifying **Diabetic Retinopathy (DR)** stages from retinal fundus images. It combines deep learning and image processing to assist clinicians in early diagnosis, especially in resource-limited settings.

## 🧠 Key Features

- **Lesion Detection** using YOLOv8 (Microaneurysms, Hemorrhages, Exudates)
- **Hybrid Classification Model**: DenseNet-121 + GLCM texture features
- **Explainable AI** with Grad-CAM heatmaps
- **Hyperparameter Optimization** using Optuna
- **PyQt5 GUI** for easy interaction and offline usage
- **Packaged Desktop App** with Inno Setup (.exe installer)

## 🖼️ Tech Stack

- Python, PyTorch, OpenCV, Skimage, PIL
- DenseNet-121, ResNet50, EfficientNet-B3
- YOLOv8 for object detection
- Grad-CAM for explainability
- PyQt5 for GUI
- Optuna for hyperparameter tuning

## 📁 Directory Structure

├── segmentation/ # YOLOv8-based lesion detection
├── classification/ # DR stage classification code
├── gui/ # PyQt5-based desktop UI
├── models/ # Pre-trained weights
├── utils/ # Helper scripts and utilities
├── app.py # Entry point for the GUI
├── requirements.txt # Required Python packages
└── README.md # This file

## 📊 Results

- Achieved high classification accuracy using DenseNet-121 + GLCM
- Grad-CAM provides interpretable diagnosis support
- GUI alerts for severe DR (Severe NPDR or PDR)

## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/Diabetic-Retinopathy.git
   cd Diabetic-Retinopathy
Install requirements:

bash-
pip install -r requirements.txt
run:
python app.py
📚 Dataset
APTOS 2019 Blindness Detection Dataset from Kaggle

