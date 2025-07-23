import sys
import os
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QWidget, QFileDialog, QTabWidget, QMessageBox, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from io import BytesIO

device = 'cpu'

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax().item()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        one_hot.requires_grad_(True)
        
        loss = torch.sum(one_hot * output)
        loss.backward(retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=[2, 3])[0, :]
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        
        for k, w in enumerate(weights):
            cam += w * self.activations[0, k, :, :]
        
        cam = torch.clamp(cam, min=0)
        cam = cam / cam.max()
        cam = cam.detach().numpy()
        
        return cam

    def overlay_cam(self, image, cam):
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, cam_heatmap, 0.4, 0)
        return overlay

class CombinedClassifier(nn.Module):
    def __init__(self, n_classes):
        super(CombinedClassifier, self).__init__()
        self.base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.base_model.classifier = nn.Identity()
        self.feature_size = 1024
        self.glcm_size = 6
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_size + self.glcm_size, momentum=0.999, eps=0.001),
            nn.Linear(self.feature_size + self.glcm_size, 1024),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )

    def forward(self, x, glcm_feats):
        x = self.base_model(x)
        combined = torch.cat((x, glcm_feats), dim=1)
        return self.classifier(combined)

class GradCamThread(QThread):
    gradcam_computation_done = pyqtSignal(object)

    def __init__(self, image_tensor, model, predicted_class, parent=None):
        super().__init__(parent)
        self.image_tensor = image_tensor
        self.model = model
        self.predicted_class = predicted_class

    def run(self):
        try:
            target_layer = self.model.features
            gradcam = GradCAM(self.model.to(device), target_layer.to(device))
            cam = gradcam.generate_cam(self.image_tensor.to(device), self.predicted_class)
            self.gradcam_computation_done.emit(cam)
        except Exception as e:
            print(f"Error in GradCAM computation: {e}")
            self.gradcam_computation_done.emit(None)

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DR Severity Image Classifier")
        self.setGeometry(100, 100, 900, 750)
        self.device_name = device
        self.model = CombinedClassifier(n_classes=5)
        self.model = torch.load('optuna_models/combined_densenet_model.pth', map_location=device)
        self.model.eval()
        self.segmentation_model = YOLO('runs/detect/train/weights/best.pt')
        self.segmentation_classes = ["MA", "HE", "EX", "SE", "OD"]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.current_prediction = None
        self.current_glcm_tensor = None
        self.init_ui()

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #a5a5a5;
            }
            QLabel {
                font-size: 16px;
                color: #333;
            }
            QPushButton {
                font-size: 15px;
                padding: 10px;
                background-color: #4a90e2;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
            QTextEdit {
                font-size: 14px;
                background-color: #ffffff;
                border: 1px solid #ccc;
                padding: 10px;
            }
            QTabWidget::pane {
                border: none;
            }
            QTabBar::tab {
                padding: 10px 20px;
                background: #ddd;
                font-size: 14px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 140px;  
            }
            QTabBar::tab:selected {
                background: #ffffff;
            }
        """)

        self.info_tab = QWidget()
        self.tabs.addTab(self.info_tab, "Project Info")
        info_layout = QVBoxLayout(self.info_tab)
        info_layout.setContentsMargins(20, 20, 20, 20)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setText(
    "📌 Project Title: AI Framework for diagnosing stages of Diabetic Retinopathy\n\n"
    "🧠 Overview:\n"
    "This application leverages deep learning to assist in the diagnosis of Diabetic Retinopathy (DR) through image classification and lesion segmentation. "
    "It is designed for ophthalmologists, researchers, and healthcare providers seeking automated assistance in retinal image analysis.\n\n"
    "🔍 Key Components:\n"
    "1. Classification Model:\n"
    "A DenseNet-121 based architecture, enhanced with GLCM (feature extraction) texture features, is trained to classify DR into five categories:\n"
    "  - No_DR\n"
    "  - Mild_NPDR (Non-Proliferative Diabetic Retinopathy)\n"
    "  - Moderate_NPDR\n"
    "  - Severe_NPDR\n"
    "  - PDR (Proliferative Diabetic Retinopathy)\n"
    "The model was optimized using Optuna (CI), an automated hyperparameter optimization framework, ensuring high accuracy and robustness.\n\n"
    "2. Segmentation Model:\n"
    "A YOLOv8n model is used for real-time lesion detection and segmentation. It identifies key DR-related abnormalities such as:\n"
    "  - MA: Microaneurysms\n"
    "  - HE: Hemorrhages\n"
    "  - EX: Hard Exudates\n"
    "  - SE: Soft Exudates\n"
    "  - OD: Optic Disc\n"
    "This allows for better interpretability and explainability of the classification results.\n\n"
    "⚙️ Features:\n"
    "- Load and visualize fundus images directly in the UI.\n"
    "- Classify DR severity using a hybrid model combining CNN and texture-based features.\n"
    "- Segment and overlay lesion predictions using YOLOv8.\n"
    "- Alert popup for severe or proliferative DR detection to prompt urgent attention.\n"
    "- GradCAM Explanation\n\n"
    "💡 Technologies Used:\n"
    "- PyTorch, TorchVision, OpenCV, PIL, Skimage\n"
    "- PyQt5 for GUI\n"
    "- YOLOv8 (Ultralytics) for segmentation\n"
    "- Optuna (CI) for hyperparameter tuning\n"
    "- GradCAM for explainability"
)
        self.info_text.setStyleSheet("border: 1px solid #ccc; background-color: #c8c8c8;")
        info_layout.addWidget(self.info_text)

        self.classification_tab = QWidget()
        self.tabs.addTab(self.classification_tab, "Classifier")
        layout = QVBoxLayout(self.classification_tab)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        self.image_label = QLabel(f"Load an image to begin\nUsing device {self.device_name}")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(500, 400)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #c8c8c8;")
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.load_button = QPushButton("📁 Load Image")
        layout.addWidget(self.load_button, alignment=Qt.AlignCenter)
        self.load_button.clicked.connect(self.load_image)

        self.classify_button = QPushButton("🧠 Perform Inference")
        layout.addWidget(self.classify_button, alignment=Qt.AlignCenter)
        self.classify_button.clicked.connect(self.classify_image)

        self.gradcam_button = QPushButton("📊 Explain with GradCAM")
        layout.addWidget(self.gradcam_button, alignment=Qt.AlignCenter)
        self.gradcam_button.clicked.connect(self.generate_gradcam)

        self.prediction_label = QLabel("")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_label, alignment=Qt.AlignCenter)

        self.legend_label = QLabel()
        self.legend_label.setText("""
        <b>Segmentation Legend:</b><br>
        MA (Microaneurysms) &nbsp;&nbsp;
        HE (Hemorrhages) &nbsp;&nbsp;
        EX (Exudates)<br>
        SE (Soft Exudates) &nbsp;&nbsp;
        OD (Optic Disc)
        """)
        self.legend_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.legend_label, alignment=Qt.AlignCenter)

        self.gradcam_tab = QWidget()
        self.tabs.addTab(self.gradcam_tab, "GradCAM Explanation")
        gradcam_layout = QVBoxLayout(self.gradcam_tab)
        self.gradcam_label = QLabel("GradCAM Explanation will appear here")
        self.gradcam_label.setAlignment(Qt.AlignCenter)
        self.gradcam_label.setStyleSheet("border: 1px solid #ccc; background-color: #c8c8c8;")
        gradcam_layout.addWidget(self.gradcam_label, alignment=Qt.AlignCenter)

        self.current_image = None

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            pixmap = QPixmap(file_name).scaled(500, 400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.current_image = Image.open(file_name).convert('RGB')

    def classify_image(self):
        if self.current_image is None:
            self.prediction_label.setText("Please load an image first.")
            return

        image_tensor = self.transform(self.current_image).unsqueeze(0)

        gray = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2GRAY)
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        glcm_feats = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0],
            graycoprops(glcm, 'ASM')[0, 0],
        ]
        glcm_tensor = torch.tensor(glcm_feats, dtype=torch.float32).unsqueeze(0)
        self.current_glcm_tensor = glcm_tensor
        with torch.no_grad():
            output = self.model(image_tensor.to(device), glcm_tensor.to(device))

        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        self.current_prediction = predicted_class
        confidence = probabilities[0][predicted_class].item()
        class_names = ['Mild_NPDR', 'Moderate_NPDR', 'No_DR', 'PDR', 'Severe_NPDR']
        result_text = f"Predicted Class: {class_names[predicted_class]}\nConfidence: {confidence:.2%}"
        self.prediction_label.setText(result_text)

        results = self.segmentation_model(self.current_image, verbose=False)
        result_image = results[0].plot(labels=True, conf=False)
        height, width, channel = result_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(500, 400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

        self.show_severity_popup(class_names[predicted_class], confidence)

    def generate_gradcam(self):
        if self.current_image is None:
            self.gradcam_label.setText("Please load an image first.")
            return
        if self.current_prediction is None:
            self.gradcam_label.setText("Please perform inference first before GradCAM analysis.")
            return
        image_tensor = self.transform(self.current_image).unsqueeze(0)
        self.gradcam_label.setText("GradCAM Explanation Loading...")
        self.gradcam_thread = GradCamThread(image_tensor.to(device), self.model.base_model.to(device), 
                                            self.current_prediction)
        self.gradcam_thread.gradcam_computation_done.connect(self.display_gradcam)
        self.gradcam_thread.start()

    def display_gradcam(self, cam):
        if cam is None:
            self.gradcam_label.setText("Error in GradCAM computation. Please try again.")
            return

        image_array = np.array(self.current_image)
        image_resized = cv2.resize(image_array, (224, 224))
        
        gradcam_wrapper = GradCAM(self.model, self.model.base_model.features)
        overlay_image = gradcam_wrapper.overlay_cam(image_resized, cam)
        
        overlay_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        height, width, channel = overlay_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(overlay_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        self.gradcam_label.setPixmap(QPixmap.fromImage(q_image).scaled(500, 400, Qt.KeepAspectRatio))

    def show_severity_popup(self, predicted_class, confidence):
        if predicted_class in ['PDR', 'Severe_NPDR']:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Severity Classification")
            msg.setText(f"❗ {predicted_class} detected.\nSevere stage of DR. Immediate attention needed!\nConfidence: {confidence:.2%}")
            msg.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())