SmartVisionAgri: Plant Leaf Disease Classification
AAI-521: Computer Vision Applications – Final Project

Author: Samiksha Kodgire
Dataset: PlantVillage (Kaggle)
Models Used: Baseline CNN, MobileNetV2, EfficientNetB0, MobileNetV3-Large

 1. Project Overview

SmartVisionAgri is a deep-learning–based image classification system designed to automatically detect plant leaf diseases using computer vision.

Agricultural productivity heavily depends on early and accurate disease detection. Manual inspection is slow, subjective, and labor-intensive. This project leverages Convolutional Neural Networks (CNNs) and modern transfer learning architectures to automate disease identification across 15 plant-disease classes.

This work is designed as a graduate-level exploration of model design, training efficiency, interpretability, and comparative evaluation.

----------------------------------------------------------------------------------------------------------------------------------------
 2. Objectives

Detect whether a plant leaf is healthy or diseased

Classify the exact disease type (15-class classification)

Perform EDA, preprocessing, sampling, augmentation

Train and compare multiple models:

      - Baseline CNN

      - MobileNetV2 (Full fine-tuning)

      - EfficientNetB0

      - MobileNetV3-Large

Evaluate models using quantitative metrics and visualization

Build a full workflow appropriate for real-world deployment

----------------------------------------------------------------------------------------------------------------------------------------
 3. Repository Structure
    
 SmartVisionAgri/
│── README.md
│── notebook/
│     └── AAI_521_Final_Notebook.ipynb
│── models/
│     ├── baseline_best.h5
│     ├── mobilenet_v2_best.h5
│     ├── effnet_best.h5
│     └── mobilenet_v3_best.h5
│── data/
│     └── (Working or sample dataset - not uploaded due to size)
│── report/
│     └── SmartVisionAgri_Technical_Report.pdf

----------------------------------------------------------------------------------------------------------------------------------------
 4. Methods & Workflow
    4.1 Data Preparation

        Dataset copied from Google Drive → Runtime for faster access
        
        Safe copy of the original dataset maintained
        
        Stratified sampling (max 500 images/class) for balanced training
        
        Resizing all images to 128×128
        
        Data augmentation applied using ImageDataGenerator

    4.2 Modeling Approaches
        Baseline CNN
        
        3 Conv2D blocks
        
        ReLU activations
        
        Dropout regularization
        
        Trained from scratch
        
        Transfer Learning Models
        
        MobileNetV2 (best performance)
        
        EfficientNetB0
        
        MobileNetV3-Large
        
        Each includes:
        
        GlobalAveragePooling
        
        Dense(256) + Dropout
        
        Softmax output

    4.3 Training Features

        Early stopping
        
        Model checkpointing
        
        Validation monitoring
        
        GPU optimization for big datasets

----------------------------------------------------------------------------------------------------------------------------------------
 5. Evaluation Metrics

      1. Accuracy and Loss curves

      2. Classification Report

      3. Confusion Matrix (normalized)

      4. Random sample predictions with visualization

      5. Model comparison table (top-1 accuracy)

----------------------------------------------------------------------------------------------------------------------------------------
 6. Results Summary
Model	Validation Accuracy	Notes
Baseline CNN	~87%	Strong for scratch model
MobileNetV2	Best model	Good generalization
EfficientNetB0	Unstable	Likely under-trained / partially frozen
MobileNetV3-Large	Moderate	Suitable for mobile deployment

MobileNetV2 showed the most stable and accurate predictions.

----------------------------------------------------------------------------------------------------------------------------------------
 7. Key Learnings

Transfer learning dramatically reduces training time

Balanced datasets avoid overfitting

MobileNet families perform well on small-resolution inputs

CNNs remain competitive despite simple architecture

----------------------------------------------------------------------------------------------------------------------------------------
 8. Future Work

Add Grad-CAM or feature-visualization methods

Train with higher image resolution

Hyperparameter tuning (optimizers, LR scheduling)

Convert model for mobile deployment (TF-Lite)

Build a simple Streamlit or Gradio UI
