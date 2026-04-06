# Improving Class Imbalance in Parkinson’s Disease Sensor Data via Conditional MLP-VAE

## 📌 Project Overview
This project investigates whether deep learning and generative data augmentation can improve the classification of early-stage Parkinson’s Disease (PD) using finger-tapping sensor data.

The task focuses on distinguishing between:
- UPDRS = 0 (non-PD)
- UPDRS = 1 (early PD)

The main challenge is **class imbalance**, which can bias machine learning models. To address this, this project explores **Variational Autoencoder (VAE)-based data augmentation**, particularly a Conditional MLP-VAE.

---

## 🎯 Objectives
- Preprocess accelerometer-based finger-tapping data
- Extract clinically meaningful features (15-dimensional feature vectors)
- Train baseline models (MLP and LSTM)
- Develop VAE-based augmentation models (MLP-VAE and LSTM-VAE)
- Evaluate whether synthetic data improves classification performance

---

## 🧠 Methodology

### Data Processing
- Sliding window segmentation (5 seconds, 90% overlap)
- Feature extraction (temporal, frequency, and kinematic features)
- Z-score normalisation
- Patient-aware train/test split

Final dataset:
- 2146 samples
- 15-dimensional feature vectors

---

### Models Implemented
- Baseline Models:
  - Multilayer Perceptron (MLP)
  - Long Short-Term Memory (LSTM)

- Augmented Models:
  - Conditional MLP-VAE
  - Conditional LSTM-VAE

---

### Evaluation Metrics
- Sensitivity
- Specificity
- Matthews Correlation Coefficient (MCC)
- Area Under Curve (AUC)

All experiments are repeated across **10 random seeds** for robustness.

---

## 📊 Results Summary
The best-performing model is:

👉 **MLP-VAE**

Performance:
- Accuracy: 0.935
- AUC: 0.981
- Sensitivity: 0.957
- Specificity: 0.912
- MCC: 0.870

VAE-based augmentation significantly:
- Reduced false negatives
- Improved classification balance
- Enhanced robustness across runs :contentReference[oaicite:1]{index=1}  

---
