# 🧠 Brain Tumor Classification using CNN and VGG16 Models

> A deep learning project for MRI-based brain tumor classification using custom CNN and VGG16 (with and without data augmentation).

---

## 📊 Project Overview

This project implements and compares:
- ✅ A custom CNN model
- ✅ Transfer learning with **VGG16**
- ✅ **VGG16 with data augmentation**

---

## 🧬 Dataset

- 📁 Source: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data)
- 🧪 Categories:
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor
  - No Tumor
- 🖼️ ~7153 images (70% train, 30% test)

---

<details>
<summary>📐 <strong>Model Architectures (Click to Expand)</strong></summary>

### 🧱 Custom CNN
- Conv2D + MaxPooling2D + Dropout + Dense
- Softmax for 4-class output
- ~25M parameters

### 🔄 VGG16 Transfer Learning
- Pretrained VGG16 (ImageNet)
- Removed top layers, added classification head
- Unfrozen last 4 conv layers for fine-tuning

### 🔄 VGG16 + Data Augmentation
- Same as above + real-time augmentation:
  - Rotation, Zoom, Flipping, Shear, Brightness, etc.

</details>

---

## 🏁 Training Details

| Model                | Training Acc | Test Acc | Notes                                |
|---------------------|--------------|----------|--------------------------------------|
| Custom CNN          | 99.90%       | 93.33%   | Slight overfitting                   |
| VGG16               | 97.32%       | 95.01%   | Best generalization                  |
| VGG16 + Augmentation| 94.65%       | 94.08%   | More robust, less overfitting        |

---

## 📈 Results

- ✅ Plotted accuracy/loss graphs
- ✅ Used EarlyStopping & ModelCheckpoint
- 🛑 Early Stopping prevented overfitting

---

## 🔮 Future Work

- Try ResNet / EfficientNet
- Add explainability (Grad-CAM, etc.)
- Use synthetic data generation
- Hyperparameter tuning

---

## 👨‍💻 Contributors

- M Rayyan