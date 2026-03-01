# Facial Expression Recognition System

Real-time facial expression recognition using EfficientNet-B0, trained on the FERPlus dataset and deployed via webcam with MTCNN face detection.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project trains a CNN-based emotion classifier on the FERPlus dataset and runs real-time inference from a webcam feed. It detects faces using MTCNN, classifies 7 emotion classes, and applies smoothing and neutral bias correction for stable predictions.

---

## Features

- **7 Emotion Classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **EfficientNet-B0** backbone fine-tuned on FERPlus
- **MTCNN** face detection (selects largest face in frame)
- **5-frame majority voting** for prediction smoothing
- **Neutral bias correction** to reduce false positives
- **Class-weighted loss** to handle dataset imbalance
- Real-time webcam inference with FPS display

---

## Tech Stack

- PyTorch, torchvision
- facenet-pytorch (MTCNN)
- OpenCV, NumPy, Pillow

---

## Project Structure

```
facial_expression/
├── data/               # Not tracked by Git (FERPlus dataset)
├── models/             # Not tracked by Git (.pth model files)
│   └── classes.json
├── notebooks/          # Jupyter notebooks for experiments
├── src/
│   ├── train_efficientnet_b0_ferplus.py
│   └── predict_efficientnet_b0_unity.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/mercanhumar/facial_expression.git
cd facial_expression
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Prepare dataset**

Download the [FERPlus dataset](https://github.com/microsoft/FERPlus) and place it under:
```
data/ferplus/train/<emotion>/
data/ferplus/val/<emotion>/
```

**4. Train the model**
```bash
python src/train_efficientnet_b0_ferplus.py
```

**5. Run real-time prediction**
```bash
python src/predict_efficientnet_b0_unity.py
```

---

## How It Works

```
[Webcam]
    |
    v
[MTCNN Face Detection] --> selects largest face
    |
    v
[EfficientNet-B0] --> softmax over 7 classes
    |
    v
[Neutral Bias + 5-frame Smoothing] --> stable emotion label
    |
    v
[OpenCV Display] --> bounding box + emotion label + FPS
```

---

## Notes

- FERPlus dataset and `.pth` model files are excluded via `.gitignore`
- Model is saved automatically when validation accuracy improves during training
- Set `SEND_TO_UNITY = True` in `predict_efficientnet_b0_unity.py` to enable TCP socket output

---

## License

MIT License