# 🎭 Facial Expression Recognition & Emotion-Adaptive Game

## 🔍 Overview

This project implements a **real-time facial expression recognition (FER) system** that interacts with a Unity game.
A **CNN model trained with PyTorch** on the FER2013 dataset detects the player’s facial expressions from the webcam and sends the results via TCP to Unity.
The Unity scene reacts to these emotions by changing **lighting, colors, and sound** to enhance the psychological thriller experience.

---

## 🚀 Features

* Emotion classes: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`
* Ensemble CNN model with 5-fold cross-validation for stable predictions
* Real-time detection using OpenCV & PyTorch
* Unity integration via TCP sockets
* Atmosphere adaptation through Shader Graph in Unity

---

## 🧠 Tech Stack

* **AI & Data Processing:** PyTorch, torchvision, OpenCV, NumPy
* **Game Development:** Unity (URP, Shader Graph), TCP Networking
* **Version Control:** Git & GitHub
* Dataset stored locally (FER2013, excluded from Git)

---

## 📁 Project Structure

```
facial_expression/
│
├── data/               # Not tracked by Git (e.g., fer2013.csv)
├── models/             # Trained CNN model files (.pth)
├── notebooks/          # Jupyter notebooks for experiments
├── src/                # Python scripts (training, prediction, preprocessing)
│   ├── train_cnn_model_cv.py
│   ├── predict_cnn.py
│   └── ...
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🖥️ System Architecture

```
[Webcam] → [PyTorch CNN Model] → [Emotion Label] → (TCP Socket) → [Unity Game Engine] → [Lighting / Colors / Sound]
```

---

## ⚙️ Installation & Usage

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/facial_expression_game.git
cd facial_expression_game
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run real-time emotion prediction**

```bash
python src/predict_cnn.py
```

**4. Unity integration**

* Unity listens on the same TCP port as `predict_cnn.py`.
* The received emotion updates the game atmosphere in real-time.

---

## 📝 Notes

* `data/fer2013.csv` and other large files are **excluded from Git**.
* Only code and necessary scripts are included in this repository.
* Models are stored in `/models` and trained offline.

---
