# 🎭 Facial Expression Recognition & Emotion-Adaptive Game

> **Real-time facial expression recognition system** integrated into a Unity-based psychological thriller game, dynamically adapting the atmosphere based on the player’s emotions.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org/)  
[![Unity](https://img.shields.io/badge/Unity-2022+-black)](https://unity.com/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

---

## 🔍 Overview

This project implements a **real-time facial expression recognition (FER) system** that interacts with a Unity game.  
A **CNN model trained with PyTorch** on the FER2013 dataset detects the player’s facial expressions from the webcam,  
and sends the results via TCP to Unity, where **lighting, colors, and sound** adapt dynamically to the detected emotion.

---

## 🚀 Features

- **Emotion Classes:** `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`
- **Ensemble CNN Model** using 5-fold cross-validation for stable predictions
- **Real-time detection** with OpenCV & PyTorch
- **Unity Integration** using TCP sockets
- **Atmosphere Adaptation** via Shader Graph in Unity

---

## 🧠 Tech Stack

**AI & Data Processing**
- PyTorch, torchvision  
- OpenCV, NumPy  

**Game Development**
- Unity (URP, Shader Graph)  
- TCP Networking for live data transfer  

**Version Control**
- Git & GitHub  
- `.gitignore` excludes large datasets  

---

## 📁 Project Structure

```

facial\_expression/
│
├── data/               # Not tracked by Git (e.g., fer2013.csv)
├── models/             # Trained CNN model files (.pth)
├── notebooks/          # Jupyter notebooks for experiments
├── src/                # Python scripts (training, prediction, preprocessing)
│   ├── train\_cnn\_model\_cv.py
│   ├── predict\_cnn.py
│   └── ...
├── .gitignore
├── README.md
└── requirements.txt

```

---

## 🖥️ System Architecture

```

\[Webcam]
│ (video frames)
▼
\[PyTorch CNN Model] -- predicts --> \[Emotion Label]
│
▼ (TCP Socket)
\[Unity Game Engine] -- updates --> \[Lighting / Colors / Sound]

````

---

## ⚙️ Installation & Usage

**1️⃣ Clone the repository**
```bash
git clone https://github.com/yourusername/facial_expression_game.git
cd facial_expression_game
````

**2️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

**3️⃣ Run real-time emotion prediction**

```bash
python src/predict_cnn.py
```

**4️⃣ Unity integration**

* Unity listens on the same TCP port as `predict_cnn.py` output.
* The received emotion updates the scene atmosphere.

---

## 📝 Notes

* FER2013 dataset is stored locally and **not included** in this repository.
* All large files (e.g., `.csv`, `.pth`) are excluded via `.gitignore`.
* Models are trained offline and stored in `/models`.

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---
