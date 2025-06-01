# facial_expression/src/predict_cnn.py

import warnings
warnings.filterwarnings("ignore")    # → Tüm warning’leri kapatır

import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from train_cnn_model_cv import EmotionCNN, COMMON_NORMALIZE
from torchvision import transforms

# Duygu etiketleri (FER2013 standardı)
LABELS  = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# GPU/CPU seçimi
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻  Çalışan cihaz:", DEVICE)

# ---------- Ensemble modellerini yükle ----------
FOLD_PTHS = [Path(f"../models/cnn_fold{i}.pth") for i in range(1, 6)]
models = []
for pth in FOLD_PTHS:
    if pth.exists():
        m = EmotionCNN().to(DEVICE)
        m.load_state_dict(torch.load(pth, map_location=DEVICE))
        m.eval()
        models.append(m)
print(f"🧩  Yüklenen fold sayısı: {len(models)}")

if not models:
    raise RuntimeError("En az bir cnn_fold*.pth bulunamadı!")

# ---------- Haar-cascade yüz algılama ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------- Predict aşamasındaki normalize pipeline ----------
predict_tf = transforms.Compose([
    transforms.Resize((48, 48)),  # PIL.Image.Image → 48×48
    transforms.ToTensor(),        # [0,255] → [0.0,1.0], boyut (1,48,48)
    COMMON_NORMALIZE              # [0,1] aralığını [-1,+1] aralığına çevirir
])

# ---------- Kamera Akışı ----------  
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("🚫  Kamera okunamadı — çıkılıyor.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            # ------- Önişleme -------
            roi48 = cv2.resize(roi, (48, 48))       # Görüntüyü 48×48’e getir
            pil   = Image.fromarray(roi48)          # PILImage formatına dönüştür

            tens  = predict_tf(pil).unsqueeze(0).to(DEVICE)  # [1,1,48,48] tensör

            # ------- Ensemble Tahmin -------
            with torch.no_grad():
                # 5 fold modelinin ortalaması (logits)
                logits = sum(m(tens) for m in models) / len(models)
                prob   = torch.softmax(logits, dim=1)[0].cpu().numpy()

            idx, conf = int(prob.argmax()), prob.max() * 100
            label     = LABELS[idx]

            # ------- Görselleştirme -------
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{label}  {conf:.1f}%",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

        cv2.imshow("Facial Expression (ensemble)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
