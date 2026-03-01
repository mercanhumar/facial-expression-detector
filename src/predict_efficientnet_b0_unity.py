import cv2
import time
import json
import socket
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image

# ================================
# 1) GENEL AYARLAR
# ================================
MODEL_PATH   = "../models/efficientnet_b0_ferplus.pth"
CLASSES_PATH = "../models/classes.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Device: {DEVICE}")

SEND_TO_UNITY = False   # Unity bağlantısı aktif değilse False bırak

SMOOTH_N      = 5       # 5-frame smoothing
CONF_NEUTRAL  = 0.60    # Bu eşiğin altını direkt neutral yap
NEUTRAL_DELTA = 0.08    # Neutral bias: P(neutral) >= P(top) - delta ise neutral seç


# ================================
# 2) SINIFLAR VE MAPPING
# ================================
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    EMO_LABELS = json.load(f)
print("🧾 Emotions:", EMO_LABELS)

POSITIVE = {"happy", "surprise"}
NEGATIVE = {"angry", "disgust", "fear", "sad"}
NEUTRAL  = {"neutral"}

def emotion_to_group(emotion_label: str) -> str:
    if emotion_label in POSITIVE:
        return "positive"
    if emotion_label in NEGATIVE:
        return "negative"
    return "neutral"


# ================================
# 3) EfficientNet-B0 Modeli
# ================================
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(EMO_LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("✅ EfficientNet-B0 model yüklendi.")

face_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================
# 4) MTCNN – Yüz tespit
# ================================
mtcnn = MTCNN(
    keep_all=True,
    device=DEVICE,
    thresholds=[0.7, 0.8, 0.9],
    post_process=False
)
print("✅ MTCNN yüz dedektörü hazır.")


# ================================
# 5) Webcam Aç
# ================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı!")

print("🎥 Webcam açık — Q ile çık.")


# ================================
# 6) LOOP
# ================================
recent_preds = deque(maxlen=SMOOTH_N)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # FPS
    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Yüz tespiti
    try:
        boxes, probs = mtcnn.detect(rgb)
    except Exception:
        boxes, probs = None, None

    current_emotion = "neutral"
    current_group   = "neutral"
    current_conf    = 0.0

    if boxes is not None and len(boxes) > 0:

        # En büyük yüz
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best_index = int(np.argmax(areas))
        box = boxes[best_index].astype(int)

        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        face_crop = rgb[y1:y2, x1:x2]

        if face_crop.size > 0:
            face_pil = Image.fromarray(face_crop)
            tens = face_tf(face_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(tens)
                probs_tensor = F.softmax(logits, dim=1)[0]

            # TOP-1 ve TOP-2
            topk_probs, topk_idx = torch.topk(probs_tensor, k=2)
            top1_prob = float(topk_probs[0])
            top1_idx  = int(topk_idx[0])
            top2_prob = float(topk_probs[1])
            top2_idx  = int(topk_idx[1])

            top1_label = EMO_LABELS[top1_idx]
            top2_label = EMO_LABELS[top2_idx]

            # Neutral bias kuralı
            neutral_idx = EMO_LABELS.index("neutral")
            p_neutral   = float(probs_tensor[neutral_idx])

            final_emo = top1_label
            final_conf = top1_prob

            # Eğer neutral çok yakınsa -> neutral seç
            if final_emo != "neutral" and p_neutral >= final_conf - NEUTRAL_DELTA:
                final_emo   = "neutral"
                final_conf  = p_neutral

            # Düşük güven: tamamen neutral'a çek
            if final_conf < CONF_NEUTRAL:
                final_emo   = "neutral"
                final_conf  = p_neutral  # ya da final_conf bırakabilirsin

            # Smoothing kuyruğu
            recent_preds.append((final_emo, final_conf))

            # Kuyrukta çoğunluk oylaması
            labels = [l for l, _ in recent_preds]
            smoothed_emo = max(set(labels), key=labels.count)

            confs = [c for (l, c) in recent_preds if l == smoothed_emo]
            smoothed_conf = sum(confs) / len(confs)

            current_emotion = smoothed_emo
            current_group   = emotion_to_group(smoothed_emo)
            current_conf    = smoothed_conf

            # Çerçeve + yazılar
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Ana gösterim (group + label)
            cv2.putText(
                frame,
                f"{current_group.upper()} / {current_emotion} [{current_conf*100:.1f}%]",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            # DEBUG amaçlı: Top-2 olasılıkları göster (istersen kapat)
            debug_text = f"1:{top1_label} {top1_prob*100:.1f}%  2:{top2_label} {top2_prob*100:.1f}%"
            cv2.putText(
                frame,
                debug_text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    else:
        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
        break

cap.release()
cv2.destroyAllWindows()
print("🔚 Kapatıldı.")
