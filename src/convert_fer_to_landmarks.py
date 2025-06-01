import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import os
from tqdm import tqdm

# Ayarlar
FER_CSV_PATH = "../data/fer2013.csv"
SAVE_DIR = "../data/landmarks_from_fer"
os.makedirs(SAVE_DIR, exist_ok=True)

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# FER2013 CSV'yi oku
df = pd.read_csv(FER_CSV_PATH)

# Kaç örnek alalım? (örn: 1000 test için)
LIMIT = 1000
processed = 0

for index, row in tqdm(df.iterrows(), total=min(len(df), LIMIT)):
    if processed >= LIMIT:
        break

    pixels = np.array(list(map(int, row["pixels"].split())), dtype=np.uint8).reshape(48, 48)

    # RGB'ye çevir
    img = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)

    # Mediapipe ile landmark çıkar
    result = face_mesh.process(img)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = np.array([[pt.x, pt.y, pt.z] for pt in face_landmarks.landmark]).flatten()
            label = row["emotion"]
            filename = f"{label}_{index}.npy"
            np.save(os.path.join(SAVE_DIR, filename), landmarks)
            processed += 1

print(f"[✓] Toplam {processed} örnek işlendi ve kaydedildi.")
