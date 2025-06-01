import cv2
import mediapipe as mp
import os
import numpy as np
import uuid

# === AYARLAR ===
DATA_DIR = '../data/raw'
LABEL = 'neutral'  # ← Bunu ihtiyacına göre değiştir
SAMPLES_PER_LABEL = 200

# === MEDIAPIPE ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# === DİZİN OLUŞTURMA ===
save_path = os.path.join(DATA_DIR, LABEL)
os.makedirs(save_path, exist_ok=True)

# === KAMERA AÇ ===
cap = cv2.VideoCapture(0)
collected = 0

print(f"[INFO] '{LABEL}' için veri toplanıyor. Çıkmak için 'q' tuşuna bas.")

while cap.isOpened() and collected < SAMPLES_PER_LABEL:
    ret, frame = cap.read()
    if not ret:
        continue

    # Görüntüyü çevir ve RGB yap
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Yüz mesh algılama
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # 468 noktanın x, y, z koordinatlarını al
            landmarks = np.array([[pt.x, pt.y, pt.z] for pt in face_landmarks.landmark]).flatten()

            # Dosya adı ve kaydetme
            filename = f'{uuid.uuid4().hex}.npy'
            np.save(os.path.join(save_path, filename), landmarks)

            collected += 1
            print(f"[{collected}/{SAMPLES_PER_LABEL}] Kaydedildi: {filename}")

            # Görsel çizim
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    # Pencereyi göster
    cv2.imshow("Veri Toplama", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n[✓] Veri toplama tamamlandı. {collected} örnek '{LABEL}' için kaydedildi.")

cap.release()
cv2.destroyAllWindows()
