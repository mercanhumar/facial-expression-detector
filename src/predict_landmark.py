import cv2
import mediapipe as mp
import torch
import numpy as np
from model import ExpressionClassifier
from collections import deque
import statistics

# Cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 Kullanılan cihaz:", device)

# Model ve etiketler
model = ExpressionClassifier(input_dim=1404, num_classes=7).to(device)
model.load_state_dict(torch.load("../models/expression_model.pth", map_location=device))
model.eval()

labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Mediapipe setup (468 landmark, default)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   refine_landmarks=False, min_detection_confidence=0.5)

# Kamera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Tahmin geçmişi (sabitleme)
history = deque(maxlen=5)
display_label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks, dtype=np.float32)

            if landmarks.shape[0] != 1404:
                continue  # Skip if wrong shape

            input_tensor = torch.tensor(landmarks, dtype=torch.float32)
            input_tensor = (input_tensor - mean) / std  # Veya MinMaxScale gibi
            input_tensor = input_tensor.to(device).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()

                print(f"🔍 Olasılıklar: {probs.cpu().numpy().round(2)}")

            # Tahmin geçmişine ekle
            history.append(pred)
            if len(history) == 5:
                majority = statistics.mode(history)
                display_label = labels[majority]

            cv2.putText(frame, display_label, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Landmark Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
