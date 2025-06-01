import os
import numpy as np
import pandas as pd

DATA_DIR = "../data/landmarks_from_fer"
OUTPUT_CSV = "../data/final_dataset_from_fer.csv"

data = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".npy"):
        label = int(file.split("_")[0])  # Örn: 3_172.npy → label = 3
        path = os.path.join(DATA_DIR, file)

        try:
            landmarks = np.load(path)
            if landmarks.shape[0] == 468 * 3:  # x,y,z koordinatları kontrol
                row = list(landmarks) + [label]
                data.append(row)
        except Exception as e:
            print(f"[!] Hata: {file} → {e}")

# DataFrame oluştur
if data:
    df = pd.DataFrame(data)
    df.columns = [f"x{i}" for i in range(len(data[0]) - 1)] + ["label"]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[✓] Dataset oluşturuldu: {OUTPUT_CSV} (Toplam {len(df)} örnek)")
else:
    print("[!] Uyarı: Hiç veri bulunamadı.")
