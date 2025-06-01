import pandas as pd
import numpy as np

# Girdi / Çıktı dosyaları
INPUT_CSV = "../data/final_dataset_from_fer.csv"
OUTPUT_CSV = "../data/normalized_final_dataset.csv"

# FER label → emotion → grup
group_map = {
    0: "negative",  # angry
    1: "negative",  # disgust
    2: "negative",  # fear
    3: "positive",  # happy
    4: "negative",  # sad
    5: "positive",  # surprise
    6: "neutral"    # neutral
}

# Veri oku
df = pd.read_csv(INPUT_CSV)

# Son sütun label
labels = df["label"]
X = df.drop("label", axis=1).values

# Normalize: (x - min) / (max - min)
X_min = X.min(axis=1).reshape(-1, 1)
X_max = X.max(axis=1).reshape(-1, 1)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

# Yeni DataFrame
df_norm = pd.DataFrame(X_norm, columns=[f"x{i}" for i in range(X.shape[1])])
df_norm["group_label"] = labels.map(group_map)

# Kaydet
df_norm.to_csv(OUTPUT_CSV, index=False)
print(f"[✓] Normalize edilmiş veri kaydedildi: {OUTPUT_CSV}")
