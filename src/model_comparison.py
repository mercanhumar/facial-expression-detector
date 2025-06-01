import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb



# Veri yükle
df = pd.read_csv("../data/normalized_final_dataset.csv")
X = df.drop("group_label", axis=1).values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df["group_label"].values)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))



# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendir (MLP, SVM, KNN için önemli)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeller
models = {
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVC": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
}

# Sonuçları burada tutacağız
results = []

print("🔍 Modeller eğitiliyor ve değerlendiriliyor...\n")

for name, model in models.items():
    # Sadece MLP, SVC, KNN ölçekli veriye ihtiyaç duyar
    if name in ["MLP", "SVC", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    results.append((name, acc, f1))

# Sonuçları yazdır
print("{:<15} {:<12} {:<12}".format("Model", "Accuracy", "F1-Score"))
print("-" * 40)
for name, acc, f1 in results:
    print(f"{name:<15} {acc*100:>6.2f}%      {f1*100:>6.2f}%")

# En iyi modeli seç
best_model = max(results, key=lambda x: x[1])  # accuracy'e göre
print("\n🏆 En iyi model (accuracy):", best_model[0])
