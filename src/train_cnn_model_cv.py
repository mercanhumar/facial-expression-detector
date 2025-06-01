# facial_expression/src/train_cnn_model_cv.py

import warnings
warnings.filterwarnings("ignore")    # → Tüm warning’leri kapatır

import os
import random
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

# ----------------- Sabitler -----------------
CSV_PATH  = "../data/fer2013.csv"
N_EPOCHS  = 25
BATCH     = 32
FOLDS     = 5
SEED      = 42
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Normalize ve Augmentasyon -----------------
COMMON_NORMALIZE = transforms.Normalize(mean=(0.5,), std=(0.5,))

_train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(48, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    COMMON_NORMALIZE
])
_test_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    COMMON_NORMALIZE
])

# ----------------- Dataset Sınıfı -----------------
class FERCsv(Dataset):
    def __init__(self, df, tf):
        # df["pixels"] içindeki her satırı split edip (48×48) uint8 matrisine çeviriyoruz
        self.x = np.stack(
            df["pixels"].str.split().apply(
                lambda s: np.asarray(s, dtype=np.uint8).reshape(48, 48)
            )
        )
        self.y = df["emotion"].to_numpy(dtype=np.int64)
        self.tf = tf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i]             # uint8 array (48×48)
        return self.tf(img), self.y[i]

# ----------------- CNN Model -----------------
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Bloc 1: 48×48 → 24×24
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloc 2: 24×24 → 12×12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloc 3: 12×12 → 6×6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloc 4: 6×6 → 3×3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Global Pooling + Fully-Connected
            nn.AdaptiveAvgPool2d(1),   # 3×3 → 1×1
            nn.Flatten(),

            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.net(x)

# ----------------- K-Fold Eğitim Fonksiyonu -----------------
def k_fold_training():
    df = pd.read_csv(CSV_PATH)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    fold_acc = []

    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(df["pixels"], df["emotion"]), start=1
    ):
        print(f"\n📂 Fold {fold}/{FOLDS}")

        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        vl_df = df.iloc[val_idx].reset_index(drop=True)

        tr_ds = FERCsv(tr_df, _train_tf)
        vl_ds = FERCsv(vl_df, _test_tf)

        # Sınıf dengesi için WeightedRandomSampler
        cls_counts = np.bincount(tr_df["emotion"].to_numpy(), minlength=7)
        class_weights = cls_counts.max() / cls_counts
        sample_weights = class_weights[tr_df["emotion"].to_numpy()]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True
        )
        tr_dl = DataLoader(
            tr_ds, batch_size=BATCH,
            sampler=sampler, drop_last=True
        )
        vl_dl = DataLoader(vl_ds, batch_size=BATCH)

        loss_fn = nn.CrossEntropyLoss()  # Sampler ile dengelediğimiz için weight=None

        model = EmotionCNN().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=2e-4)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.5)

        best_acc, best_state = 0.0, None
        patience, no_improve = 4, 0

        for ep in range(1, N_EPOCHS + 1):
            # ------- Training -------
            model.train()
            running_loss = 0.0
            for xb, yb in tr_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                running_loss += loss.item() * xb.size(0)
            scheduler.step()

            # ------- Validation -------
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for xb, yb in vl_dl:
                    xb = xb.to(DEVICE)
                    out = model(xb)
                    preds.extend(out.argmax(dim=1).cpu().numpy())
                    labels.extend(yb.numpy())
            acc = accuracy_score(labels, preds)

            print(
                f"[{ep:02}/{N_EPOCHS}] "
                f"Train Loss={(running_loss/len(tr_ds)):.4f}  "
                f"Val Acc={acc*100:.2f}%"
            )

            # Early stopping
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏹️ Early stopping at epoch {ep}")
                    break

        # Fold’un en iyi ağırlığını kaydet ve tekrar yükle
        save_path = f"../models/cnn_fold{fold}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state, save_path)
        model.load_state_dict(torch.load(save_path))
        print(f"✅ Fold {fold} en iyi doğruluk: {best_acc*100:.2f}%")
        fold_acc.append(best_acc)

    print(f"\n🔚 Ortalama doğruluk: {np.mean(fold_acc)*100:.2f}%")

# ----------------- Tek-Split / Test Fonksiyonu (İsteğe Bağlı) -----------------
# Eğer tek-split ile “best_cnn_augmented.pth” dosyasını yükleyip test etmek istersen,
# aşağıdaki fonksiyonu aktif hale getirebilirsiniz. Aksi halde bu kısmı boş bırakın
# ve yalnızca k_fold_training() fonksiyonunu kullanın.

def single_split_and_test():
    df = pd.read_csv(CSV_PATH)
    # Eğer “PrivateTest” satırlarını ayıklıyorsanız:
    if 'Usage' in df.columns:
        df = df[df['Usage'] != 'PrivateTest'].reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    # %80-%10-%10 split (örnek)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['emotion']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['emotion']
    )

    train_ds = FERCsv(train_df, _train_tf)
    val_ds   = FERCsv(val_df,   _test_tf)
    test_ds  = FERCsv(test_df,  _test_tf)

    from torch.utils.data import DataLoader
    cls_counts = train_df['emotion'].value_counts().sort_index().to_numpy()
    class_weights = cls_counts.max() / cls_counts
    sample_weights = class_weights[train_df['emotion'].to_numpy()]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dl = DataLoader(train_ds, batch_size=64, sampler=sampler)
    val_dl   = DataLoader(val_ds,   batch_size=64)
    test_dl  = DataLoader(test_ds,  batch_size=64)

    model = EmotionCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_val_acc = 0.0
    patience, no_inc = 4, 0

    for epoch in range(1, 21):
        # ------- Training -------
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        scheduler.step()

        # ------- Validation -------
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch = X_batch.to(DEVICE)
                out = model(X_batch)
                preds = out.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y_batch.numpy())
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"[{epoch:02}/20] Train Loss: {(train_loss/len(train_dl.dataset)):.4f}  Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "../models/best_cnn_augmented.pth")
            no_inc = 0
        else:
            no_inc += 1
            if no_inc >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    # ------- Test Aşaması -------
    model.load_state_dict(torch.load("../models/best_cnn_augmented.pth"))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            X_batch = X_batch.to(DEVICE)
            out = model(X_batch)
            preds = out.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(y_batch.numpy())

    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    print("\n📊 Test Raporu:")
    print(classification_report(test_labels, test_preds))
    print("🎯 Test Accuracy:", accuracy_score(test_labels, test_preds)*100, "%")

    # Confusion Matrix Görselleştirme (İsteğe Bağlı)
    cm = confusion_matrix(test_labels, test_preds, labels=[0,1,2,3,4,5,6])
    labels_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
    plt.title("Test Set Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ----------------- Ana Çalıştırma Bloğu -----------------
if __name__ == "__main__":
    # K-fold eğitimi için:
    k_fold_training()

    # Eğer Tek-Split + Test kodunu çalıştırmak isterseniz, yukarıdaki fonksiyonu çağırın:
    # single_split_and_test()
