import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from collections import Counter
import subprocess

DATASET_PATH = r"C:\Users\seren\facial_expression\data\ferplus"
SAVE_PATH = r"C:\Users\seren\facial_expression\models\efficientnet_b0_ferplus.pth"
CLASSES_PATH = r"C:\Users\seren\facial_expression\models\classes.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("💻 Cihaz:", device)


# ---------------- GPU INFO ---------------
def print_gpu_info():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            encoding="UTF-8"
        )
        util, mem = result.strip().split(", ")
        print(f"⚡ GPU Kullanım: %{util} | VRAM: {mem} MB")
    except:
        print("⚠️ GPU bilgisi alınamadı (nvidia-smi yok).")
# ------------------------------------------


def main():
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATASET_PATH, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATASET_PATH, "val"), transform=val_tf)

    classes = train_ds.classes
    print("📁 Sınıflar:", classes)

    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=4)

    class_counts = Counter([label for _, label in train_ds.samples])
    print("📊 Sınıf dağılımı:", class_counts)

    weights = torch.tensor(
        [1.0 / class_counts[i] for i in range(len(classes))],
        dtype=torch.float32
    ).to(device)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))

    if os.path.exists(SAVE_PATH):
        print("🔁 Mevcut model bulundu, eğitime devam ediliyor...")
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    else:
        print("🆕 Yeni eğitim başlatılıyor (checkpoint bulunamadı).")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0
    torch.cuda.empty_cache()

    for epoch in range(1, 25):
        print(f"\n🚀 EPOCH {epoch} başlıyor...\n")
        model.train()
        total, correct = 0, 0

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)

            

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        val_total, val_correct = 0, 0

        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"📌 Epoch {epoch}/25 — Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            torch.save(model.state_dict(), SAVE_PATH)
            best_acc = val_acc
            print("💾 Yeni en iyi model kaydedildi!")


if __name__ == "__main__":
    main()
