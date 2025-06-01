import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from torchvision import transforms


# Cihaz kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 Kullanılan cihaz:", device)


# Augmentation tanımları
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Dataset
class FERDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.X = np.array([np.fromstring(pix, sep=' ') for pix in dataframe['pixels']])
        self.X = self.X.reshape(-1, 1, 48, 48).astype(np.uint8)  # PIL için uint8 lazım
        self.y = dataframe['emotion'].values.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx][0]  # [1, 48, 48] → [48, 48]
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)
        return img, torch.tensor(self.y[idx])

# CNN Modeli
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*6*6, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        return self.net(x)
def train():
    # Veriyi yükle ve böl
    df = pd.read_csv("../data/fer2013.csv")
    df = df[df['Usage'] != 'PrivateTest']
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])

    train_dataset = FERDataset(train_df, transform=train_transform)
    test_dataset = FERDataset(test_df, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Model, loss, optimizer
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Eğitim
    epochs = 20
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # Modeli kaydet
    torch.save(model.state_dict(), "../models/cnn_fer_model_augmented.pth")
    print("✅ Model kaydedildi: cnn_fer_model_augmented.pth")

    # Test işlemi
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    # Sonuç
    print("\n📊 Sınıflandırma Raporu:")
    print(classification_report(all_labels, all_preds))
    print("🎯 Accuracy:", accuracy_score(all_labels, all_preds) * 100, "%")

    # Loss grafiği
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss with Augmentation")
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    train()