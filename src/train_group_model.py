import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Dataset
class ExpressionDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        self.X = df.drop("group_label", axis=1).values.astype(np.float32)
        self.y = LabelEncoder().fit_transform(df["group_label"])
        self.encoder = LabelEncoder().fit(df["group_label"])  # Tahmin için kullanılacak

        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Eğitim fonksiyonu
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Ana fonksiyon
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("💻 PyTorch cihaz:", device)

    df = pd.read_csv("../data/normalized_final_dataset.csv")
    X = df.drop("group_label", axis=1).values
    y = LabelEncoder().fit_transform(df["group_label"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = ExpressionDataset("../data/normalized_final_dataset.csv")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    model = MLPClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        loss = train(model, train_loader, criterion, optimizer, device)
        print(f"[{epoch+1}/20] Loss: {loss:.4f}")

    torch.save(model.state_dict(), "../models/group_expression_model.pth")
    print("✅ Model kaydedildi: group_expression_model.pth")

    # Test
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)

        y_true = y_test_tensor.cpu().numpy()
        y_pred = predictions.cpu().numpy()

        print("\n📊 Sınıflandırma Raporu:\n")
        print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))
        print("🎯 Accuracy:", accuracy_score(y_true, y_pred) * 100, "%")

if __name__ == "__main__":
    main()
