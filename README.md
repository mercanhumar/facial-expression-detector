# Facial Expression Recognition & Emotion-Adaptive Game

Bu proje, gerçek zamanlı yüz ifadesi tanıma (facial expression recognition) sistemi geliştirerek oyuncunun mimiklerine göre oyun atmosferini dinamik olarak değiştiren bir Unity tabanlı psikolojik gerilim oyunudur.

## 🔍 Proje Özeti

- **Veri seti**: FER2013 (yerel olarak tutulur, GitHub'a yüklenmez)
- **Model**: PyTorch ile eğitilmiş CNN (5-fold cross validation)
- **Gerçek zamanlı tahmin**: Webcam üzerinden canlı duygu tanıma
- **Oyun entegrasyonu**: Unity oyun motoruna TCP ile canlı veri aktarımı
- **Atmosfer değişimi**: Oyuncunun duygusuna göre ışık, renk, ses değişimi

## 🚀 Özellikler

- Duygu sınıfları: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`
- Ensemble CNN modeli ile daha dengeli ve kararlı tahminler
- Python tarafında `predict_cnn.py` ile canlı duygu tahmini
- Unity tarafında Shader Graph ile görsel atmosfer değişimi

## 🧠 Kullanılan Teknolojiler

- PyTorch, OpenCV, NumPy, torchvision
- Git & GitHub versiyon kontrolü
- Unity (URP, Shader Graph)
- Git LFS kullanılmaz (veri dosyaları hariç tutulmuştur)

## 🗂️ Klasör Yapısı

