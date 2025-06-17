#PCA  verideki en fazla bilgi içeren yönleri bulup boyut indirgeme yapan bir yöntemdir.
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Yapay renk bloklarından oluşan görüntü oluştur
yukseklik, genislik = 300, 450
gorsel = np.zeros((yukseklik, genislik, 3), dtype=np.uint8)

# OpenCV BGR formatında renkleri ayarla
gorsel[:, 0:150] = [0, 0, 255]    # Kırmızı
gorsel[:, 150:300] = [0, 255, 0]   # Yeşil 
gorsel[:, 300:450] = [255, 0, 0]  # Mavi   #Görselinizde tamamen saf renkler kullanıldığı için ayrışımı kolay

# BGR'den RGB'ye dönüştür
gorsel_rgb = cv2.cvtColor(gorsel, cv2.COLOR_BGR2RGB)

# Görüntüyü 2D forma getir
h, w, c = gorsel_rgb.shape
gorsel_2d = gorsel_rgb.reshape(-1, 3)

# PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
gorsel_pca = pca.fit_transform(gorsel_2d)

# Orijinal uzaya geri dönüştür
gorsel_geri = pca.inverse_transform(gorsel_pca)

# 0-255 aralığına getir ve uint8 yap
gorsel_geri = np.clip(gorsel_geri, 0, 255).astype(np.uint8)
gorsel_geri = gorsel_geri.reshape(h, w, c)

# Görüntüleri göster
plt.figure(figsize=(15,5))

# 1. Orijinal görüntü
plt.subplot(1,3,1)
plt.imshow(gorsel_rgb)
plt.title("Orijinal Görüntü")
plt.axis('off') 

# 2. PCA dağılım grafiği
plt.subplot(1,3,2)
plt.scatter(gorsel_pca[:, 0], gorsel_pca[:, 1], s=2, c=gorsel_2d/255.0)
plt.title("2 Bileşenli PCA (Dağılım)")
plt.xlabel("Birinci Temel Bileşen")
plt.ylabel("İkinci Temel Bileşen")

# 3. Geri yüklenen görüntü
plt.subplot(1,3,3)
plt.imshow(gorsel_geri)
plt.title("Geri Yüklenen Görüntü")
plt.axis('off') 

plt.tight_layout()
plt.show()

# Varyans oranlarını yazdır
print("Açıklanan varyans oranları:", pca.explained_variance_ratio_)#birinci ve ikinci bileşenin taşıdığı bilgi miktarı 
print("Toplam açıklanan varyans:", sum(pca.explained_variance_ratio_))
