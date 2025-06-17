"""
K arttıkça segmentasyon detaylanır.

K=2 olursa renkler daha büyük kümelerde birleşir.

K=5 olursa daha fazla renk tonu ayrılır, segmentasyon daha ince olur.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Yapay renkli bloklu görüntü oluşturalım (daha kolay görselleme için)
height, width = 200, 300
image = np.zeros((height, width, 3), dtype=np.uint8)
image[:, :100] = [255, 0, 0]    # Kırmızı
image[:, 100:200] = [0, 255, 0] # Yeşil
image[:, 200:] = [0, 0, 255]    # Mavi

# BGR'den RGB'ye çevir (matplotlib için)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görüntüyü reshape edip 2D matris yap
pixels = image_rgb.reshape((-1, 3))
pixels = np.float32(pixels)

# K değerleri, farklı cluster sayıları için deneyelim
Ks = [7, 5, 2]

plt.figure(figsize=(15, 5))

for i, K in enumerate(Ks):
    # K-Means parametreleri
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)  # Merkezleri uint8 yap
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image_rgb.shape))

    plt.subplot(1, len(Ks), i+1)
    plt.imshow(segmented_image)
    plt.title(f'K-Means Segmentasyon\nK={K}')
    plt.axis('off')

plt.tight_layout()
plt.show()
