"""
Erosion: Küçük beyaz noktalar ve kare biraz küçülür (beyazlar erir).

Dilation: Beyazlar büyür, gürültü artabilir.

Opening: Gürültü beyaz noktalar temizlenir (küçük beyazlar kaybolur).

Closing: Kare içindeki küçük siyah boşluklar kapanır.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Beyaz gürültülü binary image oluştur
img = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(img, (20, 20), (80, 80), 255, -1)  # Büyük beyaz kare
noise = (np.random.rand(100, 100) > 0.95).astype(np.uint8) * 255  # %5 beyaz gürültü
img_noisy = cv2.bitwise_or(img, noise)

# Kernel (yapısal eleman)
kernel = np.ones((5,5), np.uint8)

# Erosion
erosion = cv2.erode(img_noisy, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(img_noisy, kernel, iterations=1)

# Opening (Erosion sonra Dilation)
opening = cv2.morphologyEx(img_noisy, cv2.MORPH_OPEN, kernel)

# Closing (Dilation sonra Erosion)
closing = cv2.morphologyEx(img_noisy, cv2.MORPH_CLOSE, kernel)

# Görselleştir
plt.figure(figsize=(12, 8))

titles = ['Orijinal Gürültülü Görüntü', 'Erosion (Aşındırma)', 'Dilation (Genişletme)', 'Opening (Açma)', 'Closing (Kapama)']
images = [img_noisy, erosion, dilation, opening, closing]

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
