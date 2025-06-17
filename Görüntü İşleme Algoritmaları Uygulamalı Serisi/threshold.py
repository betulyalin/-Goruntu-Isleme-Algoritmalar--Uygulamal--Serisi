"""
Global Threshold: Sabit eşik değeri (127) ile pikseller beyaz ya da siyah yapılır. Basit ama ışık değişimlerine duyarsız.

Otsu: Görüntünün histogramını analiz edip en uygun eşik değerini otomatik bulur.

Adaptive: Görüntünün küçük bölgelerine göre ayrı ayrı eşik belirler, özellikle aydınlatmanın düzensiz olduğu yerlerde iyi çalışır.
"""
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resim indir ve oku
url = 'https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60'
resp = urllib.request.urlopen(url)
img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Griye çevir
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Global Thresholding (eşik değeri 127)
_, global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Otsu Thresholding (otomatik eşik değeri bulur)
_, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive Thresholding (bölgesel eşik)
adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, blockSize=11, C=2)

# Görselleştir
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Orijinal Görüntü')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(global_thresh, cmap='gray')
plt.title('Global Thresholding (T=127)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(otsu_thresh, cmap='gray')
plt.title("Otsu'nun Thresholding'i")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title('Adaptive Thresholding (Gaussian)')
plt.axis('off')

plt.tight_layout()
plt.show()
