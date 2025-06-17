#Sobel filtresi, görüntüdeki kenarları (gradyanları) yatay ve dikey doğrultuda bulmak için kullanılır. Genellikle iki adet filtre kullanılır
#Canny, görüntüyü önce bulanıklaştırıp gradyanları hesaplayarak, maksimum olmayan değerleri bastırıp çift eşikleme ve kenar takibi ile net ve doğru kenarları belirleyen çok aşamalı bir kenar algılama algoritmasıdır.
#Laplacian filtresi, bir ikinci türev operatörüdür. Görüntüdeki kenarları tespit ederken, hem yatay hem dikey değişimleri aynı anda algılar. Yani tek bir filtre ile çalışır. 
# büyük boyutlu bir kernel daha fazla komşu pikseli hesaba kattığı için daha güçlü bulanıklaştırma veya daha geniş kenar algılama sağlar, ancak işlem daha yavaş olur.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Resim urlsi (açık tonlu köpek)
url = "https://images.unsplash.com/photo-1558788353-f76d92427f16?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80"
resp = urllib.request.urlopen(url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
img = cv2.imdecode(image, cv2.IMREAD_COLOR)

if img is None:
    raise Exception("Resim yüklenemedi")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Gaussian Blur ile hafif bulanıklaştır
blur = cv2.GaussianBlur(gray, (5,5), 0)

# 2. Canny kenar tespiti (eşikleri düşürdük)
canny = cv2.Canny(blur, 30, 100)

# 3. Sobel kenar tespiti
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

# Sobel normalize edip eşikleme (daha belirgin kenar için)
sobel_norm = np.uint8(255 * sobel / np.max(sobel))
_, sobel_thresh = cv2.threshold(sobel_norm, 50, 255, cv2.THRESH_BINARY)

# 4. Laplacian kenar tespiti
laplacian = cv2.Laplacian(blur, cv2.CV_64F)
laplacian_norm = np.uint8(255 * np.abs(laplacian) / np.max(np.abs(laplacian)))
_, laplacian_thresh = cv2.threshold(laplacian_norm, 30, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(15,9))

plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Orijinal Renkli")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(gray, cmap='gray')
plt.title("Gri")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(canny, cmap='gray')
plt.title("Canny ")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(sobel_thresh, cmap='gray')
plt.title("Sobel")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(laplacian_thresh, cmap='gray')
plt.title("Laplacian")
plt.axis('off')

plt.tight_layout()
plt.show()
