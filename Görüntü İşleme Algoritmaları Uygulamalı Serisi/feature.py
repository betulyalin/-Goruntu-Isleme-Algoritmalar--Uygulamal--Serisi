"""
1. SIFT (Scale-Invariant Feature Transform)
Görüntüdeki dikkat çekici noktaları (köşe, kenar gibi) ve bu noktaların etrafındaki benzersiz "desenleri" çıkarır.
Bu özellikler, farklı ölçek ve açıdan çekilen resimler arasında bile benzer nesneleri tanımaya yarar.
Ağır ama güçlüdür.

2. ORB (Oriented FAST and Rotated BRIEF)
SIFT’e göre daha hızlı ve hafif, yine önemli noktalar ve bu noktaların küçük özetlerini çıkarır.
Gerçek zamanlı uygulamalar için çok uygundur.
Daha hızlı, biraz daha az detaylı.

3. HOG (Histogram of Oriented Gradients)
Görüntünün kenar ve yön bilgilerini alır.
Kenarların hangi açıda, ne yoğunlukta olduğu bilgilerini histogram olarak tutar.
Bu öznitelikler insan ya da nesne tanımada çok kullanılır.
"Kenar yönlerinin sayısal özetidir."
"""
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

# Resim URL'si
url = 'https://images.unsplash.com/photo-1517423440428-a5a00ad493e8?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60'
resp = urllib.request.urlopen(url)
image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

if img is None:
    raise Exception("Resim indirilemedi veya açılamadı.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SIFT
sift = cv2.SIFT_create()
keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)
img_sift = cv2.drawKeypoints(img, keypoints_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ORB
orb = cv2.ORB_create()
keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)
img_orb = cv2.drawKeypoints(img, keypoints_orb, None, color=(0,255,0), flags=0)

# HOG
hog_features, hog_image = hog(
    gray, 
    orientations=9, 
    pixels_per_cell=(8,8),
    cells_per_block=(2,2), 
    block_norm='L2-Hys', 
    visualize=True,
    feature_vector=True
)

# HOG görüntüsünü normalize et ve parlaklığı artır
hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())  # Normalize [0,1]
hog_image = np.power(hog_image, 0.4)  # Gamma düzeltme (0.4 ile parlaklık artırılır)
hog_image = (hog_image * 255).astype(np.uint8)

plt.figure(figsize=(15, 8))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoint'leri")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
plt.title("ORB Keypoint'leri")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(hog_image, cmap='gray')
plt.title("HOG Görselleştirme")
plt.axis('off')

plt.tight_layout()
plt.show()
