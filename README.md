# Digital-Image-Processing - Midterm-Homework
ISIC cilt lezyonu görüntülerine hem RGB (renkli) hem de gri tonlamalı görüntü işleme teknikleri uyguladım

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XDtg6OJkR6YLbwBhGW5SBiFx5eMifROF?usp=sharing)

Dizüstü bilgisayarı Google Colab'da açmak için yukarıdaki simgeye tıklayın.

# 1. Veri Yükleme 

**Proje Genel Bakış**

Bu proje, 9 farklı cilt lezyonu sınıfı içeren bir cilt kanseri veri setinin analizini içermektedir. Veri seti Kaggle'dan kaynaklanmakta olup, analiz ve işleme için 2.357 tıbbi görüntü içermektedir.

**Veri Seti Bilgileri:**

- Kaynak: Kaggle "skin-cancer9-classesisic" veri seti

- Toplam Görüntü: 2.357

- Sınıflar: 9 farklı cilt lezyonu türü


| Sınıf Dağılımı | Sınıf Adı	Adet |
|-------------|------------|
| pigmented benign keratosis | 478 |
| melanoma | 454 |
| basal cell carcinoma | 392 |
| nevus | 373 |
| squamous cell carcinoma | 197 |
| vascular lesion | 142 |
| actinic keratosis | 130 |
| dermatofibroma | 111 |
| seborrheic keratosis | 80 |

# 1.1. Kütüphanelerin İçe Aktarılması 

**Kullanılan Kütüphaneler**

pandas, numpy - Veri manipülasyonu

matplotlib, seaborn - Görselleştirme

cv2, PIL - Görüntü işleme

pathlib, os - Dosya sistemi operasyonları

kagglehub - Veri seti indirme

# 1.2. Veri Setinin Yüklenmesi

**Veri seti başarıyla indirildi ve aşağıdaki yapıya sahip bir pandas DataFrame'e yüklendi:**

file_path: Görüntü dosyasının tam yolu

file_name: Görüntü dosyasının adı

class: Cilt lezyonunun sınıflandırması

file_size_kb: Kilobyte cinsinden dosya boyutu

# 1.3. Veri Özelliklerinin İncelenmesi

**Çözünürlük Analizi Sonuçları**

Yükseklik: Min: 450px, Maks: 4.479px, Ortalama: 959.88px

Genişlik: Min: 600px, Maks: 6.088px, Ortalama: 1.346.52px

En Yaygın Çözünürlük: 450 × 600 piksel

Renk Formatı: Tüm görüntüler RGB formatında

Dosya Boyutu Analizi

Minimum: 24.727 KB

Maksimum: 20.920.374 KB

Ortalama: 350.436,44 KB

Medyan: 260.719 KB

# 2. Görüntü Yükleme ve Görselleştirme

# 2.1. Rastgele Görüntüler Seçme
   
**Analiz için 9 rastgele görüntü seçildi:**

**Seçilen Görüntüler: 400, 403, 399, 800, 1002, 2021, 20, 80, 693 indeksleri**

**Sınıf Dağılımı:**

Pigmented benign keratosis: 3 görüntü

Melanoma: 4 görüntü

Nevus: 1 görüntü

Basal cell carcinoma: 1 görüntü

<img width="753" height="1995" alt="загрузка (1)" src="https://github.com/user-attachments/assets/0e13b40a-2202-4e4e-a13a-ef806ea55562" />

# 2.2. Rastgele Görüntülerin İstatistiksel Özellikleri

**Her görüntü için RGB ve gri tonlamalı istatistikleri hesaplandı:**

**İstatistikler:**

Image 1: RGB İstatistiksel:Overall - Min: 0, Max: 255, Mean: 136.74, Std: 35.31

Image 1: Grayscale İstatistiksel:Overall - Min: 1, Max: 250,Mean: 133.59835555555554, 29.722805066135773

Image 2: RGB İstatistiksel:Overall - Min: 18, Max: 253, Mean: 212.60, Std: 37.95

Image 2: Grayscale İstatistiksel:Overall - Min: 21, Max: 240,Mean: 211.18897407407408, 36.5750795030056

Image 3: RGB İstatistiksel:Overall - Min: 51, Max: 211, Mean: 146.20, Std: 33.90

Image 3: Grayscale İstatistiksel:Overall - Min: 82, Max: 182,Mean: 140.99535185185186, 17.21370074125647

Image 4: RGB İstatistiksel:Overall - Min: 0, Max: 255, Mean: 76.48, Std: 42.84

Image 4: Grayscale İstatistiksel:Overall - Min: 13, Max: 254,Mean: 77.17396421845574, 32.505309263275706

Image 5: RGB İstatistiksel:Overall - Min: 26, Max: 255, Mean: 200.15, Std: 33.80

Image 5: Grayscale İstatistiksel:Overall - Min: 71, Max: 255,Mean: 202.76696650187174, 24.06549688041099

Image 6: RGB İstatistiksel:Overall - Min: 0, Max: 255, Mean: 159.72, Std: 37.97

Image 6: Grayscale İstatistiksel:Overall - Min: 0, Max: 255,Mean: 159.46745790381246, 34.05493528401974

Image 7: RGB İstatistiksel:Overall - Min: 0, Max: 255, Mean: 162.08, Std: 45.65

Image 7: Grayscale İstatistiksel:Overall - Min: 0, Max: 255,Mean: 152.78467399600476, 26.693659150726997

Image 8: RGB İstatistiksel:Overall - Min: 81, Max: 248, Mean: 190.73, Std: 28.16

Image 8: Grayscale İstatistiksel:Overall - Min: 103, Max: 220,Mean: 187.20162592592592, 11.115700722021536

Image 9: RGB İstatistiksel:Overall - Min: 10, Max: 247, Mean: 125.50, Std: 24.46

Image 9: Grayscale İstatistiksel:Overall - Min: 13, Max: 233,Mean: 128.43947219848633, 17.16719906729315

# 2.3. Histogram Çizimi (RGB + Grayscale)

**Tüm görüntüler için RGB ve gri tonlamalı histogramlar oluşturuldu:**

**Histogram Analiz Özeti:**

Image 1: Kırmızı orta-yüksek tonlarda (~150-200) daha baskın

Image 2: Çok parlak görüntü; histogram neredeyse tamamen sağ tarafta

Image 3: Kırmızı net şekilde baskın (~180-230)

Image 4: Görüntü koyu/orta ton ağırlıklı; özellikle mavi baskın

Image 5: Aydınlık alanlar fazla, RGB kanallar dengeli ama kırmızı önde

Image 6: Aydınlık ağırlıklı; gri histogram sağ bölgede yoğun

Image 7: Görüntü çok parlak; tüm kanallar yüksek parlaklığa yayılmış

Image 8: Aydınlık ama dengeli bir sahne; kırmızı kanal öne çıkıyor

Image 9: Orta ton ağırlıklı; yeşil kanal baskın

<img width="1388" height="2492" alt="загрузка (38)" src="https://github.com/user-attachments/assets/bd2790d7-4063-48d1-9482-51ec1f03679c" />

# 3. Görüntü İşleme ve İyileştirme

**Bu raporda sunulan analiz,  "image_400" görseli üzerinden gerçekleştirilmiştir.**

Bu bölümde, görüntü iyileştirme için üç yöntem ele alınmıştır: kontrast germe, histogram eşitleme ve gamma düzeltme. Bu yöntemler, cilt lezyonu görüntülerindeki detayların görünürlüğünü artırmak için hem RGB hem de gri tonlamalı görüntülere uygulanmıştır.

# 3.1. Kontrast Germe (Stretching) İşlemi

Kontrast germe, piksel değerlerinin aralığını tüm mevcut [0, 255] aralığını kullanacak şekilde genişleten doğrusal bir dönüşümdür.

**Uygulama**

```python
def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val - min_val > 0:
        stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        stretched = image
    
    return stretched
```

**400 Numaralı Görüntü Analizi**

**Görsel Karşılaştırma**

— Original RGB: Görüntü düşük kontrasta sahip, lezyon detayları zor seçiliyor

— Stretched RGB: Kontrast belirgin şekilde iyileşmiş, lezyon sınırları daha net görünüyor

— Original Grayscale: RGB versiyonu gibi düşük kontrastlı

— Stretched Grayscale: Lezyon dokuları ve sınırlarının görünürlüğü artmış

**Histogram Analizi**

— Original RGB Histogram: Piksel değerleri dar bir aralıkta yoğunlaşmış

— Stretched RGB Histogram: Aralık tüm [0, 255]'e genişletilmiş, dağılım daha düzgün hale gelmiş

— Original Grayscale Histogram: Aralığın orta kısmında yoğunluk

— Stretched Grayscale Histogram: Dağılım genişlemiş, kontrast iyileşmesi doğrulanmış

<img width="1189" height="970" alt="загрузка (3)" src="https://github.com/user-attachments/assets/e629e717-4e81-46d3-996a-78f88b1871e4" />

<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/3837d625-5727-4aad-b13f-fe72aef9ac61" />


**Kontrast Germe Sonuçları**

Kırmızı, Yeşil ve Mavi kanallarının histogramları, parlaklık aralığının (yaklaşık 50-200) orta bölümünde toplanmıştır. Kontrast germe işleminden sonra bu dağılım, 0 ila 255 arasındaki tam aralığa yaklaşacak şekilde önemli ölçüde genişlemiştir.

Gri tonlamalı görüntü histogramı da 80-200 arasında dar bir dağılım gösterirken, kontrast germe sonrası dağılım 0'dan 255'e kadar çok daha geniş bir aralığa yayılmıştır.

**Oluşturulan diğer görüntüler**

<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/1e19effe-46a1-417f-ad64-f5300cb8bb7f" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/7399a2a5-0451-45a4-83b3-8f5c3734caf3" />

<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/8793a702-92c0-45f3-aeb5-5de08d459d46" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/c8b339b1-1c8c-4f6c-a14f-735d21d4cecb" />

<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/c14b26ec-dfb2-48ea-886a-49545a5cb249" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/43e9811a-a479-4b5b-9084-bd365c4c26a3" />

<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/ff0cdcda-98ca-449f-8ca0-f542d855d420" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/df05e190-d24f-47b8-a773-23415e20492a" />

<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/13ffa2fb-9f5c-4057-bc12-831d7f753172" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/cea40533-99a6-4f5c-abd2-e38232d2ff6b" />

<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/2059c5fa-31f3-441e-8ad5-0aec79a644d9" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/0717ada9-7e33-4d2c-8091-314f4fe57d9b" />

<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/b054a865-9125-4920-9efe-1e00c839c10d" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/cc0b5266-fbad-4630-9f17-9267ea4fb1a3" />

<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/4c71e1d7-d7bd-4daf-be37-4c7c40971f31" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/bb07c288-aace-43c3-8620-bbbfb043e2aa" />


# 3.2. Histogram Eşitleme (Histogram Equalization)

Histogram eşitleme, piksel yoğunluk değerlerini tüm aralıkta düzgün bir dağılım elde edecek şekilde yeniden dağıtır.

**Uygulama**

RGB görüntüler için, renk bilgisini korumak amacıyla YCrCb renk uzayına dönüşüm yapılarak sadece parlaklık kanalı (Y) eşitlenir.

**400 Numaralı Görüntü Analizi**

**Görsel Karşılaştırma**

— Original RGB: Doğal renkler ancak düşük kontrast

— Equalized RGB (via YCrCb): Renkler korunmuş, kontrast iyileşmiş

— Original Grayscale: Standart gri tonlamalı görünüm

— Equalized Grayscale: Detay görünürlüğünde belirgin iyileşme

**Histogram Analizi**

— Original RGB Histogram: Tepe noktalı düzgün olmayan dağılım

— Equalized RGB Histogram: Yoğunluklar daha düzgün dağılmış

— Original Grayscale Histogram: Tıbbi görüntüler için tipik dağılım

— Equalized Grayscale Histogram: Düzgün dağılıma yaklaşmış

**Histogram Eşitleme Sonuçları**

RGB Histogram: Eşitleme (ekvalizasyon) uygulandıktan sonra parlaklık dağılımı çarpıcı biçimde değişti.

Ana özellik: Kırmızı ve Yeşil kanalların histogramları çok düz (üniform) hale geldi ve neredeyse tüm 0 ila 255 aralığını kapladı.

Mavi kanalın kenarlarında (0 ve 255) önemli zirveler (pikler) bulunmakta olup, ortası da daha düz hale gelmiştir. 0 ve 255'teki yüksek zirveler, Mavi kanaldaki birçok pikselin tamamen siyaha veya tamamen beyaza dönüştürüldüğünü gösterir.

Eşitleme sonrasında histogram, 0'dan 255'e kadar olan tüm aralık boyunca belirgin şekilde daha geniş ve daha üniform (düz) hale gelmiştir ki, bu da histogram eşitlemenin klasik bir sonucudur.

<img width="1189" height="970" alt="загрузка (5)" src="https://github.com/user-attachments/assets/324c337f-fae9-4073-a76f-447499def099" />
<img width="1389" height="788" alt="загрузка (6)" src="https://github.com/user-attachments/assets/f862ae47-fa5e-46a4-bd2a-a07ba8960f7f" />



**Oluşturulan diğer görüntüler**

<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/dc55b8d7-2165-4f7c-83e6-427ff25949a0" />
<img width="1390" height="788" alt="image" src="https://github.com/user-attachments/assets/08850317-520e-4312-8859-8bbb6d991820" />
<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/17ff81a3-59fb-488c-ac46-f53a605d7823" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/0532e86a-9060-4c18-99fc-bb8ef2a2087c" />
<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/a7682b51-e55b-48e0-86a1-60f687c47a1a" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/8bf5fc32-e5fc-453f-9fc4-bcf44daa76ad" />
<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/08a25593-e436-4f81-a1b8-2193210d958c" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/6e1b84ee-690a-4774-83cb-7c797bf21e82" />
<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/4f1225b4-3d4f-46db-b80a-d4ac30ee0dae" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/152b0070-797b-464c-b4ee-7bff7ffe8e5b" />
<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/14691eb4-7145-4105-8158-99685923dbf8" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/24695c9d-740d-4234-9b4d-fe38ecc9b364" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/aed32aac-46bd-430d-826c-c4cebff44e43" />
<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/ab368eb0-fe32-483a-bfa0-81a9d16cd67e" />
<img width="1189" height="970" alt="image" src="https://github.com/user-attachments/assets/9d265bf4-5ce2-4d2b-b9e8-d00bdfdd6ac1" />
<img width="1389" height="788" alt="image" src="https://github.com/user-attachments/assets/34ea12c5-5826-4a3e-a43d-e3de344e1e05" />


# 3.3. Gamma Düzeltme (Gamma Correction)

Gamma düzeltme, görüntünün parlaklığını ayarlayan doğrusal olmayan bir dönüşümdür. γ < 1 için görüntü aydınlanır, γ > 1 için kararır.

**Uygulama**

```python
def gamma_correction(image, gamma):
    normalized = image / 255.0
    corrected = np.power(normalized, gamma)
    return (corrected * 255).astype(np.uint8)
```
**400 Numaralı Görüntü Analizi**

**Farklı γ Değerleri Karşılaştırması:**

— γ = 0.5: Görüntü aydınlatılmış, karanlık bölgelerdeki detaylar daha görünür hale gelmiş

— γ = 1.0: Orijinal görüntü, değişiklik yok

— γ = 2.0: Görüntü karartılmış, parlak bölgeler vurgulanmış

**Histogram Analizi**

— γ = 0.5: Histogramın sağa kayması (daha parlak değerlere)

— γ = 1.0: Orijinal dağılım

— γ = 2.0: Histogramın sola kayması (daha karanlık değerlere)

**Gamma Düzeltme Sonuçları**

**Pozlama düzeltmesi için esnek bir yöntem**

— γ = 0.5 yetersiz pozlanmış görüntüler için kullanışlı

— γ = 2.0 fazla pozlanmış görüntüler için kullanışlı

— Doğrusal olmayan dönüşüm özellikle orta tonları etkiler

<img width="1489" height="985" alt="загрузка (7)" src="https://github.com/user-attachments/assets/c0c42e23-436e-4c15-b653-51d6d40ae788" />
<img width="1489" height="985" alt="загрузка (8)" src="https://github.com/user-attachments/assets/8feb9299-1dba-4576-b2c4-9f0ca3ce762e" />

**Oluşturulan diğer görüntüler**

<img width="1490" height="985" alt="image" src="https://github.com/user-attachments/assets/3051c2eb-af79-488d-86a4-c696812b83ab" />
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/fcec04e9-95a3-445c-80bb-2c442c90c45b" />
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/e9edcb78-6432-4f3a-a3a8-2a3d6f6719a7" />
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/de88a72e-47c2-4e4a-9cc9-1d61bca72c36" />
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/134b9b04-d7b0-4718-8276-413a63ff3e9a" />
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/b1793912-9853-49fa-a14f-fccaf7f15a55" />
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/60aef44a-a3e0-4ff1-8390-f1df53bd61a2" />
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/79dbebf9-3894-41e4-8c60-ff2a5ed5c45b" />
<img width="1490" height="985" alt="image" src="https://github.com/user-attachments/assets/f1399def-4c6e-41c1-9f16-817cc9f851d8" />
<img width="1490" height="985" alt="image" src="https://github.com/user-attachments/assets/e7aa5c05-e673-413b-bf84-9b6473d195c7" />
<img width="1490" height="985" alt="image" src="https://github.com/user-attachments/assets/2b855b29-d18c-4dfa-a5ee-fb29c086e930" />
<img width="1490" height="985" alt="image" src="https://github.com/user-attachments/assets/a3915e6a-b28d-4ec6-af85-3e7ffd0c9445" />
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/10ca1f36-1483-4e65-ad8a-ea8a5943c33b" />
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/58de984a-80fb-48b5-8439-01e2fe801444" />
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/a233dba5-91eb-441d-978b-5a0788344ae6" />
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/b7014e1f-8acf-4f94-bc4d-34b9ad0ff664" />



# 4. Gürültü Azaltma

Bu bölümde, görüntülerdeki gürültüyü azaltmak ve görüntü kalitesini analiz etmek için iki farklı bulanıklaştırma yöntemi karşılaştırılmıştır: Medyan Bulanıklaştırma ve Gauss Bulanıklaştırma.

# 4.1. Median Blur Uygulama 

Medyan bulanıklaştırma, doğrusal olmayan bir filtreleme yöntemidir. Her piksel değeri, komşu piksellerin medyan değeri ile değiştirilir.

**Uygulama**

Farklı kernel boyutları ile medyan bulanıklaştırma
``` python
kernel_sizes = [3, 5, 7]

for ksize in kernel_sizes:
    img_median = cv2.medianBlur(img_rep_original, ksize)
```

**400 Numaralı Görüntü Analizi**

**Görsel Sonuçlar:**

— Kernel boyutu 3: Hafif düzleştirme, çoğu detay korunmuş

— Kernel boyutu 5: Orta düzeyde düzleştirme, iyi denge

— Kernel boyutu 7: Güçlü düzleştirme, ince detaylar kaybolabilir

**Histogram Analizi:**

— Tüm kernel boyutlarında histogram dağılımı korunmuş

— Piksel değerlerinde önemli değişiklik gözlemlenmemiş

— Görüntünün genel istatistiksel özellikleri korunmuş

**Medyan Bulanıklaştırma Sonuçları**

— Tuz-biber gürültüsü için mükemmel sonuçlar

— Kenarları doğrusal filtrelerden daha iyi korur

— Uç değerleri (outliers) etkili bir şekilde bastırır

<img width="1589" height="789" alt="загрузка (11)" src="https://github.com/user-attachments/assets/298a6cb0-b573-45cd-b615-841f6b1d86cf" />

<img width="1589" height="789" alt="загрузка (12)" src="https://github.com/user-attachments/assets/251c9ac6-5c4f-4227-861d-1c12337e52f5" />

**Oluşturulan diğer görüntüler**

<img width="1590" height="789" alt="image" src="https://github.com/user-attachments/assets/d98f6529-d432-412f-a609-0a47be610b81" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/c6d92b65-c96c-4311-9428-26e34601fec6" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/3849ee26-4905-43d5-b81f-90d145b8a12b" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/2259b062-c4a4-423d-9e62-7a88f0136aba" />
<img width="1589" height="788" alt="image" src="https://github.com/user-attachments/assets/3152bee5-21ec-4e0b-8ca5-0113ab36fa16" />
<img width="1589" height="788" alt="image" src="https://github.com/user-attachments/assets/dfaa31ed-5443-4a41-b2ac-a710fead8df4" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/2d7393e9-4f9b-4736-a869-0cb4dfe91943" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/305ee441-bced-427e-9e02-278d878f16d0" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/bb129203-22ed-4fcd-b85e-68322e613e72" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/b98130d0-eec3-4b54-9494-78ca3476ac5f" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/c17c4ce3-e6dc-4065-8c5a-8fbeab973e3d" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/c07b159c-e08b-4a50-9dae-47aa29cf4d7d" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/dd440f89-8f6a-4280-a75f-050a3d739e77" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/778d45b1-319d-4aad-ba1f-7caed7dc3c55" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/ecd24fb8-e201-48af-860a-2aa251f6cbd7" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/caf2099e-6bd3-40e4-bd8e-a7c8c8b6f0c6" />


# 4.2. Gaussian Blur Uygulama 

Gauss bulanıklaştırma, doğrusal bir filtreleme yöntemidir. Gaussian dağılımına dayalı ağırlıklı ortalama kullanır.

**Uygulama**

```python
kernel_sizes_gauss = [(3,3), (5,5), (7,7)]
sigma = 0  # OpenCV sigma değerini otomatik hesaplar

for ksize in kernel_sizes_gauss:
    img_gaussian = cv2.GaussianBlur(img_rep_original, ksize, sigma)
```

**400 Numaralı Görüntü Analizi**

**Görsel Sonuçlar:**

- Kernel boyutu 3x3: Minimal bulanıklık ile hafif düzleştirme

- Kernel boyutu 5x5: Yaygın olarak kullanılan orta düzeyde düzleştirme

- Kernel boyutu 7x7: Belirgin bulanıklık ile yoğun düzleştirme

**Histogram Analizi:**

- Yumuşak geçişler ve daha düzgün dağılım

- Gaussian dağılımına uygun ağırlıklı ortalama

- Gürültü azalması ile daha temel histogram profili

**Gauss Bulanıklaştırma Sonuçları**

- Gaussian dağılımına dayalı ağırlıklı ortalama

- Kenarlar dahil tüm görüntüyü düzgün şekilde yumuşatır

- Gaussian gürültüsünü azaltmada daha etkili

<img width="1589" height="789" alt="загрузка (13)" src="https://github.com/user-attachments/assets/51d592ba-c9b0-4803-a35c-ced84ff09f0e" />

<img width="1589" height="789" alt="загрузка (14)" src="https://github.com/user-attachments/assets/36862a10-dfcb-4814-b678-e85a15c44ba7" />

**Oluşturulan diğer görüntüler**

<img width="1590" height="789" alt="image" src="https://github.com/user-attachments/assets/f0dd2b90-08a4-477c-b7a3-698e21819ab5" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/56ba7c8f-989a-4f62-a9c7-2d637d2ec631" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/4294d35b-5d77-4fd5-9e88-8bb6c2c1e7d6" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/94c7ff0c-e90b-4eba-a399-63ff3b436757" />
<img width="1589" height="788" alt="image" src="https://github.com/user-attachments/assets/4306d9a7-68f0-4355-bd5b-d1b8012170c4" />
<img width="1589" height="788" alt="image" src="https://github.com/user-attachments/assets/73b5036f-fd85-4269-a987-1033f74bcafa" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/e7757df9-2dfd-4178-9bc7-9608166d31ce" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/63c581c2-3bb9-4621-b331-df786a557628" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/51bb4176-8a4c-43bf-99d3-149456cadd76" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/ffa2da4b-084e-4071-bd14-26844dade302" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/7158fd78-3b01-4568-8b0c-c0a6690e2fa9" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/38cb7533-0a7d-4cad-a400-0a0fd619b79f" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/9b8fee9a-c9c2-4bfe-a0b7-003c37d3301f" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/4d2579f9-e243-4528-819a-3ca27ac1dce9" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/a5398d01-c211-48c6-899a-9444184ef4c7" />
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/b180dd22-d271-43c2-b7c7-e09091543b5c" />


# Karşılaştırmalı Analiz: Medyan vs Gauss

**Görsel Karşılaştırma (Kernel=5)**

- Medyan Bulanıklaştırma: Kenarlar daha keskin, dokular daha iyi korunmuş

- Gauss Bulanıklaştırma: Daha homojen yumuşatma, kenarlar daha az belirgin

**Kenar Koruma Analizi**

Canny Kenar Tespiti Sonuçları:

- Orijinal görüntü: 1,663 kenar pikseli

- Medyan Bulanıklaştırma: 687 kenar pikseli (%58.69 kayıp)

- Gauss Bulanıklaştırma: 713 kenar pikseli (%57.13 kayıp)

<img width="1489" height="917" alt="загрузка (15)" src="https://github.com/user-attachments/assets/b0a11610-bb66-4d52-8573-51f026c0b179" />
<img width="1489" height="457" alt="загрузка (16)" src="https://github.com/user-attachments/assets/93ae69ad-9881-421d-a894-840b6d02d8b9" />


**Oluşturulan diğer görüntüler**


<img width="1489" height="917" alt="image" src="https://github.com/user-attachments/assets/ce0c63b5-5a4a-424c-9d49-5bae73d51c1d" />
<img width="1489" height="457" alt="image" src="https://github.com/user-attachments/assets/6bdea45b-0b63-4756-8d07-8b2227c11075" />
<img width="1489" height="917" alt="image" src="https://github.com/user-attachments/assets/d19ee1b1-bd23-4b51-90fd-4b0de29e19a8" />
<img width="1489" height="457" alt="image" src="https://github.com/user-attachments/assets/256fde67-5368-481d-bd0d-d6e9780f1738" />
<img width="1489" height="917" alt="image" src="https://github.com/user-attachments/assets/a438d13c-4eb2-4b82-84bb-14710048ae84" />
<img width="1489" height="457" alt="image" src="https://github.com/user-attachments/assets/0f4b8b02-7ad0-4790-8837-96265a0e6516" />
<img width="1489" height="917" alt="image" src="https://github.com/user-attachments/assets/c4aa4f44-7387-4ab4-b777-a9b9aa749395" />
<img width="1489" height="457" alt="image" src="https://github.com/user-attachments/assets/d1fb0552-02f2-49f0-8c88-df4ae75769fc" />
<img width="1489" height="917" alt="image" src="https://github.com/user-attachments/assets/b9553bf3-6ff0-4f65-8d97-68ac20a517f1" />
<img width="1489" height="457" alt="image" src="https://github.com/user-attachments/assets/f7454f53-02f3-449a-8719-76c38a15ed60" />
<img width="1489" height="917" alt="image" src="https://github.com/user-attachments/assets/bf2f23b0-6c0a-4453-bf51-4be8690b8cf4" />
<img width="1489" height="457" alt="image" src="https://github.com/user-attachments/assets/178f9d48-27ff-4833-80d1-99455f1c7eef" />
<img width="1489" height="917" alt="image" src="https://github.com/user-attachments/assets/0a51b49e-0254-4e82-bc31-2e1d2fc933fb" />
<img width="1489" height="457" alt="image" src="https://github.com/user-attachments/assets/de9b9bdf-6219-4df0-ab9c-2f4632ba2b5f" />
<img width="1489" height="917" alt="image" src="https://github.com/user-attachments/assets/f92ec902-ab3e-4474-b371-397f1a5d5c12" />
<img width="1489" height="457" alt="image" src="https://github.com/user-attachments/assets/9a18df41-ef23-4cd4-8dcb-99abb562145f" />


**Kalite Metrikleri Karşılaştırması**

<img width="815" height="259" alt="Снимок экрана 2025-11-26 150957" src="https://github.com/user-attachments/assets/645c1f72-d722-474e-bc40-31a8dfeb2381" />

• Median blur kenarları daha iyi korur mu? Evet, görüntülere göre Median Blur, Gaussian Blur'a kıyasla kenarları daha iyi korur. Bu sonucu, özellikle "Edge Detection: Comparing Edge Preservation - Image 400" başlıklı görüntüye bakarak çıkarıyoruz:
Original Edges (Orijinal Kenarlar): 1663 kenar pikseli, Median Blur Edges (Medyan Bulanıklık Kenarları): 687 kenar pikseli, Gaussian Blur Edges (Gauss Bulanıklık Kenarları): 713 kenar pikseli.

• Gaussian blur detay kaybına neden oluyor mu? Evet, görüntülere göre Gaussian Blur, hem RGB hem de Gri Tonlamalı görüntülerde detay kaybına neden olmuştur. Bu sonucu, özellikle "Comparison: Median vs Gaussian Blur (kernel=5) - Image 400" başlıklı görüntüye bakarak ve kenar tespiti sonuçlarını ("загрузка (16).png") dikkate alarak çıkarıyoruz.


**Oluşturulan diğer görüntüler**


<img width="476" height="161" alt="image" src="https://github.com/user-attachments/assets/f368d2a9-42f6-46e7-93b5-d4c6cfe68a65" />
<img width="483" height="171" alt="image" src="https://github.com/user-attachments/assets/833287ea-73c2-4ef0-841e-a73e6a1e879c" />
<img width="464" height="175" alt="image" src="https://github.com/user-attachments/assets/bf5b9637-cba6-47f5-a131-c7407c09d188" />
<img width="461" height="173" alt="image" src="https://github.com/user-attachments/assets/b8cb2559-bcf3-4b11-b9d6-1cc6f64e65f7" />
<img width="458" height="176" alt="image" src="https://github.com/user-attachments/assets/cac0ffda-34b2-4c7b-a796-27a52d6c3be5" />
<img width="454" height="171" alt="image" src="https://github.com/user-attachments/assets/ed88d937-ef49-4070-83e9-6f735c814653" />
<img width="428" height="170" alt="image" src="https://github.com/user-attachments/assets/7a47bbe1-4ce5-414b-b794-1757ef8a67ba" />
<img width="458" height="173" alt="image" src="https://github.com/user-attachments/assets/ec82125f-0db7-41ee-95ab-0f6e91bee8bc" />



# 5. Döndürme ve Ayna Çevirme (Flipping)

Bu bölümde, görüntü veri artırma (data augmentation) ve simetri analizi için döndürme ve ayna çevirme işlemleri uygulanmıştır. Bu teknikler, deri lezyonu görüntülerinin makine öğrenimi modellerinde daha iyi genelleme yapabilmesi için önemlidir.

# 5.1. Rastgele Döndürme 

Görüntüler, merkez etrafında belirli açılarla saat yönünün tersine döndürülmüştür. Küçük açılı döndürmeler (0-10°), görüntü bilgisinin çoğunu korurken veri çeşitliliği sağlar.

**Uygulama**

```python
def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    return rotated
```

**400 Numaralı Görüntü Analizi**

**Üretilen Rastgele Açılar:**

- Açı 1: 3.75°

- Açı 2: 9.51°

- Açı 3: 7.32°

- Açı 4: 5.99°

- Açı 5: 1.56°

**Görsel Sonuçlar:**

- Tüm döndürme açılarında görüntü kalitesi korunmuş

- Kenar pikselleri BORDER_REFLECT modu ile doldurulmuş

- Lezyon yapısı ve detaylar bozulmamış

**Histogram Analizi:**

- Piksel yoğunluk dağılımı büyük ölçüde değişmemiş

- İstatistiksel özellikler korunmuş

- Renk dağılımı sabit kalmış

<img width="853" height="382" alt="Снимок экрана 2025-11-26 153406" src="https://github.com/user-attachments/assets/b46fbc41-52e9-46c9-ac92-6206db0604d6" />

<img width="1789" height="690" alt="загрузка (17)" src="https://github.com/user-attachments/assets/923e10b7-610b-42b1-998b-58deb3c2a770" />

<img width="1789" height="690" alt="загрузка (18)" src="https://github.com/user-attachments/assets/31d8203b-b39f-4fc5-9420-5edaddc8bef9" />

<img width="1789" height="623" alt="загрузка (19)" src="https://github.com/user-attachments/assets/a159e395-8fca-4270-a157-c87b64e5cc8f" />

**Döndürme Analizi Sonuçları**

- Küçük açılar (0-10°): Görüntü bilgisinin çoğunu korur

- Kenar dolgusu: BORDER_REFLECT modu kullanılmış

- Makine öğrenimi: Veri artırma için kullanışlı

- Döndürme-değişmez özellikler: Modelin öğrenmesine yardımcı olur

- Histogram: Dağılım büyük ölçüde değişmez

- Tıbbi önem: Deri lezyonları her yönde görünebilir

# 5.2. Yatay Ayna Çevirme 

Görüntüler dikey eksen etrafında ayna çevirme işlemine tabi tutulmuştur.

**Uygulama**

```python
img_rgb_flipped = cv2.flip(img_rgb_original, 1)  # 1 = yatay çevirme
img_gray_flipped = cv2.flip(img_gray_original, 1)
```

**400 Numaralı Görüntü Analizi**

**Görsel Sonuçlar:**

- Orijinal RGB: Doğal görüntü düzeni

- Yatay Çevrilmiş RGB: Dikey eksende ayna görüntüsü

- Orijinal Grayscale: Standart gri tonlamalı

- Yatay Çevrilmiş Grayscale: Gri tonlamalı ayna görüntüsü

**Histogram Analizi:**

- Önemli bulgu: Histogram dağılımı TAMAMEN AYNI kalmış

- Piksel yoğunluk istatistikleri değişmemiş

- Uzamsal düzen değişmiş ancak istatistiksel özellikler korunmuş

<img width="792" height="296" alt="Снимок экрана 2025-11-26 153857" src="https://github.com/user-attachments/assets/37faf1fc-8e75-4608-986d-f5b22111f66d" />

<img width="1589" height="788" alt="загрузка (20)" src="https://github.com/user-attachments/assets/60bf8e85-d346-4942-8cf6-cdaf13c55d56" />

<img width="1589" height="741" alt="загрузка (21)" src="https://github.com/user-attachments/assets/dfa60cc3-a35a-4250-b4a6-a3ef49514294" />

**Yatay Çevirme Analizi Sonuçları**

- Dikey eksende ayna görüntüsü oluşturur

- Piksel yoğunluk dağılımı (histogram) AYNI kalır

- Uzamsal düzen değişir ancak istatistiksel özellikler değişmez

- Tıbbi görüntülemede veri artırma için önemli

# Simetri Analizi (Symmetry Analysis)

Orijinal ve çevrilmiş görüntüler arasındaki fark haritaları hesaplanarak simetri skorları elde edilmiştir.

**Simetri Hesaplama**

```python
def analyze_symmetry(original, flipped):
    diff = np.abs(original.astype(float) - flipped.astype(float))
    symmetry_score = np.mean(diff)
    return diff, symmetry_score
```

**400 Numaralı Görüntü Simetri Sonuçları**

**Simetri Skorları (düşük = daha simetrik):**

- RGB Simetri Skoru: 40.16

- Gri Tonlamalı Simetri Skoru: 38.27

**Fark Haritası Analizi:**

- Yüksek fark değerleri (ısı haritasında daha parlak): Asimetrik bölgeleri gösterir

- Deri lezyonları genellikle asimetri gösterir (tanısal özellik)

- Melanom tespiti için ABCDE kriterlerinden biri:

• A = Asimetri (Asymmetry)

• B = Kenar (Border)

• C = Renk (Color)

• D = Çap (Diameter)

• E = Gelişim (Evolution)

<img width="1489" height="972" alt="загрузка (22)" src="https://github.com/user-attachments/assets/fe9305a7-147b-4cb7-b240-b06755be3d1e" />

# Simetri Gözlemleri

- Asimetrik bölgeler tanı için önemli ipuçları sağlar

- Çevirme işlemi, lezyonun her iki tarafta tutarlı özelliklere sahip olup olmadığını belirlemeye yardımcı olur

- Yüksek simetri skorları, lezyonun asimetrik doğasını doğrular

# Birleşik Dönüşümler (Combined Transformations)

Döndürme ve çevirme işlemleri birleştirilerek daha karmaşık veri artırma teknikleri gösterilmiştir.

**400 Numaralı Görüntü için Birleşik Dönüşüm:**

- Döndürme açısı: 1.56°

- İşlem: Döndürme + Yatay Çevirme

**Görsel Sonuçlar:**

- Orijinal: Temel referans

- Döndürülmüş (1.56°): Hafif açısal değişim

- Çevrilmiş: Yatay ayna görüntüsü

- Döndürülmüş + Çevrilmiş: Karmaşık dönüşüm

**Birleşik Dönüşümlerin Avantajları:**

- Daha zengin veri çeşitliliği

- Modelin çeşitli görüntü varyasyonlarına adaptasyonu

- Gerçek dünya koşullarını daha iyi temsil etme

- Overfitting'in azaltılması

<img width="1589" height="737" alt="загрузка (23)" src="https://github.com/user-attachments/assets/535d25f7-b238-4386-972e-9504fdaacad0" />

# 6. Frekans Alanında Filtreleme (FFT)

Bu bölümde, görüntü işleme tekniklerinden Fourier Dönüşümü ve frekans alanında filtreleme yöntemleri ele alınmıştır. FFT (Fast Fourier Transform), görüntülerin frekans bileşenlerini analiz etmek ve çeşitli filtreleme işlemleri uygulamak için kullanılmıştır.

# 6.1. Fourier Dönüşümü 

Fourier Dönüşümü, bir görüntüyü uzaysal alandan frekans alanına dönüştürerek, görüntünün farklı frekans bileşenlerini analiz etmemizi sağlar.

**Uygulama**

```python
def apply_fft(image):
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    magnitude_spectrum_log = np.log(magnitude_spectrum + 1)
    phase_spectrum = np.angle(fft_shift)
    return fft_shift, magnitude_spectrum, magnitude_spectrum_log, phase_spectrum
```

**400 Numaralı Görüntü Analizi**

**Frekans Spektrumu Görselleştirmesi:**

- Orijinal RGB: Renkli referans görüntü

- RGB → Gri: FFT için dönüştürülmüş gri tonlamalı

- Genlik Spektrumu: Frekans dağılımını gösteren log ölçekli harita

- Faz Spektrumu: Uzaysal bilgi içeren faz bileşeni

**Temel Özellikler:**

- Spektrum merkezi: Düşük frekanslar (yumuşak değişimler)

- Spektrum kenarları: Yüksek frekanslar (keskin detaylar, kenarlar)

- Parlak merkez: Baskın düşük frekans bileşenlerini gösterir

<img width="1790" height="859" alt="загрузка (39)" src="https://github.com/user-attachments/assets/1e36acc8-b322-4084-886e-bf6ef652cf37" />

**Fourier Dönüşümü Analizi**

- Genlik spektrumu frekans dağılımını gösterir

- Faz spektrumu uzaysal bilgi içerir

- Düşük frekanslar görüntünün genel yapısını belirler

- Yüksek frekanslar detayları ve kenarları temsil eder

# 6.2. Alçak Geçiren Filtre Uygulama 

Alçak geçiren filtre, yüksek frekans bileşenlerini bastırarak görüntüyü yumuşatır. Dairesel maske kullanılarak frekans alanında filtreleme yapılır.

**Filtre Maskesi Oluşturma**

```python
def create_lowpass_filter(shape, cutoff_radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    mask = np.zeros((rows, cols), dtype=np.float32)
    mask[distance <= cutoff_radius] = 1.0
    return mask
```

<img width="722" height="277" alt="Снимок экрана 2025-11-26 155358" src="https://github.com/user-attachments/assets/6f5b8a41-0d43-4816-87f0-390f84a4b5c2" />

**400 Numaralı Görüntü Filtreleme Sonuçları**

**RGB-Gri Görüntü için:**

- Yarıçap 30: Güçlü filtreleme, çok bulanık, sadece büyük yapılar görünür

- Yarıçap 50: Orta düzey filtreleme, iyi denge, lezyon şekli korunmuş

- Yarıçap 100: Zayıf filtreleme, çoğu detay korunmuş

**Orijinal Gri Görüntü için:**

- Benzer sonuçlar gözlemlenmiş

- Filtreleme etkisi tutarlı

<img width="1990" height="1122" alt="загрузка (40)" src="https://github.com/user-attachments/assets/542bf0fb-d554-4be0-b779-980cae8342b3" />

<img width="1990" height="1122" alt="загрузка (41)" src="https://github.com/user-attachments/assets/fd8181e0-f7dc-4290-bc7b-edee6d6c68ff" />

**Alçak Geçiren Filtre Analizi**

- Küçük yarıçap = daha fazla bulanıklık = daha fazla yüksek frekans kaldırma

- Alçak geçiren filtre gürültüyü ve ince detayları kaldırır

- Yumuşatma için kullanışlı ancak kenar bilgisini kaybettirir

# 6.3. Ters Fourier (Inverse FFT) 

Frekans alanında filtrelenmiş görüntüyü tekrar uzaysal alana dönüştürme işlemidir.

**Yeniden Yapılandırma Süreci:**

1. Orijinal görüntü → Uzaysal alan

2. FFT Spektrumu → Frekans alanı

3. Filtrelenmiş Spektrum → Maske uygulanmış

4. Yeniden Yapılandırılmış → Ters FFT ile uzaysal alana dönüş

**400 Numaralı Görüntü için Yeniden Yapılandırma (Yarıçap=50)**

**RGB-Gri Yol:**

- Orijinal görüntü başarıyla yeniden yapılandırılmış

- Yumuşatılmış versiyon elde edilmiş

**Orijinal Gri Yol:**

- Benzer kalitede yeniden yapılandırma

- Filtreleme etkisi tutarlı

<img width="1789" height="831" alt="загрузка (42)" src="https://github.com/user-attachments/assets/0b509e93-5ac4-4897-b80d-d58b98ad0373" />

**Ters FFT Analizi**

• Frekans alanını uzaysal alana dönüştürür

• Filtrelenmiş spektrum = yumuşatılmış uzaysal görüntü

• İşlem tersinirdir (FFT ↔ IFFT)

• Frekans alanında filtreleme = uzaysal alanda konvolüsyon

# 6.4. Karşılaştırma 

<img width="752" height="276" alt="Снимок экрана 2025-11-26 155809" src="https://github.com/user-attachments/assets/bc162686-bd41-467f-8152-5006be4a4a70" />

<img width="1489" height="917" alt="загрузка (28)" src="https://github.com/user-attachments/assets/e650ed2b-f76c-4e70-886c-b6d58a3de044" />

<img width="1187" height="495" alt="загрузка (29)" src="https://github.com/user-attachments/assets/41099690-dcd5-4b16-89cd-5e75d31705dc" />

**Fark Analizi:**

• Orijinal görüntüler arası fark:

- Ortalama mutlak fark: 0.0000

- Maksimum mutlak fark: 0.0000

• Filtrelenmiş görüntüler arası fark:

- Ortalama mutlak fark: 0.0000

- Maksimum mutlak fark: 0.0000

**Önemli Gözlemler:**

• RGB'den griye dönüşüm ve orijinal gri görüntü arasında istatistiksel fark yok

• Filtreleme sonrası standart sapmada hafif azalma (29.72 → 29.21)

• Minimum ve maksimum değerlerde filtreleme nedeniyle değişim

_____________________________________________________________

# FFT Sonuçlarının Özel Olarak Yorumlanması

Görüntü İşleme ve FFT analizi, verilen görsellerde bir görüntünün frekans bileşenlerini ayrıştırma, filtreleme ve yeniden yapılandırma sürecini göstermektedir. 

1. Fourier Dönüşümü ve Frekans Spektrumu Yorumu: Görüntünün Hızlı Fourier Dönüşümü (FFT) ile elde edilen Genlik Spektrumu, görüntüdeki frekans içeriğinin dağılımını gösterir. Spektrumun merkezindeki parlaklık, görüntünün ana yapısını oluşturan Düşük Frekansları (genel parlaklık ve büyük ölçekli yapılar) temsil ederken, merkezden uzaklaşan kısım Yüksek Frekansları (ince detaylar, kenarlar, gürültü) gösterir. Görüntüdeki yüksek frekans bileşenlerinin merkeze göre hızla zayıflaması, görüntünün büyük çoğunluğunun büyük ölçekli yapılar ve yumuşak geçişlerden oluştuğunu belirtir. Faz spektrumu ise, bileşenlerin uzamsal yerleşimini belirleyen kritik yapısal bilgiyi içerir.

2. İstatistiksel Karşılaştırma: Orijinal ve FFT ile filtrelenmiş görüntüler arasındaki istatistiksel karşılaştırma, filtrenin etkisini nicel olarak doğrular. Filtreleme sonrası ortalama piksel değeri ($133.60$) korunurken, Standart Sapma ($29.72$'den $29.21$'e) düşer ve Min/Max aralığı ($1$-$250$'den $14.34$-$216.97$'ye) daralır. Bu, filtrenin yüksek frekanslı bileşenleri (gürültü ve en keskin kontraslar) başarıyla azalttığını, dolayısıyla piksel değerlerindeki genel varyasyonu yumuşattığını gösterir.

3. Düşük Geçişli Filtrelemenin (Low-Pass Filter) Etkisi: Düşük Geçişli Filtre (LPF), frekans spektrumunda merkezi bölgeyi (düşük frekanslar) koruyarak yüksek frekansları engeller ve böylece mekansal alanda yumuşatma/bulanıklaştırma etkisi yaratır. Filtre yarıçapı ($r$) filtrenin gücünü belirler: $r=30$ (küçük yarıçap) güçlü filtreleme ve belirgin bulanıklık yaparak gürültüyü azaltır; $r=50$ orta güçte denge sağlar; $r=100$ (büyük yarıçap) ise zayıf filtreleme yaparak detay korumayı hedefler. Bu karşılaştırma, istenen yumuşatma derecesine ulaşmak için $r$ değerinin kritik bir parametre olduğunu gösterir.

4. FFT Sonuçlarının Karşılaştırılması: Görüntülerin son karşılaştırması, filtrenin nihai görsel etkisini özetler. Orijinal gri tonlamalı görüntülere kıyasla, Düşük Geçişli Filtre uygulandıktan sonra elde edilen görüntülerde bariz bir yumuşama ve bulanıklık gözlemlenir. Özellikle lezyonun ince detayları ve keskin kenarları, filtrenin yüksek frekansları ortadan kaldırması sonucu yumuşatılmış, ancak ana hatları ve büyük ölçekli yapıları korunmuştur. Bu görsel doğrulama, düşük geçişli filtrenin gürültüyü azaltma ve detayları yumuşatma işlevini başarıyla yerine getirdiğini gösterir.

5. Fark Analizi Yorumu: Fark analizi, uygulanan işlemlerin tutarlılığını kesinleştirir. Orijinal RGB'den Gri Tonlamaya dönüştürülen görüntü ile orijinal Gri Tonlamalı görüntü arasındaki farkın sıfır (ortalama 0.0000) olması, giriş verilerinin özdeş olduğunu kanıtlar. Aynı şekilde, bu iki farklı girişten elde edilen filtrelenmiş sonuçların farkının da sıfır (ortalama 0.0000) olması, tüm FFT Düşük Geçişli Filtreleme sürecinin, başlangıç gri tonlamalı verinin kaynağından bağımsız olarak mükemmel bir tutarlılıkla aynı sonuçları ürettiğini gösterir.


# 7. Keskinleştirme ve Enterpolasyon

Bu bölümde, görüntü keskinleştirme teknikleri ve enterpolasyon yöntemleri ele alınmıştır. Unsharp masking yöntemi ile görüntüler keskinleştirilmiş ve çeşitli enterpolasyon teknikleri ile görüntü boyutlandırma işlemleri gerçekleştirilmiştir.

# 7.1. Unsharp Masking ile Keskinleştirme 

Unsharp masking, görüntülerdeki kenarları ve ince detayları vurgulamak için kullanılan bir keskinleştirme tekniğidir. Temel prensip, orijinal görüntüden bulanıklaştırılmış versiyonu çıkararak bir maske oluşturmak ve bu maskeyi ağırlıklı olarak orijinal görüntüye eklemektir.

**Uygulama**

```python
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    # 1. Adım: Bulanık versiyon oluştur
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # 2. Adım: Maske hesapla (orijinal - bulanık)
    mask = cv2.subtract(image, blurred)
    
    # 3. Adım: Ağırlıklı maskeyi orijinale ekle
    sharpened = cv2.addWeighted(image, 1.0, mask, amount, 0)
    
    return sharpened, blurred, mask
```

<img width="617" height="235" alt="Снимок экрана 2025-11-26 164004" src="https://github.com/user-attachments/assets/ac8a43e4-07f1-4078-9049-86ed544a53e6" />

**400 Numaralı Görüntü Analizi**

**RGB Görüntü için Sonuçlar:**

- Hafif Keskinleştirme: İnce geliştirme, doğal görünüm

- Orta Keskinleştirme: İyi denge, kenarları iyi geliştiriyor

- Güçlü Keskinleştirme: Maksimum detay, gürültüyü artırabilir

**Grayscale Görüntü için Sonuçlar:**

- Benzer etkiler gözlemlenmiş

- Kenar geliştirme daha belirgin

<img width="1589" height="1524" alt="загрузка (50)" src="https://github.com/user-attachments/assets/019d95e2-492a-4d73-b956-ca3aca7b0dc1" />

<img width="1589" height="1524" alt="загрузка (51)" src="https://github.com/user-attachments/assets/9f01efa5-e180-43ce-b9a0-7e3a223a296e" />

**İşlem Adımları:**

- Orijinal görüntü - Referans

- Bulanık versiyon - Gaussian filtre uygulanmış

- Unsharp mask - Orijinal ile bulanık arasındaki fark

- Keskinleştirilmiş - Orijinal + (Amount × Mask)

<img width="1389" height="985" alt="загрузка (52)" src="https://github.com/user-attachments/assets/9cb09fa6-0caf-4786-9f8c-61d4fc42131a" />

**Histogram Karşılaştırması**

- Orijinal RGB Histogram: Doğal piksel dağılımı

- Keskinleştirilmiş RGB Histogram: Benzer dağılım, istatistikler korunmuş

- Orijinal Gri Histogram: Standart dağılım

- Keskinleştirilmiş Gri Histogram: Minimal değişiklik

**Unsharp Masking Analizi**

- Kenarları ve ince detayları geliştirir

- Deri lezyonu sınırlarını vurgulamak için önemli

- Gürültüyü artırabileceğinden dikkatli kullanılmalı

- Amount parametresi keskinleştirme gücünü kontrol eder

# 7.2. Bicubic Enterpolasyon 

Görüntüleri 2x büyütmek için çeşitli enterpolasyon yöntemleri karşılaştırılmıştır. Bicubic enterpolasyon, kalite ve hız arasında iyi denge sağladığı için tıbbi görüntülerde önerilir.

**Boyut Değişikliği:**

- Orijinal RGB: (459, 600, 3) → 275,400 piksel

- Büyütülmüş RGB: (900, 1200, 3) → 1,080,000 piksel

- Orijinal Gri: (459, 600) → 275,400 piksel

**Enterpolasyon Yöntemleri Karşılaştırması**

1. Nearest Neighbor (En Yakın Komşu)
   
   • Hız: En hızlı

   • Kalite: Düşük, bloklu artefaktlar

   • Avantaj: Keskin kenarları korur

   • Dezavantaj: Pikselleşme

2. Bilinear (Çift Doğrusal)
   
   • Hız: Hızlı

   • Kalite: Orta, hafif bulanık

   • Avantaj: Yumuşak geçişler

   • Dezavantaj: Detay kaybı

3. Bicubic (Çift Kübik)

   • Hız: Orta

   •   Kalite: İyi, bilineardan daha yumuşak

   • Avantaj: İyi kalite/hız dengesi

   • Dezavantaj: Bilineardan yavaş

4. Lanczos

   • Hız: En yavaş

   • Kalite: En iyi, en keskin

   • Avantaj: Maksimum kalite

   • Dezavantaj: Halkalama (ringing) artefaktları olabilir

   • Büyütülmüş Gri: (900, 1200) → 1,080,000 piksel


**400 Numaralı Görüntü Enterpolasyon Sonuçları**

**RGB Görüntü için:**

— Bicubic: En iyi genel performans

— Lanczos: Daha keskin ancak daha yavaş

— Bilinear: Kabul edilebilir kalite

— Nearest: Düşük kalite

**Grayscale Görüntü için:**

— Benzer sonuçlar gözlemlenmiş

— Bicubic önerilen yöntem olarak öne çıkmış

<img width="1990" height="788" alt="загрузка (53)" src="https://github.com/user-attachments/assets/0631bc45-9642-4f29-bbdb-0490d6b1a6c4" />

<img width="1990" height="788" alt="загрузка (54)" src="https://github.com/user-attachments/assets/1e40b5c6-61fd-47fb-b505-a47277be2647" />

**Enterpolasyon Yöntemleri Analizi**

— Nearest Neighbor: En hızlı, bloklu artefaktlar, keskin kenarları korur

— Bilinear: Hızlı, yumuşak ama hafif bulanık

— Bicubic: İyi denge, bilineardan daha yumuşak, daha yavaş

— Lanczos: En iyi kalite, en keskin, en yavaş, halkalama olabilir

— Tıbbi görüntüler için Bicubic önerilir - iyi kalite/hız dengesi

**Kalite Değerlendirmesi**

Kenar yoğunluğu, görüntüdeki detay seviyesini ölçmek için kullanılmıştır.

```python
def calculate_edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / edges.size
    return edge_density, edges
```
<img width="471" height="114" alt="Снимок экрана 2025-11-26 203019" src="https://github.com/user-attachments/assets/e39109d3-04d2-4627-b94f-7e373e96d60c" />

<img width="1189" height="978" alt="загрузка (55)" src="https://github.com/user-attachments/assets/a8221311-30a9-46d4-b500-70bbf09a63bd" />


**Kenar Koruma Analizi:**

- Büyütme işlemi sonrasında kenar yoğunluğunda hafif azalma

- Bicubic enterpolasyon kenar bilgisini iyi korumuş

- Keskinleştirme işlemi kenar yoğunluğunu artırmış

**İşlem Hattı Görselleştirmesi**

<img width="1489" height="921" alt="загрузка (56)" src="https://github.com/user-attachments/assets/659d92ad-53dc-4873-985d-4a7e570dce33" />


**Tam İşlem Hattı:**

1. Orijinal Görüntü - Ham veri

2. Keskinleştirilmiş - Kenar geliştirme

3. Büyütülmüş (2x) - Boyut artırma

**RGB İşlem Hattı:**

• Orijinal RGB → Keskinleştirilmiş RGB → Büyütülmüş RGB

• Boyut: (459, 600) → (459, 600) → (900, 1200)

**GrayScale İşlem Hattı:**

• Orijinal Gri → Keskinleştirilmiş Gri → Büyütülmüş Gri

• Boyut: (459, 600) → (459, 600) → (900, 1200)

<img width="721" height="212" alt="Снимок экрана 2025-11-26 203328" src="https://github.com/user-attachments/assets/ea6d747e-6a45-43a6-95b4-67c32cb3e2ad" />





