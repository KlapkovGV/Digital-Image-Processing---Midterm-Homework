# Digital-Image-Processing - Midterm-Homework
I applied both RGB (color) and grayscale image processing techniques on ISIC skin lesion images

# 1. Veri Yükleme 

Proje Genel Bakış

Bu proje, 9 farklı cilt lezyonu sınıfı içeren bir cilt kanseri veri setinin analizini içermektedir. Veri seti Kaggle'dan kaynaklanmakta olup, analiz ve işleme için 2.357 tıbbi görüntü içermektedir.

Veri Seti Bilgileri:

— Kaynak: Kaggle "skin-cancer9-classesisic" veri seti

— Toplam Görüntü: 2.357

— Sınıflar: 9 farklı cilt lezyonu türü

Sınıf Dağılımı / Sınıf Adı	Adet

— pigmented benign keratosis	/ 478

— melanoma	/ 454

— basal cell carcinoma	/ 392

— nevus	373

— squamous cell carcinoma	/ 197

— vascular lesion	/ 142

— actinic keratosis	/ 130

— dermatofibroma	/ 111

— seborrheic keratosis	/ 80

Teknik Uygulama

# 1.1 Veri Yükleme ve İlk Analiz

Kullanılan Kütüphaneler
pandas, numpy - Veri manipülasyonu

matplotlib, seaborn - Görselleştirme

cv2, PIL - Görüntü işleme

pathlib, os - Dosya sistemi operasyonları

kagglehub - Veri seti indirme

# 1.2. Veri Setinin Yüklenmesi
Veri seti başarıyla indirildi ve aşağıdaki yapıya sahip bir pandas DataFrame'e yüklendi:

file_path: Görüntü dosyasının tam yolu

file_name: Görüntü dosyasının adı

class: Cilt lezyonunun sınıflandırması

file_size_kb: Kilobyte cinsinden dosya boyutu

# 1.3. Veri Özelliklerinin İncelenmesi

Çözünürlük Analizi Sonuçları

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
   
Analiz için 9 rastgele görüntü seçildi:

Seçilen Görüntüler: 400, 403, 399, 800, 1002, 2021, 20, 80, 693 indeksleri

Sınıf Dağılımı:

Pigmented benign keratosis: 3 görüntü

Melanoma: 4 görüntü

Nevus: 1 görüntü

Basal cell carcinoma: 1 görüntü

<img width="753" height="1995" alt="загрузка (1)" src="https://github.com/user-attachments/assets/0e13b40a-2202-4e4e-a13a-ef806ea55562" />

# 2.2. Rastgele Görüntülerin İstatistiksel Özellikleri

Her görüntü için RGB ve gri tonlamalı istatistikleri hesaplandı:

Örnek İstatistikler:

Görüntü 1: RGB Mean: 136.74, Grayscale Mean: 133.59

Görüntü 2: RGB Mean: 212.60, Grayscale Mean: 211.18

Görüntü 3: RGB Mean: 146.20, Grayscale Mean: 140.99

# 2.3. Histogram Çizimi (RGB + Grayscale)

Tüm görüntüler için RGB ve gri tonlamalı histogramlar oluşturuldu:

Histogram Analiz Özeti:

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

Bu bölümde, görüntü iyileştirme için üç yöntem ele alınmıştır: kontrast germe, histogram eşitleme ve gamma düzeltme. Bu yöntemler, cilt lezyonu görüntülerindeki detayların görünürlüğünü artırmak için hem RGB hem de gri tonlamalı görüntülere uygulanmıştır.

# 3.1. Kontrast Germe (Stretching) İşlemi

Kontrast germe, piksel değerlerinin aralığını tüm mevcut [0, 255] aralığını kullanacak şekilde genişleten doğrusal bir dönüşümdür.

Uygulama

 stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

400 Numaralı Görüntü Analizi

Görsel Karşılaştırma

— Original RGB: Görüntü düşük kontrasta sahip, lezyon detayları zor seçiliyor

— Stretched RGB: Kontrast belirgin şekilde iyileşmiş, lezyon sınırları daha net görünüyor

— Original Grayscale: RGB versiyonu gibi düşük kontrastlı

— Stretched Grayscale: Lezyon dokuları ve sınırlarının görünürlüğü artmış

Histogram Analizi

— Original RGB Histogram: Piksel değerleri dar bir aralıkta yoğunlaşmış

— Stretched RGB Histogram: Aralık tüm [0, 255]'e genişletilmiş, dağılım daha düzgün hale gelmiş

— Original Grayscale Histogram: Aralığın orta kısmında yoğunluk

— Stretched Grayscale Histogram: Dağılım genişlemiş, kontrast iyileşmesi doğrulanmış

<img width="1189" height="970" alt="загрузка (3)" src="https://github.com/user-attachments/assets/e629e717-4e81-46d3-996a-78f88b1871e4" />

<img width="1389" height="788" alt="загрузка (37)" src="https://github.com/user-attachments/assets/86e2a5be-22a2-4824-8388-a5725bf3c750" />

Kontrast Germe Sonuçları

— Düşük kontrastlı görüntüler için etkili

— Piksel değer aralığını genişletirken histogram şeklini değiştirmez

— Uygulaması ve hesaplaması basit

# 3.2. Histogram Eşitleme (Histogram Equalization)

Yöntem Açıklaması

Histogram eşitleme, piksel yoğunluk değerlerini tüm aralıkta düzgün bir dağılım elde edecek şekilde yeniden dağıtır.

Uygulama

RGB görüntüler için, renk bilgisini korumak amacıyla YCrCb renk uzayına dönüşüm yapılarak sadece parlaklık kanalı (Y) eşitlenir.

400 Numaralı Görüntü Analizi

Görsel Karşılaştırma

Original RGB: Doğal renkler ancak düşük kontrast

Equalized RGB (via YCrCb): Renkler korunmuş, kontrast iyileşmiş

Original Grayscale: Standart gri tonlamalı görünüm

Equalized Grayscale: Detay görünürlüğünde belirgin iyileşme

Histogram Analizi

Original RGB Histogram: Tepe noktalı düzgün olmayan dağılım

Equalized RGB Histogram: Yoğunluklar daha düzgün dağılmış

Original Grayscale Histogram: Tıbbi görüntüler için tipik dağılım

Equalized Grayscale Histogram: Düzgün dağılıma yaklaşmış

Histogram Eşitleme Sonuçları

Kontrast germeye göre daha agresif bir kontrast iyileştirme yöntemi

YCrCb dönüşümü ile RGB'de renklerin korunması

Düzgün bölgelerde gürültüyü artırabilir

<img width="1189" height="970" alt="загрузка (5)" src="https://github.com/user-attachments/assets/324c337f-fae9-4073-a76f-447499def099" />

<img width="1389" height="788" alt="загрузка (6)" src="https://github.com/user-attachments/assets/f862ae47-fa5e-46a4-bd2a-a07ba8960f7f" />


# 3.3. Gamma Düzeltme (Gamma Correction)

Yöntem Açıklaması

Gamma düzeltme, görüntünün parlaklığını ayarlayan doğrusal olmayan bir dönüşümdür. γ < 1 için görüntü aydınlanır, γ > 1 için kararır.

Uygulama

def gamma_correction(image, gamma):
    normalized = image / 255.0
    corrected = np.power(normalized, gamma)
    return (corrected * 255).astype(np.uint8)

400 Numaralı Görüntü Analizi

Farklı γ Değerleri Karşılaştırması:

— γ = 0.5: Görüntü aydınlatılmış, karanlık bölgelerdeki detaylar daha görünür hale gelmiş

— γ = 1.0: Orijinal görüntü, değişiklik yok

— γ = 2.0: Görüntü karartılmış, parlak bölgeler vurgulanmış

Histogram Analizi

— γ = 0.5: Histogramın sağa kayması (daha parlak değerlere)

— γ = 1.0: Orijinal dağılım

— γ = 2.0: Histogramın sola kayması (daha karanlık değerlere)

Gamma Düzeltme Sonuçları

Pozlama düzeltmesi için esnek bir yöntem

— γ = 0.5 yetersiz pozlanmış görüntüler için kullanışlı

— γ = 2.0 fazla pozlanmış görüntüler için kullanışlı

— Doğrusal olmayan dönüşüm özellikle orta tonları etkiler

<img width="1489" height="985" alt="загрузка (7)" src="https://github.com/user-attachments/assets/c0c42e23-436e-4c15-b653-51d6d40ae788" />

<img width="1489" height="985" alt="загрузка (8)" src="https://github.com/user-attachments/assets/8feb9299-1dba-4576-b2c4-9f0ca3ce762e" />

# 4. Gürültü Azaltma

Bu bölümde, görüntülerdeki gürültüyü azaltmak ve görüntü kalitesini analiz etmek için iki farklı bulanıklaştırma yöntemi karşılaştırılmıştır: Medyan Bulanıklaştırma ve Gauss Bulanıklaştırma.

# 4.1. Median Blur Uygulama 

Medyan bulanıklaştırma, doğrusal olmayan bir filtreleme yöntemidir. Her piksel değeri, komşu piksellerin medyan değeri ile değiştirilir.

Uygulama

Farklı kernel boyutları ile medyan bulanıklaştırma
kernel_sizes = [3, 5, 7]

for ksize in kernel_sizes:
    img_median = cv2.medianBlur(img_rep_original, ksize)

400 Numaralı Görüntü Analizi

Görsel Sonuçlar:

— Kernel boyutu 3: Hafif düzleştirme, çoğu detay korunmuş

— Kernel boyutu 5: Orta düzeyde düzleştirme, iyi denge

— Kernel boyutu 7: Güçlü düzleştirme, ince detaylar kaybolabilir

Histogram Analizi:

— Tüm kernel boyutlarında histogram dağılımı korunmuş

— Piksel değerlerinde önemli değişiklik gözlemlenmemiş

— Görüntünün genel istatistiksel özellikleri korunmuş

Medyan Bulanıklaştırma Sonuçları

— Tuz-biber gürültüsü için mükemmel sonuçlar

— Kenarları doğrusal filtrelerden daha iyi korur

— Uç değerleri (outliers) etkili bir şekilde bastırır

<img width="1589" height="789" alt="загрузка (11)" src="https://github.com/user-attachments/assets/298a6cb0-b573-45cd-b615-841f6b1d86cf" />

<img width="1589" height="789" alt="загрузка (12)" src="https://github.com/user-attachments/assets/251c9ac6-5c4f-4227-861d-1c12337e52f5" />

# 4.2. Gaussian Blur Uygulama 

Gauss bulanıklaştırma, doğrusal bir filtreleme yöntemidir. Gaussian dağılımına dayalı ağırlıklı ortalama kullanır.

Uygulama

kernel_sizes_gauss = [(3,3), (5,5), (7,7)]
sigma = 0  # OpenCV sigma değerini otomatik hesaplar

for ksize in kernel_sizes_gauss:
    img_gaussian = cv2.GaussianBlur(img_rep_original, ksize, sigma)

400 Numaralı Görüntü Analizi

Görsel Sonuçlar:

— Kernel boyutu 3x3: Minimal bulanıklık ile hafif düzleştirme

— Kernel boyutu 5x5: Yaygın olarak kullanılan orta düzeyde düzleştirme

— Kernel boyutu 7x7: Belirgin bulanıklık ile yoğun düzleştirme

Histogram Analizi:

— Yumuşak geçişler ve daha düzgün dağılım

— Gaussian dağılımına uygun ağırlıklı ortalama

— Gürültü azalması ile daha temel histogram profili

Gauss Bulanıklaştırma Sonuçları

— Gaussian dağılımına dayalı ağırlıklı ortalama

— Kenarlar dahil tüm görüntüyü düzgün şekilde yumuşatır

— Gaussian gürültüsünü azaltmada daha etkili

<img width="1589" height="789" alt="загрузка (13)" src="https://github.com/user-attachments/assets/51d592ba-c9b0-4803-a35c-ced84ff09f0e" />

<img width="1589" height="789" alt="загрузка (14)" src="https://github.com/user-attachments/assets/36862a10-dfcb-4814-b678-e85a15c44ba7" />

Karşılaştırmalı Analiz: Medyan vs Gauss

Görsel Karşılaştırma (Kernel=5)

— Medyan Bulanıklaştırma: Kenarlar daha keskin, dokular daha iyi korunmuş

— Gauss Bulanıklaştırma: Daha homojen yumuşatma, kenarlar daha az belirgin

Kenar Koruma Analizi

Canny Kenar Tespiti Sonuçları:

— Orijinal görüntü: 1,663 kenar pikseli

— Medyan Bulanıklaştırma: 687 kenar pikseli (%58.69 kayıp)

— Gauss Bulanıklaştırma: 713 kenar pikseli (%57.13 kayıp)

<img width="1489" height="917" alt="загрузка (15)" src="https://github.com/user-attachments/assets/b0a11610-bb66-4d52-8573-51f026c0b179" />

<img width="1489" height="457" alt="загрузка (16)" src="https://github.com/user-attachments/assets/93ae69ad-9881-421d-a894-840b6d02d8b9" />

Kalite Metrikleri Karşılaştırması

<img width="815" height="259" alt="Снимок экрана 2025-11-26 150957" src="https://github.com/user-attachments/assets/645c1f72-d722-474e-bc40-31a8dfeb2381" />

• Median blur kenarları daha iyi korur mu? Evet, görüntülere göre Median Blur, Gaussian Blur'a kıyasla kenarları daha iyi korur. Bu sonucu, özellikle "Edge Detection: Comparing Edge Preservation - Image 400" başlıklı görüntüye bakarak çıkarıyoruz:
Original Edges (Orijinal Kenarlar): 1663 kenar pikseli, Median Blur Edges (Medyan Bulanıklık Kenarları): 687 kenar pikseli, Gaussian Blur Edges (Gauss Bulanıklık Kenarları): 713 kenar pikseli.

• Gaussian blur detay kaybına neden oluyor mu? Evet, görüntülere göre Gaussian Blur, hem RGB hem de Gri Tonlamalı görüntülerde detay kaybına neden olmuştur. Bu sonucu, özellikle "Comparison: Median vs Gaussian Blur (kernel=5) - Image 400" başlıklı görüntüye bakarak ve kenar tespiti sonuçlarını ("загрузка (16).png") dikkate alarak çıkarıyoruz.

# 5. Döndürme ve Ayna Çevirme (Flipping)

Bu bölümde, görüntü veri artırma (data augmentation) ve simetri analizi için döndürme ve ayna çevirme işlemleri uygulanmıştır. Bu teknikler, deri lezyonu görüntülerinin makine öğrenimi modellerinde daha iyi genelleme yapabilmesi için önemlidir.

# 5.1. Rastgele Döndürme 

Görüntüler, merkez etrafında belirli açılarla saat yönünün tersine döndürülmüştür. Küçük açılı döndürmeler (0-10°), görüntü bilgisinin çoğunu korurken veri çeşitliliği sağlar.

Uygulama

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    return rotated

400 Numaralı Görüntü Analizi

Üretilen Rastgele Açılar:

— Açı 1: 3.75°

— Açı 2: 9.51°

— Açı 3: 7.32°

— Açı 4: 5.99°

— Açı 5: 1.56°

Görsel Sonuçlar:

— Tüm döndürme açılarında görüntü kalitesi korunmuş

— Kenar pikselleri BORDER_REFLECT modu ile doldurulmuş

— Lezyon yapısı ve detaylar bozulmamış

Histogram Analizi:

— Piksel yoğunluk dağılımı büyük ölçüde değişmemiş

— İstatistiksel özellikler korunmuş

— Renk dağılımı sabit kalmış

<img width="853" height="382" alt="Снимок экрана 2025-11-26 153406" src="https://github.com/user-attachments/assets/b46fbc41-52e9-46c9-ac92-6206db0604d6" />

<img width="1789" height="690" alt="загрузка (17)" src="https://github.com/user-attachments/assets/923e10b7-610b-42b1-998b-58deb3c2a770" />

<img width="1789" height="690" alt="загрузка (18)" src="https://github.com/user-attachments/assets/31d8203b-b39f-4fc5-9420-5edaddc8bef9" />

<img width="1789" height="623" alt="загрузка (19)" src="https://github.com/user-attachments/assets/a159e395-8fca-4270-a157-c87b64e5cc8f" />

Döndürme Analizi Sonuçları

— Küçük açılar (0-10°): Görüntü bilgisinin çoğunu korur

— Kenar dolgusu: BORDER_REFLECT modu kullanılmış

— Makine öğrenimi: Veri artırma için kullanışlı

— Döndürme-değişmez özellikler: Modelin öğrenmesine yardımcı olur

— Histogram: Dağılım büyük ölçüde değişmez

— Tıbbi önem: Deri lezyonları her yönde görünebilir

# 5.2. Yatay Ayna Çevirme 

Görüntüler dikey eksen etrafında ayna çevirme işlemine tabi tutulmuştur.

Uygulama

img_rgb_flipped = cv2.flip(img_rgb_original, 1)  # 1 = yatay çevirme
img_gray_flipped = cv2.flip(img_gray_original, 1)

400 Numaralı Görüntü Analizi

Görsel Sonuçlar:

— Orijinal RGB: Doğal görüntü düzeni

— Yatay Çevrilmiş RGB: Dikey eksende ayna görüntüsü

— Orijinal Grayscale: Standart gri tonlamalı

— Yatay Çevrilmiş Grayscale: Gri tonlamalı ayna görüntüsü

Histogram Analizi:

— Önemli bulgu: Histogram dağılımı TAMAMEN AYNI kalmış

— Piksel yoğunluk istatistikleri değişmemiş

— Uzamsal düzen değişmiş ancak istatistiksel özellikler korunmuş

<img width="792" height="296" alt="Снимок экрана 2025-11-26 153857" src="https://github.com/user-attachments/assets/37faf1fc-8e75-4608-986d-f5b22111f66d" />

<img width="1589" height="788" alt="загрузка (20)" src="https://github.com/user-attachments/assets/60bf8e85-d346-4942-8cf6-cdaf13c55d56" />

<img width="1589" height="741" alt="загрузка (21)" src="https://github.com/user-attachments/assets/dfa60cc3-a35a-4250-b4a6-a3ef49514294" />

Yatay Çevirme Analizi Sonuçları

— Dikey eksende ayna görüntüsü oluşturur

— Piksel yoğunluk dağılımı (histogram) AYNI kalır

— Uzamsal düzen değişir ancak istatistiksel özellikler değişmez

— Tıbbi görüntülemede veri artırma için önemli

Simetri Analizi (Symmetry Analysis)

Orijinal ve çevrilmiş görüntüler arasındaki fark haritaları hesaplanarak simetri skorları elde edilmiştir.

Simetri Hesaplama

def analyze_symmetry(original, flipped):
    diff = np.abs(original.astype(float) - flipped.astype(float))
    symmetry_score = np.mean(diff)
    return diff, symmetry_score

400 Numaralı Görüntü Simetri Sonuçları

Simetri Skorları (düşük = daha simetrik):

— RGB Simetri Skoru: 40.16

— Gri Tonlamalı Simetri Skoru: 38.27

Fark Haritası Analizi:

— Yüksek fark değerleri (ısı haritasında daha parlak): Asimetrik bölgeleri gösterir

— Deri lezyonları genellikle asimetri gösterir (tanısal özellik)

— Melanom tespiti için ABCDE kriterlerinden biri:

• A = Asimetri (Asymmetry)

• B = Kenar (Border)

• C = Renk (Color)

• D = Çap (Diameter)

• E = Gelişim (Evolution)

<img width="1489" height="972" alt="загрузка (22)" src="https://github.com/user-attachments/assets/fe9305a7-147b-4cb7-b240-b06755be3d1e" />

Simetri Gözlemleri

— Asimetrik bölgeler tanı için önemli ipuçları sağlar

— Çevirme işlemi, lezyonun her iki tarafta tutarlı özelliklere sahip olup olmadığını belirlemeye yardımcı olur

— Yüksek simetri skorları, lezyonun asimetrik doğasını doğrular

Birleşik Dönüşümler (Combined Transformations)

Döndürme ve çevirme işlemleri birleştirilerek daha karmaşık veri artırma teknikleri gösterilmiştir.

400 Numaralı Görüntü için Birleşik Dönüşüm:

— Döndürme açısı: 1.56°

— İşlem: Döndürme + Yatay Çevirme

Görsel Sonuçlar:

— Orijinal: Temel referans

— Döndürülmüş (1.56°): Hafif açısal değişim

— Çevrilmiş: Yatay ayna görüntüsü

— Döndürülmüş + Çevrilmiş: Karmaşık dönüşüm

Birleşik Dönüşümlerin Avantajları:

— Daha zengin veri çeşitliliği

— Modelin çeşitli görüntü varyasyonlarına adaptasyonu

— Gerçek dünya koşullarını daha iyi temsil etme

— Overfitting'in azaltılması

<img width="1589" height="737" alt="загрузка (23)" src="https://github.com/user-attachments/assets/535d25f7-b238-4386-972e-9504fdaacad0" />

# 6. Frekans Alanında Filtreleme (FFT)

Bu bölümde, görüntü işleme tekniklerinden Fourier Dönüşümü ve frekans alanında filtreleme yöntemleri ele alınmıştır. FFT (Fast Fourier Transform), görüntülerin frekans bileşenlerini analiz etmek ve çeşitli filtreleme işlemleri uygulamak için kullanılmıştır.

# 6.1. Fourier Dönüşümü 

Fourier Dönüşümü, bir görüntüyü uzaysal alandan frekans alanına dönüştürerek, görüntünün farklı frekans bileşenlerini analiz etmemizi sağlar.

Uygulama

def apply_fft(image):
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    magnitude_spectrum_log = np.log(magnitude_spectrum + 1)
    phase_spectrum = np.angle(fft_shift)
    return fft_shift, magnitude_spectrum, magnitude_spectrum_log, phase_spectrum

400 Numaralı Görüntü Analizi

Frekans Spektrumu Görselleştirmesi:

— Orijinal RGB: Renkli referans görüntü

— RGB → Gri: FFT için dönüştürülmüş gri tonlamalı

— Genlik Spektrumu: Frekans dağılımını gösteren log ölçekli harita

— Faz Spektrumu: Uzaysal bilgi içeren faz bileşeni

Temel Özellikler:

— Spektrum merkezi: Düşük frekanslar (yumuşak değişimler)

— Spektrum kenarları: Yüksek frekanslar (keskin detaylar, kenarlar)

— Parlak merkez: Baskın düşük frekans bileşenlerini gösterir

<img width="1790" height="859" alt="загрузка (39)" src="https://github.com/user-attachments/assets/1e36acc8-b322-4084-886e-bf6ef652cf37" />

Fourier Dönüşümü Analizi

— Genlik spektrumu frekans dağılımını gösterir

— Faz spektrumu uzaysal bilgi içerir

— Düşük frekanslar görüntünün genel yapısını belirler

— Yüksek frekanslar detayları ve kenarları temsil eder

# 6.2. Alçak Geçiren Filtre Uygulama 

Alçak geçiren filtre, yüksek frekans bileşenlerini bastırarak görüntüyü yumuşatır. Dairesel maske kullanılarak frekans alanında filtreleme yapılır.

Filtre Maskesi Oluşturma

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

<img width="722" height="277" alt="Снимок экрана 2025-11-26 155358" src="https://github.com/user-attachments/assets/6f5b8a41-0d43-4816-87f0-390f84a4b5c2" />

400 Numaralı Görüntü Filtreleme Sonuçları

RGB-Gri Görüntü için:

— Yarıçap 30: Güçlü filtreleme, çok bulanık, sadece büyük yapılar görünür

— Yarıçap 50: Orta düzey filtreleme, iyi denge, lezyon şekli korunmuş

— Yarıçap 100: Zayıf filtreleme, çoğu detay korunmuş

Orijinal Gri Görüntü için:

— Benzer sonuçlar gözlemlenmiş

— Filtreleme etkisi tutarlı

<img width="1990" height="1122" alt="загрузка (40)" src="https://github.com/user-attachments/assets/542bf0fb-d554-4be0-b779-980cae8342b3" />

<img width="1990" height="1122" alt="загрузка (41)" src="https://github.com/user-attachments/assets/fd8181e0-f7dc-4290-bc7b-edee6d6c68ff" />

Alçak Geçiren Filtre Analizi

— Küçük yarıçap = daha fazla bulanıklık = daha fazla yüksek frekans kaldırma

— Alçak geçiren filtre gürültüyü ve ince detayları kaldırır

— Yumuşatma için kullanışlı ancak kenar bilgisini kaybettirir

# 6.3. Ters Fourier (Inverse FFT) 

Frekans alanında filtrelenmiş görüntüyü tekrar uzaysal alana dönüştürme işlemidir.

Yeniden Yapılandırma Süreci:

1. Orijinal görüntü → Uzaysal alan

2. FFT Spektrumu → Frekans alanı

3. Filtrelenmiş Spektrum → Maske uygulanmış

4. Yeniden Yapılandırılmış → Ters FFT ile uzaysal alana dönüş

400 Numaralı Görüntü için Yeniden Yapılandırma (Yarıçap=50)

RGB-Gri Yol:

— Orijinal görüntü başarıyla yeniden yapılandırılmış

— Yumuşatılmış versiyon elde edilmiş

Orijinal Gri Yol:

— Benzer kalitede yeniden yapılandırma

— Filtreleme etkisi tutarlı

<img width="1789" height="831" alt="загрузка (42)" src="https://github.com/user-attachments/assets/0b509e93-5ac4-4897-b80d-d58b98ad0373" />

Ters FFT Analizi

• Frekans alanını uzaysal alana dönüştürür

• Filtrelenmiş spektrum = yumuşatılmış uzaysal görüntü

• İşlem tersinirdir (FFT ↔ IFFT)

• Frekans alanında filtreleme = uzaysal alanda konvolüsyon

# 6.4. Karşılaştırma 

<img width="752" height="276" alt="Снимок экрана 2025-11-26 155809" src="https://github.com/user-attachments/assets/bc162686-bd41-467f-8152-5006be4a4a70" />

<img width="1489" height="917" alt="загрузка (28)" src="https://github.com/user-attachments/assets/e650ed2b-f76c-4e70-886c-b6d58a3de044" />

<img width="1187" height="495" alt="загрузка (29)" src="https://github.com/user-attachments/assets/41099690-dcd5-4b16-89cd-5e75d31705dc" />

Fark Analizi:

• Orijinal görüntüler arası fark:

— Ortalama mutlak fark: 0.0000

— Maksimum mutlak fark: 0.0000

• Filtrelenmiş görüntüler arası fark:

— Ortalama mutlak fark: 0.0000

— Maksimum mutlak fark: 0.0000

Önemli Gözlemler:

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

Uygulama

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    # 1. Adım: Bulanık versiyon oluştur
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # 2. Adım: Maske hesapla (orijinal - bulanık)
    mask = cv2.subtract(image, blurred)
    
    # 3. Adım: Ağırlıklı maskeyi orijinale ekle
    sharpened = cv2.addWeighted(image, 1.0, mask, amount, 0)
    
    return sharpened, blurred, mask

<img width="617" height="235" alt="Снимок экрана 2025-11-26 164004" src="https://github.com/user-attachments/assets/ac8a43e4-07f1-4078-9049-86ed544a53e6" />

400 Numaralı Görüntü Analizi

RGB Görüntü için Sonuçlar:

— Hafif Keskinleştirme: İnce geliştirme, doğal görünüm

— Orta Keskinleştirme: İyi denge, kenarları iyi geliştiriyor

— Güçlü Keskinleştirme: Maksimum detay, gürültüyü artırabilir

Grayscale Görüntü için Sonuçlar:

— Benzer etkiler gözlemlenmiş

— Kenar geliştirme daha belirgin

<img width="1589" height="1524" alt="загрузка (43)" src="https://github.com/user-attachments/assets/25d2d787-34d9-4a64-a993-bf3240a6ff96" />

<img width="1589" height="1524" alt="загрузка (44)" src="https://github.com/user-attachments/assets/d59e9196-a76c-4e3b-9d9c-392772376276" />

İşlem Adımları:

— Orijinal görüntü - Referans

— Bulanık versiyon - Gaussian filtre uygulanmış

— Unsharp mask - Orijinal ile bulanık arasındaki fark

— Keskinleştirilmiş - Orijinal + (Amount × Mask)

<img width="1389" height="985" alt="загрузка (45)" src="https://github.com/user-attachments/assets/7aa33cd1-a511-4362-bc7d-a98d29aeed2a" />

Histogram Karşılaştırması

— Orijinal RGB Histogram: Doğal piksel dağılımı

— Keskinleştirilmiş RGB Histogram: Benzer dağılım, istatistikler korunmuş

— Orijinal Gri Histogram: Standart dağılım

— Keskinleştirilmiş Gri Histogram: Minimal değişiklik

Unsharp Masking Analizi

— Kenarları ve ince detayları geliştirir

— Deri lezyonu sınırlarını vurgulamak için önemli

— Gürültüyü artırabileceğinden dikkatli kullanılmalı

— Amount parametresi keskinleştirme gücünü kontrol eder

# 7.2. Bicubic Enterpolasyon 

Görüntüleri 2x büyütmek için çeşitli enterpolasyon yöntemleri karşılaştırılmıştır. Bicubic enterpolasyon, kalite ve hız arasında iyi denge sağladığı için tıbbi görüntülerde önerilir.

Boyut Değişikliği:

— Orijinal RGB: (459, 600, 3) → 275,400 piksel

— Büyütülmüş RGB: (900, 1200, 3) → 1,080,000 piksel

— Orijinal Gri: (459, 600) → 275,400 piksel

Enterpolasyon Yöntemleri Karşılaştırması

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

• Kalite: İyi, bilineardan daha yumuşak

• Avantaj: İyi kalite/hız dengesi

• Dezavantaj: Bilineardan yavaş

4. Lanczos

• Hız: En yavaş

• Kalite: En iyi, en keskin

• Avantaj: Maksimum kalite

• Dezavantaj: Halkalama (ringing) artefaktları olabilir

• Büyütülmüş Gri: (900, 1200) → 1,080,000 piksel

400 Numaralı Görüntü Enterpolasyon Sonuçları

RGB Görüntü için:

— Bicubic: En iyi genel performans

— Lanczos: Daha keskin ancak daha yavaş

— Bilinear: Kabul edilebilir kalite

— Nearest: Düşük kalite

Grayscale Görüntü için:

— Benzer sonuçlar gözlemlenmiş

— Bicubic önerilen yöntem olarak öne çıkmış

<img width="1990" height="788" alt="загрузка (46)" src="https://github.com/user-attachments/assets/202c6819-770e-40ba-97c7-c184bac544d9" />

<img width="1990" height="788" alt="загрузка (47)" src="https://github.com/user-attachments/assets/d6505887-6d0d-487c-8ec5-c0501f6dbb7b" />

Enterpolasyon Yöntemleri Analizi

— Nearest Neighbor: En hızlı, bloklu artefaktlar, keskin kenarları korur

— Bilinear: Hızlı, yumuşak ama hafif bulanık

— Bicubic: İyi denge, bilineardan daha yumuşak, daha yavaş

— Lanczos: En iyi kalite, en keskin, en yavaş, halkalama olabilir

— Tıbbi görüntüler için Bicubic önerilir - iyi kalite/hız dengesi

Kalite Değerlendirmesi

Kenar yoğunluğu, görüntüdeki detay seviyesini ölçmek için kullanılmıştır.

def calculate_edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / edges.size
    return edge_density, edges

<img width="757" height="312" alt="Снимок экрана 2025-11-26 164511" src="https://github.com/user-attachments/assets/c2fd9cdc-a7e2-4a6f-9a74-da11ca6a7eff" />

<img width="1189" height="978" alt="загрузка (48)" src="https://github.com/user-attachments/assets/cd2d6951-cd82-4c71-af66-90270fbaf90e" />

Kenar Koruma Analizi:

— Büyütme işlemi sonrasında kenar yoğunluğunda hafif azalma

— Bicubic enterpolasyon kenar bilgisini iyi korumuş

— Keskinleştirme işlemi kenar yoğunluğunu artırmış

İşlem Hattı Görselleştirmesi

<img width="1489" height="921" alt="загрузка (49)" src="https://github.com/user-attachments/assets/aad82360-b0c6-4748-8ff4-2ceebc74c21f" />

Tam İşlem Hattı:

1. Orijinal Görüntü - Ham veri

2. Keskinleştirilmiş - Kenar geliştirme

3. Büyütülmüş (2x) - Boyut artırma

RGB İşlem Hattı:

• Orijinal RGB → Keskinleştirilmiş RGB → Büyütülmüş RGB

• Boyut: (459, 600) → (459, 600) → (900, 1200)

GrayScale İşlem Hattı:

• Orijinal Gri → Keskinleştirilmiş Gri → Büyütülmüş Gri

• Boyut: (459, 600) → (459, 600) → (900, 1200)

<img width="736" height="376" alt="Снимок экрана 2025-11-26 164723" src="https://github.com/user-attachments/assets/b5d2c6bc-a710-42bc-925e-53a98b2fec93" />





