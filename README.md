# Bank Marketing – MLP ile Sınıflandırma: Dört Yöntem, Nested CV ve Bağımsız Test

**Hazırlayanlar:** Süleyman Balsak (254312022), Seda Erten (254312008), İlkin Urcan (244329078), Aslıhan Semiz Varlı (254329024)
**Ders:** Yapay Sinir Ağları – Yüksek Lisans Ara Sınav Projesi · **Ders sorumlusu:** Gökalp Tulum
**Temel seed:** `14048` (= 12022 + 2026; kaynak: Süleyman Balsak)

## Veri
UCI Bank Marketing, yalnızca `bank-additional.csv` (4.119 gözlem). Moro, Rita ve Cortez (2014), https://doi.org/10.24432/C5K306.
Ham veri repoya eklenmez; `grup01_eda.ipynb` resmî bağlantıdan indirir. `duration` tüm modellerden çıkarılmıştır.

## Çalıştırma (Google Colab)
Her defter ortak kodu (`src_grup/`) ve proje ayarlarını kendi içinde taşır; ayrıca dosya yüklemek gerekmez. Defteri Colab'da açıp **Tümünü çalıştır** demek yeterlidir. Çıktılar Drive'da `MyDrive/YSA_Grup_Projesi/` altına yazılır. `bank-additional.csv` önce Drive'da aranır, bulunamazsa resmî UCI bağlantısından indirilir.
1. `grup01_eda.ipynb` – seed'ler, indirme, kopya grupları, %80/%20 bölünme, development-only EDA, hipotezler
2. `grup02_nested_cv.ipynb` – her üyenin yöntemi (A–D) için 3×3 nested CV ve tam-dev seçimi; birincil yöntem kuralı
3. `grup03_final_test.ipynb` – Bölüm A: 4 yöntem × 3 seed, kilitli tahmin → *ayrı commit* → Bölüm B: test, GA, şekiller, rapor taslağı

Ortak kod: `src_grup/ysa_utils.py`, yöntemler: `src_grup/group_methods.py`.

## Rapor
Teknik rapor: [`report.pdf`](report.pdf) (kaynak: `report.md`).

## Dört yöntem
| Yöntem | Üye | Ad | Nasıl | Nihai aday | Eşik |
|---|---|---|---|---|---|
| A | Süleyman Balsak | Standart (yönerge akışı) | Adaylar yönergede tanımlandığı gibi eğitilir: ağırlıksız BCE, `unknown` ayrı kategori, M1–M6'da kırpma yok, `pdays = 999` → `prev_contacted` göstergesi + 0. Karar eşiği 0.50. | M5 | 0.5000 |
| B | Seda Erten | Sınıf ağırlıklı kayıp | Her eğitimde eğitim kaybı `BCEWithLogitsLoss(pos_weight = neg/pos)` olur. Oran, o eğitimin fit alt kümesinden hesaplanır (kat-içi, sızıntısız). Early stopping ve scheduler ağırlıksız BCE'yi izler. Eşik 0.50. | M4 | 0.5000 |
| C | İlkin Urcan | İç CV'de seçilen karar eşiği | Model eğitimi ve aday seçimi Yöntem A ile aynıdır (aynı ağırlıklar). Fark karar kuralındadır: seçilen adayın iç CV out-of-fold tahminlerinde F1'i en yüksek yapan eşik seçilir ve dış değerlendirme ile testte bu eşik kullanılır. Eşik hiçbir zaman değerlendirme/test verisiyle seçilmez. | M5 | 0.1540 |
| D | Aslıhan Semiz Varlı | Alternatif ön işleme | Tüm adaylarda `unknown` kategorileri kat-içi en sık sınıfla doldurulur, sayısal değişkenler eğitim katında öğrenilen P1–P99 sınırlarıyla kırpılır, `pdays` ham kodlamasıyla (999 dahil) kullanılır ve `prev_contacted` göstergesi eklenmez. Diğer her şey A ile aynıdır. Eşik 0.50. | R6 | 0.5000 |

## Temel sonuçlar
- Birincil yöntem (önceden tanımlı kural): **A – Standart (yönerge akışı)**, aday M5
- Bağımsız test (seed 14058): AP = 0.4478 [0.3404, 0.5503],
  ROC-AUC = 0.7471, F1 = 0.4032
- Tüm yöntemler: `results_grup/methods_three_level_comparison.csv`, `results_grup/test_metrics.csv`
- Kilitli tahmin commit'i: `265cc0e` · SHA-256 `e9ae93e39d7f9124…`

## Sistem
Google Colab · Linux-6.6.159+-x86_64-with-glibc2.39 · Python 3.13.18 · PyTorch 2.14.1+cpu · CPU, tek thread

## Deney geçmişi
Grup daha önce farklı bir kod tabanıyla ve farklı bir seed'le (34104) aynı veri seti üzerinde bir deneme yapmıştır. Bu sürüm yeni seed (14048) ile yeni bir dev/test bölünmesi kullanır ve tüm kararlar yalnızca development verisiyle alınmıştır. Ancak veri seti aynı olduğundan yeni test kümesindeki satırların bir kısmı önceki denemede test olarak değerlendirilmiş olabilir; bu nedenle test sonuçları tam anlamıyla kör kabul edilmemelidir.
