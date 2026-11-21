# MLP ile Sınıflandırma: EDA, Ön İşleme, Hiperparametre Optimizasyonu ve Nested Cross-Validation
**Yapay Sinir Ağları – Yüksek Lisans Ara Sınav Projesi** · Hazırlayanlar: Süleyman Balsak (254312022), Seda Erten (254312008), İlkin Urcan (244329078), Aslıhan Semiz Varlı (254329024) · Ders sorumlusu: Gökalp Tulum

> Bu dosya `grup03_final_test.ipynb` tarafından otomatik üretilmiş **taslaktır**. Tüm sayılar `results_grup/` dosyalarından gelir. _İtalik_ yer tutucuları grup kendi cümleleriyle doldurmalıdır. Hedef uzunluk 8–12 sayfa (Bölüm 16).

## Özet ve araştırma sorusu
_Problem, veri, yaklaşım ve temel bulguyu 120–150 kelimede yazın._ Anahtar sayılar: birincil yöntem **A (Standart (yönerge akışı))**, bağımsız test AP = 0.4478 [%95 GA 0.3404–0.5503], ROC-AUC = 0.7471; Logistic Regression test AP = 0.4484; pozitif oranı = 0.1091.
**Araştırma sorusu:** Dengesiz bir ikili sınıflandırma probleminde MLP'nin performansı ön işleme, optimizer, eğitim kaybı ve karar kuralı tercihlerine ne kadar duyarlıdır ve bu tercihler doğrusal bir referansa göre fark yaratır mı?

## 1. Giriş
_MLP, hiperparametre optimizasyonu, sınıf dengesizliği ve nested CV'nin neden gerekli olduğu; çalışmanın amacı ve dört yöntemin neden seçildiği._

## 2. Veri seti ve EDA
| Küme | n | Pozitif (y=1) | Negatif (y=0) | Pozitif oranı % | Seed |
|---|---|---|---|---|---|
| Tüm veri | 4119 | 451 | 3668 | 10.9500 | 14049 |
| Development (%80) | 3303 | 362 | 2941 | 10.9600 | 14049 |
| Independent Test (%20) | 816 | 89 | 727 | 10.9100 | 14049 |

Kopya kontrolü:

| Kontrol | Sayı |
|---|---|
| Ham veride (duration dahil) tam kopya satır | 0 |
| duration çıkarıldıktan sonra kopya (özellikler + y) | 19 |
| Yalnızca özellikler bazında kopya | 20 |
| Kopya grubuna ait toplam satır | 40 |
| Etiketi çelişen (aynı X, farklı y) satır | 2 |
| Hem dev hem test'e düşen kopya grubu (grup-duyarlı bölünme sonrası) | 0 |

`duration` görüşme öncesinde bilinmediği için çıkarılmıştır. `pdays = 999` kodu `prev_contacted` göstergesine ayrılmıştır (Yöntem D'de ham bırakılır). Şekil 2a–2g: `figures_grup/fig2*.png`. _EDA bulgularını ve 01'deki hipotezleri (H1–H6) özetleyin._

## 3. Yöntem
- **Bölünme ve seed:** temel seed 14048 (Süleyman Balsak); alt seed'ler `configs_grup/seeds.json`. Tüm bölünmeler kopya grubuna duyarlıdır (Şekil 1).
- **MLP:** PyTorch, tek logit + BCEWithLogitsLoss, ReLU/tanh, en fazla 150 epoch, early stopping (patience 15, en iyi ağırlıklar), ReduceLROnPlateau (factor 0.5, patience 5, min_lr 1e-5), iç early-stop kümesi %15. Birincil modelin parametre sayısı: 6,145 (Şekil 3).
- **Kat-içi ön işleme:** median imputation → (opsiyonel) P1–P99 clipping → Standard/MinMax/Robust scaler; kategorikler için `unknown` stratejisi → OneHotEncoder(handle_unknown="ignore"). Standardization (z-skoru) ile normalization (MinMax aralık ölçekleme) farkı _açıklanmalı_.
- **Nested CV:** dış 3 kat / iç 3 kat; 6 manuel (M1–M6) + 6 otomatik (R1–R6, optimizer başına 2) aday; seçim iç CV ortalama AP ile.
- **Dört yöntem (üye iş bölümü):**

| Yöntem | Üye | Ad | Nasıl | Nihai aday | Eşik |
|---|---|---|---|---|---|
| A | Süleyman Balsak | Standart (yönerge akışı) | Adaylar yönergede tanımlandığı gibi eğitilir: ağırlıksız BCE, `unknown` ayrı kategori, M1–M6'da kırpma yok, `pdays = 999` → `prev_contacted` göstergesi + 0. Karar eşiği 0.50. | M5 | 0.5000 |
| B | Seda Erten | Sınıf ağırlıklı kayıp | Her eğitimde eğitim kaybı `BCEWithLogitsLoss(pos_weight = neg/pos)` olur. Oran, o eğitimin fit alt kümesinden hesaplanır (kat-içi, sızıntısız). Early stopping ve scheduler ağırlıksız BCE'yi izler. Eşik 0.50. | M4 | 0.5000 |
| C | İlkin Urcan | İç CV'de seçilen karar eşiği | Model eğitimi ve aday seçimi Yöntem A ile aynıdır (aynı ağırlıklar). Fark karar kuralındadır: seçilen adayın iç CV out-of-fold tahminlerinde F1'i en yüksek yapan eşik seçilir ve dış değerlendirme ile testte bu eşik kullanılır. Eşik hiçbir zaman değerlendirme/test verisiyle seçilmez. | M5 | 0.1540 |
| D | Aslıhan Semiz Varlı | Alternatif ön işleme | Tüm adaylarda `unknown` kategorileri kat-içi en sık sınıfla doldurulur, sayısal değişkenler eğitim katında öğrenilen P1–P99 sınırlarıyla kırpılır, `pdays` ham kodlamasıyla (999 dahil) kullanılır ve `prev_contacted` göstergesi eklenmez. Diğer her şey A ile aynıdır. Eşik 0.50. | R6 | 0.5000 |

- **Birincil yöntem kuralı:** tam development iç CV mean AP en yüksek (eşitlikte düşük SD, sonra A→B→C→D); dış CV/test kullanılmadı.

## 4. Referans modeller
Dummy (çoğunluk sınıfı) ve Logistic Regression (sabit ön işleme) aynı dış katlarda değerlendirilmiştir (Tablo 5).

## 5. Deneyler – dört yöntem
### 5.1 Yöntem A – Standart (yönerge akışı) (Sorumlu: Süleyman Balsak)
**Yöntem:** Adaylar yönergede tanımlandığı gibi eğitilir: ağırlıksız BCE, `unknown` ayrı kategori, M1–M6'da kırpma yok, `pdays = 999` → `prev_contacted` göstergesi + 0. Karar eşiği 0.50.
**Hipotez:** Yönergenin temel akışı, diğer yöntemlerin karşılaştırılacağı referans performansı verir.

**Tablo 2-A / 3-A.** Manuel ve otomatik adayların tam development iç CV sonuçları:

| config_id | optimizer | hidden_layers | lr | scaler | mean_AP | sd_AP | mean_best_epoch | total_time_sec |
|---|---|---|---|---|---|---|---|---|
| M5 | AdamW | [64, 32] | 0.0010 | StandardScaler | 0.4279 | 0.0283 | 8.0000 | 3.2243 |
| M4 | AdamW | [32] | 0.0010 | MinMaxScaler | 0.4273 | 0.0303 | 11.0000 | 2.5352 |
| M6 | AdamW | [32] | 0.0010 | StandardScaler | 0.4226 | 0.0372 | 13.6667 | 3.8476 |
| M1 | AdamW | [32] | 0.0010 | StandardScaler | 0.4226 | 0.0372 | 13.6667 | 2.7987 |
| M2 | Adam | [32] | 0.0010 | StandardScaler | 0.4209 | 0.0377 | 12.0000 | 2.9729 |
| R6 | AdamW | [64] | 0.0006 | MinMaxScaler | 0.4207 | 0.0279 | 7.3333 | 4.7325 |
| R2 | SGD | [32] | 0.0226 | StandardScaler | 0.4179 | 0.0430 | 4.3333 | 3.1008 |
| R3 | Adam | [32] | 0.0008 | RobustScaler | 0.4147 | 0.0442 | 11.6667 | 2.6173 |
| M3 | SGD | [32] | 0.0010 | StandardScaler | 0.4080 | 0.0657 | 106.6667 | 9.5550 |
| R5 | AdamW | [128, 64] | 0.0010 | RobustScaler | 0.4040 | 0.0326 | 11.0000 | 2.0596 |
| R1 | SGD | [128, 64] | 0.0529 | MinMaxScaler | 0.4037 | 0.0461 | 5.3333 | 3.6838 |
| R4 | Adam | [128, 64] | 0.0001 | StandardScaler | 0.3998 | 0.0405 | 18.3333 | 10.9196 |

Nihai aday: **M5** (manual_search), iç CV AP = 0.4279 ± 0.0283, eşik = 0.500. Şekil 5-A: `figures_grup/fig5A_method_A_search.png`.

**Yorum (Süleyman Balsak):** _grup02_nested_cv.ipynb'deki "Yorum – Süleyman Balsak" bölümündeki yanıtı buraya aktarın._
### 5.2 Yöntem B – Sınıf ağırlıklı kayıp (Sorumlu: Seda Erten)
**Yöntem:** Her eğitimde eğitim kaybı `BCEWithLogitsLoss(pos_weight = neg/pos)` olur. Oran, o eğitimin fit alt kümesinden hesaplanır (kat-içi, sızıntısız). Early stopping ve scheduler ağırlıksız BCE'yi izler. Eşik 0.50.
**Hipotez:** Azınlık sınıfa daha fazla ağırlık vermek 0.50 eşiğinde Sensitivity'yi belirgin artırır, sıralama metriği AP'yi az değiştirir, olasılık kalibrasyonunu (Brier) bozar.

**Tablo 2-B / 3-B.** Manuel ve otomatik adayların tam development iç CV sonuçları:

| config_id | optimizer | hidden_layers | lr | scaler | mean_AP | sd_AP | mean_best_epoch | total_time_sec |
|---|---|---|---|---|---|---|---|---|
| M4 | AdamW | [32] | 0.0010 | MinMaxScaler | 0.4249 | 0.0367 | 10.0000 | 2.4553 |
| R6 | AdamW | [64] | 0.0006 | MinMaxScaler | 0.4167 | 0.0358 | 5.3333 | 5.0679 |
| R3 | Adam | [32] | 0.0008 | RobustScaler | 0.4148 | 0.0452 | 16.6667 | 3.3698 |
| R4 | Adam | [128, 64] | 0.0001 | StandardScaler | 0.4123 | 0.0407 | 19.6667 | 11.0957 |
| M3 | SGD | [32] | 0.0010 | StandardScaler | 0.4096 | 0.0544 | 57.6667 | 6.8310 |
| M1 | AdamW | [32] | 0.0010 | StandardScaler | 0.4093 | 0.0299 | 35.0000 | 4.7961 |
| M2 | Adam | [32] | 0.0010 | StandardScaler | 0.4062 | 0.0309 | 24.3333 | 3.8228 |
| R1 | SGD | [128, 64] | 0.0529 | MinMaxScaler | 0.3994 | 0.0203 | 5.6667 | 3.8735 |
| R2 | SGD | [32] | 0.0226 | StandardScaler | 0.3933 | 0.0332 | 5.3333 | 3.2678 |
| M5 | AdamW | [64, 32] | 0.0010 | StandardScaler | 0.3923 | 0.0041 | 19.6667 | 4.2777 |
| M6 | AdamW | [32] | 0.0010 | StandardScaler | 0.3921 | 0.0115 | 51.6667 | 7.8009 |
| R5 | AdamW | [128, 64] | 0.0010 | RobustScaler | 0.3808 | 0.0342 | 24.3333 | 3.3147 |

Nihai aday: **M4** (manual_search), iç CV AP = 0.4249 ± 0.0367, eşik = 0.500. Şekil 5-B: `figures_grup/fig5B_method_B_search.png`.

**Yorum (Seda Erten):** _grup02_nested_cv.ipynb'deki "Yorum – Seda Erten" bölümündeki yanıtı buraya aktarın._
### 5.3 Yöntem C – İç CV'de seçilen karar eşiği (Sorumlu: İlkin Urcan)
**Yöntem:** Model eğitimi ve aday seçimi Yöntem A ile aynıdır (aynı ağırlıklar). Fark karar kuralındadır: seçilen adayın iç CV out-of-fold tahminlerinde F1'i en yüksek yapan eşik seçilir ve dış değerlendirme ile testte bu eşik kullanılır. Eşik hiçbir zaman değerlendirme/test verisiyle seçilmez.
**Hipotez:** Dengesiz veride 0.50 eşiği gereğinden yüksektir. İç CV ile seçilen eşik AP'yi değiştirmeden F1, Sensitivity ve Balanced Accuracy'yi artırır.

**Tablo 2-C / 3-C.** Manuel ve otomatik adayların tam development iç CV sonuçları:

| config_id | optimizer | hidden_layers | lr | scaler | mean_AP | sd_AP | mean_best_epoch | total_time_sec |
|---|---|---|---|---|---|---|---|---|
| M5 | AdamW | [64, 32] | 0.0010 | StandardScaler | 0.4279 | 0.0283 | 8.0000 | 3.2243 |
| M4 | AdamW | [32] | 0.0010 | MinMaxScaler | 0.4273 | 0.0303 | 11.0000 | 2.5352 |
| M6 | AdamW | [32] | 0.0010 | StandardScaler | 0.4226 | 0.0372 | 13.6667 | 3.8476 |
| M1 | AdamW | [32] | 0.0010 | StandardScaler | 0.4226 | 0.0372 | 13.6667 | 2.7987 |
| M2 | Adam | [32] | 0.0010 | StandardScaler | 0.4209 | 0.0377 | 12.0000 | 2.9729 |
| R6 | AdamW | [64] | 0.0006 | MinMaxScaler | 0.4207 | 0.0279 | 7.3333 | 4.7325 |
| R2 | SGD | [32] | 0.0226 | StandardScaler | 0.4179 | 0.0430 | 4.3333 | 3.1008 |
| R3 | Adam | [32] | 0.0008 | RobustScaler | 0.4147 | 0.0442 | 11.6667 | 2.6173 |
| M3 | SGD | [32] | 0.0010 | StandardScaler | 0.4080 | 0.0657 | 106.6667 | 9.5550 |
| R5 | AdamW | [128, 64] | 0.0010 | RobustScaler | 0.4040 | 0.0326 | 11.0000 | 2.0596 |
| R1 | SGD | [128, 64] | 0.0529 | MinMaxScaler | 0.4037 | 0.0461 | 5.3333 | 3.6838 |
| R4 | Adam | [128, 64] | 0.0001 | StandardScaler | 0.3998 | 0.0405 | 18.3333 | 10.9196 |

Nihai aday: **M5** (manual_search), iç CV AP = 0.4279 ± 0.0283, eşik = 0.154. Şekil 5-C: `figures_grup/fig5C_method_C_search.png`.

**Yorum (İlkin Urcan):** _grup02_nested_cv.ipynb'deki "Yorum – İlkin Urcan" bölümündeki yanıtı buraya aktarın._
### 5.4 Yöntem D – Alternatif ön işleme (Sorumlu: Aslıhan Semiz Varlı)
**Yöntem:** Tüm adaylarda `unknown` kategorileri kat-içi en sık sınıfla doldurulur, sayısal değişkenler eğitim katında öğrenilen P1–P99 sınırlarıyla kırpılır, `pdays` ham kodlamasıyla (999 dahil) kullanılır ve `prev_contacted` göstergesi eklenmez. Diğer her şey A ile aynıdır. Eşik 0.50.
**Hipotez:** `unknown`'ı bilgi taşıyan ayrı bir kategori olarak tutmak ve `pdays` özel kodunu ayrı göstermek bilgi kaybını önler. Bu yüzden D'nin AP'si A'dan eşit veya düşük olur.

**Tablo 2-D / 3-D.** Manuel ve otomatik adayların tam development iç CV sonuçları:

| config_id | optimizer | hidden_layers | lr | scaler | mean_AP | sd_AP | mean_best_epoch | total_time_sec |
|---|---|---|---|---|---|---|---|---|
| R6 | AdamW | [64] | 0.0006 | MinMaxScaler | 0.4263 | 0.0273 | 8.6667 | 4.5263 |
| R2 | SGD | [32] | 0.0226 | StandardScaler | 0.4246 | 0.0330 | 6.6667 | 3.9373 |
| M4 | AdamW | [32] | 0.0010 | MinMaxScaler | 0.4244 | 0.0225 | 14.3333 | 3.0565 |
| M3 | SGD | [32] | 0.0010 | StandardScaler | 0.4233 | 0.0413 | 123.6667 | 10.4561 |
| R5 | AdamW | [128, 64] | 0.0010 | RobustScaler | 0.4226 | 0.0545 | 12.6667 | 2.4514 |
| M5 | AdamW | [64, 32] | 0.0010 | StandardScaler | 0.4219 | 0.0335 | 8.0000 | 2.8632 |
| M6 | AdamW | [32] | 0.0010 | StandardScaler | 0.4215 | 0.0207 | 12.3333 | 2.6701 |
| M1 | AdamW | [32] | 0.0010 | StandardScaler | 0.4215 | 0.0207 | 12.3333 | 3.3333 |
| R4 | Adam | [128, 64] | 0.0001 | StandardScaler | 0.4210 | 0.0442 | 15.0000 | 10.2804 |
| M2 | Adam | [32] | 0.0010 | StandardScaler | 0.4209 | 0.0231 | 12.3333 | 2.6869 |
| R1 | SGD | [128, 64] | 0.0529 | MinMaxScaler | 0.4099 | 0.0042 | 8.3333 | 5.2662 |
| R3 | Adam | [32] | 0.0008 | RobustScaler | 0.3714 | 0.0180 | 14.6667 | 2.9026 |

Nihai aday: **R6** (random_search), iç CV AP = 0.4263 ± 0.0273, eşik = 0.500. Şekil 5-D: `figures_grup/fig5D_method_D_search.png`.

**Yorum (Aslıhan Semiz Varlı):** _grup02_nested_cv.ipynb'deki "Yorum – Aslıhan Semiz Varlı" bölümündeki yanıtı buraya aktarın._


## 6. Nested CV bulguları
**Tablo 5.** Dış CV Mean ± SD (n = 3; katlar bağımsız tekrar değildir, SD güven aralığı değildir):

| grp | AP | ROC_AUC | F1 | Sensitivity | Specificity | Balanced_Accuracy |
|---|---|---|---|---|---|---|
| MLP-A manual | 0.4279 ± 0.0558 | 0.7679 ± 0.0299 | 0.3100 ± 0.0231 | 0.2125 ± 0.0396 | 0.9813 ± 0.0139 | 0.5969 ± 0.0131 |
| MLP-A random | 0.4257 ± 0.0575 | 0.7648 ± 0.0335 | 0.3334 ± 0.0289 | 0.2387 ± 0.0473 | 0.9768 ± 0.0159 | 0.6078 ± 0.0158 |
| MLP-B manual | 0.4222 ± 0.0636 | 0.7734 ± 0.0337 | 0.4236 ± 0.0217 | 0.6228 ± 0.0380 | 0.8378 ± 0.0142 | 0.7303 ± 0.0245 |
| MLP-B random | 0.4189 ± 0.0527 | 0.7716 ± 0.0299 | 0.4393 ± 0.0526 | 0.5797 ± 0.0165 | 0.8645 ± 0.0493 | 0.7221 ± 0.0173 |
| MLP-C manual | 0.4279 ± 0.0558 | 0.7679 ± 0.0299 | 0.4658 ± 0.0324 | 0.5069 ± 0.0340 | 0.9173 ± 0.0101 | 0.7121 ± 0.0221 |
| MLP-C random | 0.4257 ± 0.0575 | 0.7648 ± 0.0335 | 0.4587 ± 0.0333 | 0.5165 ± 0.0709 | 0.9106 ± 0.0068 | 0.7136 ± 0.0340 |
| MLP-D manual | 0.4173 ± 0.0409 | 0.7674 ± 0.0233 | 0.3018 ± 0.0306 | 0.2023 ± 0.0378 | 0.9837 ± 0.0113 | 0.5930 ± 0.0147 |
| MLP-D random | 0.4167 ± 0.0720 | 0.7683 ± 0.0328 | 0.3006 ± 0.0254 | 0.2019 ± 0.0382 | 0.9833 ± 0.0128 | 0.5926 ± 0.0134 |
| Dummy | 0.1096 ± 0.0072 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| LogReg | 0.4332 ± 0.0603 | 0.7579 ± 0.0305 | 0.3239 ± 0.0269 | 0.2243 ± 0.0195 | 0.9796 ± 0.0124 | 0.6019 ± 0.0104 |

Şekil 5: `figures_grup/fig5_methods_outer_cv.png`. _Yöntemleri ve referansları adil biçimde karşılaştırın._

## 7. Bağımsız test
**Tablo 6.** Test metrikleri (birincil seed 14058):

| Model | threshold | AP | ROC_AUC | F1 | Sensitivity | Specificity | Balanced_Accuracy |
|---|---|---|---|---|---|---|---|
| A – Standart (yönerge akışı) (Süleyman Balsak) | 0.5000 | 0.4478 | 0.7471 | 0.4032 | 0.2809 | 0.9862 | 0.6336 |
| B – Sınıf ağırlıklı kayıp (Seda Erten) | 0.5000 | 0.4429 | 0.7809 | 0.4370 | 0.5843 | 0.8666 | 0.7254 |
| C – İç CV'de seçilen karar eşiği (İlkin Urcan) | 0.1538 | 0.4478 | 0.7471 | 0.4435 | 0.5730 | 0.8762 | 0.7246 |
| C – İç CV'de seçilen karar eşiği (İlkin Urcan) @0.50 | 0.5000 | 0.4478 | 0.7471 | 0.4032 | 0.2809 | 0.9862 | 0.6336 |
| D – Alternatif ön işleme (Aslıhan Semiz Varlı) | 0.5000 | 0.4319 | 0.7832 | 0.3279 | 0.2247 | 0.9821 | 0.6034 |
| Logistic Regression | 0.5000 | 0.4484 | 0.7862 | 0.3750 | 0.2697 | 0.9794 | 0.6245 |
| Dummy (çoğunluk) | 0.5000 | 0.1091 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 |

**Bootstrap %95 GA (1.000 tekrar):**

| Model | AP | F1 | ROC_AUC |
|---|---|---|---|
| A – Standart (yönerge akışı) (Süleyman Balsak) | 0.4478 [0.3404, 0.5503] | 0.4032 [0.2807, 0.5152] | 0.7471 [0.6797, 0.8098] |
| B – Sınıf ağırlıklı kayıp (Seda Erten) | 0.4429 [0.3357, 0.5532] | 0.4370 [0.3582, 0.5106] | 0.7809 [0.7210, 0.8347] |
| C – İç CV'de seçilen karar eşiği (İlkin Urcan) | 0.4478 [0.3404, 0.5503] | 0.4435 [0.3643, 0.5172] | 0.7471 [0.6797, 0.8098] |
| D – Alternatif ön işleme (Aslıhan Semiz Varlı) | 0.4319 [0.3275, 0.5446] | 0.3279 [0.2157, 0.4407] | 0.7832 [0.7244, 0.8402] |
| Logistic Regression | 0.4484 [0.3473, 0.5608] | 0.3750 [0.2655, 0.4853] | 0.7862 [0.7270, 0.8433] |

**Tablo 7.** Seed kararlılığı (Mean ± SD, n = 3):

| Yöntem | AP | ROC_AUC | F1 |
|---|---|---|---|
| A | 0.4554 ± 0.0139 | 0.7603 ± 0.0114 | 0.3968 ± 0.0065 |
| B | 0.4431 ± 0.0009 | 0.7856 ± 0.0041 | 0.4429 ± 0.0057 |
| C | 0.4554 ± 0.0139 | 0.7603 ± 0.0114 | 0.4513 ± 0.0076 |
| D | 0.4316 ± 0.0009 | 0.7817 ± 0.0033 | 0.3368 ± 0.0111 |

Şekil 6–7: `figures_grup/fig6_confusion_matrices.png`, `figures_grup/fig7_roc_pr_curves.png`. Hata analizi: `results_grup/error_analysis_groups.csv`.

## 8. Tartışma
**Üç seviyenin karşılaştırması:**

| Yöntem | Üye | Dev iç CV AP (seçim) | Dış CV AP (seçilen arama) | Dış CV F1 | Test AP | Test F1 | Test Sensitivity | Birincil |
|---|---|---|---|---|---|---|---|---|
| A | Süleyman Balsak | 0.4279 ± 0.0283 | 0.4279 ± 0.0558 | 0.3100 ± 0.0231 | 0.4478 | 0.4032 | 0.2809 | ✓ |
| B | Seda Erten | 0.4249 ± 0.0367 | 0.4222 ± 0.0636 | 0.4236 ± 0.0217 | 0.4429 | 0.4370 | 0.5843 |  |
| C | İlkin Urcan | 0.4279 ± 0.0283 | 0.4279 ± 0.0558 | 0.4658 ± 0.0324 | 0.4478 | 0.4435 | 0.5730 |  |
| D | Aslıhan Semiz Varlı | 0.4263 ± 0.0273 | 0.4167 ± 0.0720 | 0.3006 ± 0.0254 | 0.4319 | 0.3279 | 0.2247 |  |
| LogReg | - | - | 0.4332 ± 0.0603 | 0.3239 ± 0.0269 | 0.4484 | 0.3750 | 0.2697 |  |

_Her üye kendi yönteminin etkisini 3–4 cümleyle tartışır; ardından ortak değerlendirme:_ hiperparametre etkisi, optimizer farkları, sınıf ağırlığı ve eşik seçiminin pratik anlamı, ön işleme duyarlılığı, MLP–LogReg farkı.
**Sınırlamalar:** n = 3 dış kat; küçük test kümesi (816 gözlem, 89 pozitif); müşteri kimliği olmadığı için kişi-bağımsız genelleme kanıtlanamaz; kamuya açık veri nedeniyle kilitli commit kör test garantisi değildir. Grup daha önce farklı bir kod tabanıyla ve farklı bir seed'le (34104) aynı veri seti üzerinde bir deneme yapmıştır. Bu sürüm yeni seed (14048) ile yeni bir dev/test bölünmesi kullanır ve tüm kararlar yalnızca development verisiyle alınmıştır. Ancak veri seti aynı olduğundan yeni test kümesindeki satırların bir kısmı önceki denemede test olarak değerlendirilmiş olabilir; bu nedenle test sonuçları tam anlamıyla kör kabul edilmemelidir.

## 9. Sonuç
_Kanıta dayalı 4–6 cümle._

## Kaynakça
1. Moro, S., Rita, P., Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
2. scikit-learn. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html
3. PyTorch. ReduceLROnPlateau. https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
4. scikit-learn. Cross-validation, model selection and classification metrics. https://scikit-learn.org/stable/

## Ek – Üye katkıları
| Üye | Yöntem | Defter bölümü | Çıktılar |
|---|---|---|---|
| Süleyman Balsak | A – Standart (yönerge akışı) | grup02_nested_cv.ipynb, Bölüm 2 | results_grup/method_A/, Tablo 2-A…5-A, Şekil 5-A |
| Seda Erten | B – Sınıf ağırlıklı kayıp | grup02_nested_cv.ipynb, Bölüm 3 | results_grup/method_B/, Tablo 2-B…5-B, Şekil 5-B |
| İlkin Urcan | C – İç CV'de seçilen karar eşiği | grup02_nested_cv.ipynb, Bölüm 4 | results_grup/method_C/, Tablo 2-C…5-C, Şekil 5-C |
| Aslıhan Semiz Varlı | D – Alternatif ön işleme | grup02_nested_cv.ipynb, Bölüm 5 | results_grup/method_D/, Tablo 2-D…5-D, Şekil 5-D |
Ortak bölümler (EDA, test değerlendirmesi, rapor) grup tarafından birlikte yürütülmüştür.
