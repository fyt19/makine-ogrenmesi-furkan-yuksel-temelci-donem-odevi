# makine-ogrenmesi-furkan-yuksel-temelci-donem-odevi

Ph-negative MPN (Polycythemia Vera, Essential Thrombocythemia, Myelofibrosis) histopatoloji sınıflandırması — CNN, meta-sezgisel optimizasyon (PSO, GWO, Hybrid) ve Grad-CAM XAI.

## Proje yapısı

```
├── makaleler/    # Literatür PDF'leri
├── code/         # Kaynak kod, notebook ve veri seti
├── çıktılar/     # Model ağırlıkları, grafikler, metrikler
└── sunum/        # Sunum dosyası
```

## Kurulum ve çalıştırma

```bash
cd code
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Grad-CAM yalnızca (eğitim olmadan):

```bash
cd code
python main.py --gradcam-only --weights-slug baseline
```

## Notlar

- Eğitim veri seti `code/dataset/` altındadır (~2 GB). GitHub boyut sınırı nedeniyle repoda yer almayabilir; yerel kopyanızı bu klasöre koyun.
- Model ağırlıkları ve tüm grafikler `çıktılar/` klasöründedir.
