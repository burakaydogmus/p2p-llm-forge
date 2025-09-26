# P2P-LLM-Forge

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

P2P-LLM-Forge, PyTorch DistributedDataParallel (DDP) kullanarak peer-to-peer küme ortamında nedensel dil modellerinin (causal language models) veri-paralel eğitimini gerçekleştiren bir MVP (Minimum Viable Product) projesidir.

Bu proje, modern dağıtılmış makine öğrenmesi uygulamalarında karşılaşılan zorlukları ele alır ve araştırmacılar ile geliştiricilerin çok düğümlü eğitim ortamlarını test etmelerine olanak sağlar.

## ✨ Özellikler

- **Merkezi Konfigürasyon**: Hiperparametreler ve dağıtılmış çalışma zamanı ayarları için merkezi konfigürasyon sistemi
- **Otomatik Süreç Grubu Yönetimi**: PyTorch DDP için otomatik süreç grubu kurulumu ve kapatılması
- **Akıllı Veri İşleme**: Metin korpusları için tokenization ve paylaşılan depolama üzerinde shard edilmiş veri yükleme
- **Gelişmiş Eğitim Döngüsü**: Gradyan senkronizasyonu, isteğe bağlı gradyan kırpma ve rank 0'dan checkpoint'leme
- **Esnek Dağıtım**: Tek makine ve çok düğümlü dağıtılmış eğitim desteği
- **CPU/GPU Uyumluluğu**: Hem CPU hem GPU ortamlarında çalışabilme

## 📋 Gereksinimler

- **Python**: 3.10 veya üzeri
- **PyTorch**: 2.8+
- **CUDA**: GPU eğitimi için (isteğe bağlı)
- **uv**: Modern Python paket yöneticisi

## 🚀 Kurulum

### 1. Depoyu Klonlayın
```bash
git clone <repository-url>
cd p2p-llm-forge
```

### 2. Bağımlılıkları Yükleyin
```bash
uv sync
```

Bu komut tüm gerekli bağımlılıkları (PyTorch, Transformers, tqdm) yükleyecektir.

## 📖 Kullanım

### Tek Makine Testi

Proje ile birlikte gelen örnek veri seti ile hızlı bir test çalıştırmak için:

```bash
uv run python tests/smoke_single_node.py
```

Bu komut:
- Küçük `sshleifer/tiny-gpt2` modelini indirir
- `data/sample_corpus.txt` dosyasındaki örnek veriyi kullanır
- CPU dostu `gloo` backend ile 1 epoch eğitim gerçekleştirir
- Sonuçları `artifacts/smoke/` klasörüne kaydeder

### Dağıtılmış Eğitim

Çok düğümlü eğitim için her düğümde aşağıdaki komutu çalıştırın:

```bash
# Temel kullanım
uv run p2p-llm-forge \
  --model-name gpt2 \
  --dataset-path /path/to/dataset.txt \
  --output-dir ./artifacts \
  --epochs 1 \
  --batch-size 2 \
  --sequence-length 128

# torchrun ile çok düğümlü eğitim
torchrun --nnodes=2 --nproc_per_node=1 \
  --master_addr=node1 --master_port=29500 \
  uv run p2p-llm-forge \
  --model-name gpt2 \
  --dataset-path /path/to/dataset.txt \
  --output-dir ./artifacts \
  --epochs 10 \
  --batch-size 4 \
  --sequence-length 512 \
  --backend nccl
```

### Komut Satırı Seçenekleri

Tüm kullanılabilir seçenekleri görmek için:

```bash
uv run p2p-llm-forge --help
```

**Önemli Parametreler:**
- `--model-name`: HuggingFace model tanımlayıcısı
- `--dataset-path`: Eğitim veri seti yolu
- `--output-dir`: Checkpoint'lerin kaydedileceği dizin
- `--epochs`: Eğitim epoch sayısı
- `--batch-size`: Küresel batch boyutu
- `--sequence-length`: Token başına maksimum sekans uzunluğu
- `--backend`: Dağıtılmış backend (`nccl` veya `gloo`)
- `--master-addr/--master-port`: Ana düğüm bağlantı bilgileri

## 🏗️ Proje Yapısı

```
p2p-llm-forge/
├── src/
│   └── p2p_llm_forge/
│       ├── __init__.py          # Paket başlatma
│       ├── __main__.py          # CLI giriş noktası
│       ├── config.py            # Konfigürasyon yönetimi
│       ├── data.py              # Veri yükleme ve işleme
│       ├── distributed.py       # Dağıtılmış eğitim yönetimi
│       └── trainer.py           # Eğitim döngüsü
├── data/
│   └── sample_corpus.txt        # Örnek veri seti
├── tests/
│   └── smoke_single_node.py     # Tek makine test senaryosu
├── pyproject.toml               # Proje konfigürasyonu
└── README.md                    # Bu dosya
```

### Ana Modüller

- **`config.py`**: Eğitim hiperparametreleri ve dağıtılmış ayarlar için merkezi konfigürasyon
- **`data.py`**: Tokenization ve shard edilmiş veri yükleme işlemleri
- **`distributed.py`**: PyTorch DDP süreç grubu yönetimi
- **`trainer.py`**: Eğitim döngüsü, kayıp hesaplama ve optimizasyon

## ⚙️ Konfigürasyon

Proje aşağıdaki konfigürasyon seçeneklerini destekler:

### Dağıtılmış Eğitim
- **Backend**: `nccl` (GPU) veya `gloo` (CPU)
- **Rank/World Size**: Süreç sıralaması ve toplam süreç sayısı
- **Master Node**: Ana koordinasyon düğümü bilgileri

### Model ve Veri
- **Model**: Herhangi bir HuggingFace causal LM modeli
- **Veri**: Metin dosyası formatında veri setleri
- **Sequence Length**: Maksimum token uzunluğu

### Eğitim Parametreleri
- **Batch Size**: Küresel batch boyutu (tüm süreçlerde toplam)
- **Learning Rate**: Optimizasyon öğrenme oranı
- **Epochs**: Eğitim dönemi sayısı
- **Gradient Clipping**: Gradyan kırpma eşiği

### Geliştirme

```bash
# Geliştirme ortamını kurun
uv sync --editable

# Testleri çalıştırın
uv run python tests/smoke_single_node.py

# Kod kalitesi kontrolü
uv run ruff check .
uv run ruff format .
```

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

## 📞 İletişim

Sorularınız ve geri bildirimleriniz için issue açmaktan çekinmeyin!

---
