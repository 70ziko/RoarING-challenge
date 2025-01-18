# System Detekcji Anomalii w Logach HTTP

## Przegląd Projektu

Projekt implementuje system detekcji anomalii w logach HTTP wykorzystując architekturę sieci rekurencyjnej (RNN). Głównym celem jest wykrywanie nietypowych wzorców w ruchu sieciowym, które mogą wskazywać na potencjalne zagrożenia bezpieczeństwa.


### Struktura Projektu
```
.
├── data/
│   ├── logs.csv                    # Surowe logi
│   └── splits/                     # Podzielone dane
├── RNN/
│   ├── train.py                    # Skrypt trenujący
│   ├── inference.py                # Skrypt inferencji
│   └── weights/                    # Zapisane modele
└── scripts/
    ├── split_data.py              # Podział danych
    └── analyze_results.py         # Analiza wyników
```


## Eksploracja Danych

Notebook `explore_logs.ipynb` zawiera szczegółową analizę danych wejściowych, w tym:
- Rozkład metod HTTP
- Analiza ścieżek URL
- Wzorce czasowe ruchu
- Statystyki kodów odpowiedzi
- Wizualizacje kluczowych metryk
- Uzasadnienie oraz przeprowadzenie podziału danych

## Architektura Modelu

### Preprocessor
- Ekstrakcja cech czasowych (godzina, minuta, sekunda, dzień tygodnia)
- Kodowanie zmiennych kategorycznych (metoda HTTP, ścieżka, status, IP)
- Normalizacja cech numerycznych
- Obsługa nieznanych kategorii podczas inferencji

### Model RNN
Model wykorzystuje architekturę LSTM (Long Short-Term Memory) z następujących powodów:
1. **Kontekst sekwencyjny**: LSTM pozwala na uczenie długoterminowych zależności w sekwencjach logów
2. **Adaptacyjność**: Model automatycznie dostosowuje się do zmian w normalnym ruchu
3. **Wykrywanie anomalii**: Wykorzystuje rekonstrukcję sekwencji do identyfikacji nietypowych wzorców

Kluczowe elementy architektury:
- Warstwa LSTM (2 warstwy, 64 jednostki ukryte)
- Dropout (0.2) dla regularyzacji i zabezpieczeniu przed przeuczeniem
- Fully connected layers do rekonstrukcji wektora cech
- Błąd rekonstrukcji sekwencji jako miara anomalii

### Detekcja Anomalii:
   - Wykorzystanie błędu rekonstrukcji jako miary anomalii
   - Adaptacyjny próg bazujący na percentylu rozkładu błędów
   - Możliwość dostosowania czułości poprzez parametr threshold_percentile

## Instalacja i Uruchomienie

### Utworzenie środowiska Conda

```bash
# Utworzenie środowiska
conda create -n log-anomaly python=3.9
conda activate log-anomaly

# Instalacja PyTorch
conda install pytorch torchvision torchaudio -c pytorch

# Instalacja pozostałych zależności
pip install -r requirements.txt
```

### Trenowanie Modelu (pominąć, aby użyć zapisanych wag modelu)

```bash
cd RNN
python train.py
```

### Inferencja

W skrypcie należy zmienić ścieżkę do danych testowych, jeśli nie korzystamy z domyślnych zapisanych w `data/splits/test_logs.csv`.

```bash
python inference.py
```


