# System Detekcji Anomalii w Logach HTTP

## Przegląd Projektu

Projekt implementuje system detekcji anomalii w logach HTTP wykorzystując architekturę sieci rekurencyjnej (RNN). Głównym celem jest wykrywanie nietypowych wzorców w ruchu sieciowym, które mogą wskazywać na potencjalne zagrożenia bezpieczeństwa.

## Eksploracja Danych

Notebook `explore_logs.ipynb` zawiera szczegółową analizę danych wejściowych, w tym:
- Rozkład metod HTTP
- Analiza ścieżek URL
- Wzorce czasowe ruchu
- Statystyki kodów odpowiedzi
- Wizualizacje kluczowych metryk
- Uzasadnienie oraz przeprowadzenie podziału zbioru danych

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
- Dropout (0.2) dla regularyzacji
- Fully connected layers do rekonstrukcji wektora cech

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

### Trenowanie Modelu

```bash
cd RNN
python train.py
```

### Inferencja

```bash
python inference.py
```

## Uzasadnienie Architektury

1. **Preprocessor**:
   - Elastyczne kodowanie kategorii z obsługą nieznanych wartości
   - Ekstrakcja cech czasowych dla wykrywania anomalii w wzorcach czasowych
   - Standaryzacja dla stabilnego treningu

2. **Model LSTM**:
   - Dwuwarstwowa architektura dla lepszego modelowania złożonych wzorców
   - Dropout dla zapobiegania przeuczeniu
   - Rekonstrukcja sekwencji jako miara anomalii

3. **Detekcja Anomalii**:
   - Wykorzystanie błędu rekonstrukcji jako miary anomalii
   - Adaptacyjny próg bazujący na percentylu rozkładu błędów
   - Możliwość dostosowania czułości poprzez parametr threshold_percentile
