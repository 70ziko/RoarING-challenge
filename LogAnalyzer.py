import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

class LogAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.features_encoded = None
        self.model = None
        
    def load_data(self):
        """Wczytanie i wstępne przetworzenie danych"""
        try:
            # Wczytanie danych
            self.df = pd.read_csv(self.file_path)
            
            # Zmiana nazw kolumn na czytelniejsze
            self.df.columns = ['date', 'time', 'ip', 'method', 'path', 'protocol', 
                              'status', 'referrer', 'user_agent', 'additional_info']
            
            # Połączenie daty i czasu
            self.df['timestamp'] = pd.to_datetime(self.df['date'] + ' ' + self.df['time'])
        except Exception as e:
            print(f"An error occurred in load_data: {e}")
        
        return self
    
    def preprocess_data(self):
        """Przygotowanie cech do analizy"""
        try:
            # Ekstrakcja cech z ścieżki URL
            self.df['path_length'] = self.df['path'].str.len()
            self.df['path_depth'] = self.df['path'].str.count('/')
            
            # Konwersja kodów statusu na liczby
            self.df['status'] = pd.to_numeric(self.df['status'])
            
            # Ekstrakcja cech czasowych
            self.df['hour'] = self.df['timestamp'].dt.hour
            self.df['minute'] = self.df['timestamp'].dt.minute
            self.df['is_weekend'] = self.df['timestamp'].dt.weekday >= 5
            
            # Zliczanie requestów per IP
            ip_counts = self.df['ip'].value_counts()
            self.df['requests_per_ip'] = self.df['ip'].map(ip_counts)
            
            # Kodowanie kategorycznych zmiennych
            le = LabelEncoder()
            self.df['method_encoded'] = le.fit_transform(self.df['method'])
            self.df['protocol_encoded'] = le.fit_transform(self.df['protocol'])
            
            # Wybór cech do modelu
            feature_columns = ['path_length', 'path_depth', 'status', 'hour', 'minute',
                             'requests_per_ip', 'method_encoded', 'protocol_encoded']
            
            # Normalizacja cech
            scaler = StandardScaler()
            self.features_encoded = scaler.fit_transform(self.df[feature_columns])
        except Exception as e:
            print(f"An error occurred in preprocess_data: {e}")
        
        return self
    
    def train_anomaly_detector(self, contamination=0.1):
        """Trenowanie modelu detekcji anomalii"""
        try:
            self.model = IsolationForest(contamination=contamination, random_state=42)
            self.model.fit(self.features_encoded)
            
            # Dodanie predykcji do dataframe
            self.df['is_anomaly'] = self.model.predict(self.features_encoded)
            self.df['is_anomaly'] = self.df['is_anomaly'].map({1: 0, -1: 1})  # 1 dla anomalii
        except Exception as e:
            print(f"An error occurred in train_anomaly_detector: {e}")
        
        return self
    
    def analyze_anomalies(self):
        """Analiza wykrytych anomalii"""
        try:
            anomalies = self.df[self.df['is_anomaly'] == 1]
            
            print(f"Wykryto {len(anomalies)} anomalii ({len(anomalies)/len(self.df)*100:.2f}%)")
            
            # Analiza anomalii według metody HTTP
            print("\nRozkład anomalii według metody HTTP:")
            print(anomalies['method'].value_counts())
            
            # Analiza anomalii według kodu statusu
            print("\nRozkład anomalii według kodu statusu:")
            print(anomalies['status'].value_counts())
            
            # Analiza czasowa anomalii
            plt.figure(figsize=(15, 5))
            plt.hist(anomalies['hour'], bins=24)
            plt.title('Rozkład anomalii w ciągu doby')
            plt.xlabel('Godzina')
            plt.ylabel('Liczba anomalii')
            plt.show()
        except Exception as e:
            print(f"An error occurred in analyze_anomalies: {e}")
        
        return anomalies
    
    def generate_security_report(self):
        """Generowanie raportu bezpieczeństwa"""
        try:
            report = {
                'total_requests': len(self.df),
                'unique_ips': self.df['ip'].nunique(),
                'total_anomalies': sum(self.df['is_anomaly']),
                'status_4xx': len(self.df[self.df['status'].between(400, 499)]),
                'status_5xx': len(self.df[self.df['status'].between(500, 599)]),
                'top_suspicious_ips': self.df[self.df['is_anomaly'] == 1]['ip'].value_counts().head(),
                'suspicious_paths': self.df[self.df['is_anomaly'] == 1]['path'].value_counts().head()
            }
        except Exception as e:
            print(f"An error occurred in generate_security_report: {e}")
        
        return report

# Przykład użycia:
"""
# Inicjalizacja i wczytanie danych
analyzer = LogAnalyzer('logs.csv')
analyzer.load_data()
analyzer.preprocess_data()

# Trenowanie modelu
analyzer.train_anomaly_detector(contamination=0.1)

# Analiza anomalii
anomalies = analyzer.analyze_anomalies()

# Generowanie raportu
report = analyzer.generate_security_report()
print("\nRaport bezpieczeństwa:")
for key, value in report.items():
    print(f"\n{key}:")
    print(value)
"""