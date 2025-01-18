import torch
import pandas as pd
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader

from train import LogAnomalyDetector, LogPreprocessor, LogDataset

class AnomalyDetectorInference:
    """Handles inference for log anomaly detection."""
    
    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        device: str = None,
        threshold_percentile: float = 95
    ):
        if device is None:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.preprocessor = LogPreprocessor.load(preprocessor_path)
        
        self.model = LogAnomalyDetector(
            input_dim=self.preprocessor.feature_dim,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    def calculate_anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores for input features."""
        dataset = LogDataset(features, self.preprocessor.sequence_length)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        reconstruction_errors = []
        
        with torch.no_grad():
            for sequences, targets in dataloader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                reconstructions = self.model(sequences)
                errors = torch.mean((reconstructions - targets) ** 2, dim=1)
                reconstruction_errors.extend(errors.cpu().numpy())
        
        return np.array(reconstruction_errors)
    
    def set_threshold(self, anomaly_scores: np.ndarray):
        """Set anomaly threshold based on percentile of scores."""
        self.threshold = np.percentile(anomaly_scores, self.threshold_percentile)
    
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict anomalies in input data."""
        features = self.preprocessor.preprocess(df, fit=False)
        
        anomaly_scores = self.calculate_anomaly_scores(features)
        
        # If threshold not set, set it based on current data
        if self.threshold is None:
            self.set_threshold(anomaly_scores)
        
        # Detect anomalies
        anomalies = anomaly_scores > self.threshold
        
        # Add padding for sequences that couldn't be scored
        padding_length = self.preprocessor.sequence_length
        full_anomaly_scores = np.pad(anomaly_scores, (padding_length, 0), 'constant', constant_values=np.nan)
        full_anomalies = np.pad(anomalies, (padding_length, 0), 'constant', constant_values=False)
        
        return {
            'anomaly_scores': full_anomaly_scores,
            'anomalies': full_anomalies,
            'threshold': self.threshold
        }

def main():
    model_path = 'weights/best_model.pth'
    preprocessor_path = 'weights/preprocessor.pkl'
    test_logs_path = '../data/splits/test_logs.csv'
    
    column_names = ['date', 'time', 'ip', 'method', 'path', 'protocol', 
                   'status', 'referrer', 'user_agent', 'payload']
    new_logs = pd.read_csv(test_logs_path, header=None, names=column_names)
    
    detector = AnomalyDetectorInference(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        threshold_percentile=95
    )
    
    results = detector.predict(new_logs)
    
    
    # Print summary
    total_anomalies = results['anomalies'].sum()
    print(f"\nSummary:")
    print("-" * 50)
    print(f"Detected {total_anomalies} anomalies in {len(new_logs)} logs")
    print(f"Anomaly threshold: {results['threshold']:.4f}")
    
    # Create output DataFrame with results
    output_df = new_logs.copy()
    output_df['anomaly_score'] = results['anomaly_scores']
    output_df['is_anomaly'] = results['anomalies']
    
    # Save results
    output_df.to_csv('anomaly_detection_results.csv', index=False)
    
if __name__ == '__main__':
    main()