import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
from typing import Tuple, Dict, List
import json

class LogPreprocessor:
    """Handles preprocessing of HTTP log data for anomaly detection."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.sequence_length = 10
        self.feature_dim = None
        
        # Define column names for raw logs
        self.columns = [
            'date', 'time', 'ip', 'method', 'path', 'protocol', 
            'status', 'referrer', 'user_agent', 'payload'
        ]
        
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from date and time columns."""
        df = df.copy()
        
        # Combine date and time into timestamp
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%b/%Y %H:%M:%S')
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df

    def extract_payload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from payload data."""
        df = df.copy()
        df['payload_length'] = df['payload'].fillna('').astype(str).str.len()
        df['has_payload'] = (df['payload'].fillna('') != '').astype(int)
        return df
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables using LabelEncoder."""
        df = df.copy()
        for col in columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].fillna('MISSING'))
            else:
                # Handle unseen categories
                df[col] = df[col].fillna('MISSING')
                unseen = ~df[col].isin(self.label_encoders[col].classes_)
                if unseen.any():
                    df.loc[unseen, col] = 'MISSING'
                df[col] = self.label_encoders[col].transform(df[col])
        return df

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Full preprocessing pipeline."""
        # If DataFrame has no column names, set them
        if all(df.columns == range(len(df.columns))):
            df.columns = self.columns
            
        # Extract features
        df = self.extract_time_features(df)
        df = self.extract_payload_features(df)
        
        # Select features for model
        categorical_cols = ['method', 'path', 'status', 'ip']
        numerical_cols = ['hour', 'minute', 'second', 'day_of_week', 
                         'payload_length', 'has_payload']
        
        # Convert status to string for categorical encoding
        df['status'] = df['status'].astype(str)
        
        # Encode categorical variables
        df = self.encode_categorical(df, categorical_cols, fit)
        
        # Combine features
        features = df[categorical_cols + numerical_cols].values
        
        # Scale numerical features
        if fit:
            self.feature_dim = features.shape[1]
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
            
        return features
    
    def save(self, path: str):
        """Save preprocessor state."""
        with open(path, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_dim': self.feature_dim,
                'sequence_length': self.sequence_length,
                'columns': self.columns
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'LogPreprocessor':
        """Load preprocessor state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls()
        preprocessor.label_encoders = state['label_encoders']
        preprocessor.scaler = state['scaler']
        preprocessor.feature_dim = state['feature_dim']
        preprocessor.sequence_length = state['sequence_length']
        preprocessor.columns = state['columns']
        return preprocessor

class LogDataset(Dataset):
    """Dataset for sequence-based log anomaly detection."""
    
    def __init__(self, features: np.ndarray, sequence_length: int):
        self.features = features
        self.sequence_length = sequence_length
        
    def __len__(self) -> int:
        return max(0, len(self.features) - self.sequence_length)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.features[idx:idx + self.sequence_length]
        target = self.features[idx + self.sequence_length]
        return torch.FloatTensor(sequence), torch.FloatTensor(target)

class LogAnomalyDetector(nn.Module):
    """RNN-based anomaly detection model."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        reconstruction = self.fc(last_hidden)
        return reconstruction

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: LogAnomalyDetector,
    device: torch.device,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    patience: int = 5
) -> Dict:
    """Train the anomaly detection model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            reconstructions = model(sequences)
            loss = criterion(reconstructions, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                reconstructions = model(sequences)
                loss = criterion(reconstructions, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'weights/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
    
    return history

def main():
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    df = pd.read_csv('../data/logs.csv', header=None)
    preprocessor = LogPreprocessor()
    features = preprocessor.preprocess(df, fit=True)
    
    # Save preprocessor
    preprocessor.save('preprocessor.pkl')
    
    # Create datasets
    X_train, X_val = train_test_split(features, test_size=0.2, shuffle=False)
    
    train_dataset = LogDataset(X_train, preprocessor.sequence_length)
    val_dataset = LogDataset(X_val, preprocessor.sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize and train model
    model = LogAnomalyDetector(
        input_dim=preprocessor.feature_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2
    )
    
    history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        epochs=50,
        learning_rate=1e-3,
        patience=5
    )
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    main()