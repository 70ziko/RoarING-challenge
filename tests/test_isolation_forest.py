import numpy as np
from sklearn.ensemble import IsolationForest

def test_isolation_forest():
    """Testowanie IsolationForest na syntetycznym zbiorze danych"""
    try:
        print("Testing IsolationForest with synthetic data...")
        X = np.random.rand(100, 8)
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        print("Synthetic data test completed successfully.")
    except Exception as e:
        print(f"An error occurred in test_isolation_forest: {e}")

if __name__ == "__main__":
    test_isolation_forest()
