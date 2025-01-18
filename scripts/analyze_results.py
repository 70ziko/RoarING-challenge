import pandas as pd

def analyze_anomaly_results(results_file: str):
    """Analyze anomaly detection results and provide comprehensive summary."""
    
    # Read results
    df = pd.read_csv(results_file)
    
    # Convert timestamp for time-based analysis
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], 
                                   format='%d/%b/%Y %H:%M:%S')
    
    # Basic statistics
    print("=== Anomaly Detection Summary ===")
    print(f"Total logs analyzed: {len(df):,}")
    total_anomalies = df['is_anomaly'].sum()
    print(f"Total anomalies detected: {total_anomalies:,} ({total_anomalies/len(df):.1%} of logs)")
    print(f"Average anomaly score: {df['anomaly_score'].mean():.4f}")
    print(f"Anomaly score threshold: {df[df['is_anomaly']]['anomaly_score'].min():.4f}")
    
    # Time-based analysis
    print("\n=== Temporal Distribution ===")
    hourly_anomalies = df[df['is_anomaly']].groupby(df['timestamp'].dt.hour)['is_anomaly'].count()
    peak_hour = hourly_anomalies.idxmax()
    print(f"Peak anomaly hour: {peak_hour:02d}:00 ({hourly_anomalies[peak_hour]} anomalies)")
    
    # IP-based analysis
    print("\n=== IP Address Analysis ===")
    suspicious_ips = df[df['is_anomaly']].groupby('ip')['is_anomaly'].count().sort_values(ascending=False)
    print("Top 5 IPs with most anomalies:")
    for ip, count in suspicious_ips.head().items():
        print(f"  {ip}: {count} anomalies")
    
    # Path analysis
    print("\n=== Endpoint Analysis ===")
    path_anomalies = df[df['is_anomaly']].groupby('path')['is_anomaly'].count().sort_values(ascending=False)
    print("Most attacked endpoints:")
    for path, count in path_anomalies.head().items():
        print(f"  {path}: {count} anomalies")
    
    # Sample anomalies
    print("\n=== Sample Anomalies ===")
    print("Top 3 anomalies by score:")
    top_anomalies = df[df['is_anomaly']].nlargest(3, 'anomaly_score')
    for _, row in top_anomalies.iterrows():
        print("-" * 80)
        print(f"Time: {row['date']} {row['time']}")
        print(f"IP: {row['ip']}")
        print(f"Request: {row['method']} {row['path']}")
        print(f"Status: {row['status']}")
        if pd.notna(row['payload']):
            print(f"Payload: {row['payload']}")
        print(f"Anomaly Score: {row['anomaly_score']:.4f}")
    
    # Payload analysis
    print("\n=== Payload Analysis ===")
    anomalous_payloads = df[df['is_anomaly'] & df['payload'].notna()]
    if len(anomalous_payloads) > 0:
        print(f"Number of anomalies with payloads: {len(anomalous_payloads)}")
        print("\nSample suspicious payloads:")
        for _, row in anomalous_payloads.head(3).iterrows():
            print(f"- {row['payload']}")
    else:
        print("No anomalies with payloads detected")
    
    # Status code distribution
    print("\n=== Status Code Distribution in Anomalies ===")
    status_dist = df[df['is_anomaly']]['status'].value_counts()
    for status, count in status_dist.items():
        print(f"  Status {status}: {count} occurrences ({count/total_anomalies:.1%})")

if __name__ == "__main__":
    analyze_anomaly_results('RNN/anomaly_detection_results.csv')