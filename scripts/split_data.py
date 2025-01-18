import pandas as pd
import os

def split_logs_dataset(input_file='data/logs.csv', output_dir='data/splits'):
    """
    Split logs.csv into train and test sets based on dates.
    Test set contains logs from Jan 8th and Jan 14th (days with most anomalies).
    """
    train_path = os.path.join(output_dir, 'train_logs.csv')
    test_path = os.path.join(output_dir, 'test_logs.csv')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Split files already exist in {output_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_file, header=None,
                    names=['date', 'time', 'ip', 'method', 'path', 'protocol',
                          'status', 'referrer', 'user_agent', 'payload'])
    
    # Convert date strings to datetime for filtering
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], 
                                  format='%d/%b/%Y %H:%M:%S')
    
    # Dates with the most anomalies as found in explore_logs.ipynb
    test_dates = ['08/Jan/2025', '14/Jan/2025']
    
    # Split the data
    test_mask = df['date'].isin(test_dates)
    test_df = df[test_mask]
    train_df = df[~test_mask]
    
    # Remove the datetime column we added for filtering
    test_df = test_df.drop('datetime', axis=1)
    train_df = train_df.drop('datetime', axis=1)
    
    # Print split statistics
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    print(f"Train set: {len(train_df)} rows ({len(train_df)/total_rows:.1%})")
    print(f"Test set: {len(test_df)} rows ({len(test_df)/total_rows:.1%})")
    
    # Save to CSV files in original format (without headers)
    train_df.to_csv(train_path, header=False, index=False)
    test_df.to_csv(test_path, header=False, index=False)
    
    print(f"\nFiles saved:")
    print(f"Train set: {train_path}")
    print(f"Test set: {test_path}")

if __name__ == "__main__":
    split_logs_dataset()