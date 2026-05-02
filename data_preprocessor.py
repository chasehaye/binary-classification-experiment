import pandas as pd
import os 
from sklearn.preprocessing import StandardScaler


def preprocess_and_split(file_path, train_ratio=0.7, test_ratio=0.15):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file: {file_path}")
        
    df = pd.read_csv(file_path, index_col=0)
    df = df.sort_index()
    
    df = df.ffill().dropna()
    
    n = len(df)
    train_end = int(n * train_ratio)
    test_end = int(n * (train_ratio + test_ratio))

    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:test_end]


    X_train = train_df.drop('Target', axis=1)
    y_train = train_df['Target']
    X_test = test_df.drop('Target', axis=1)
    y_test = test_df['Target']


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    return X_train_scaled, X_test_scaled, y_train, y_test