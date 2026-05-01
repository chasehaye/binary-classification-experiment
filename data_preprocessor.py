import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_and_split(file_path, split_ratio=0.8):
    df = pd.read_csv(file_path, index_col=0)
    
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df.drop('Target', axis=1)
    y_train = train_df['Target']
    
    X_test = test_df.drop('Target', axis=1)
    y_test = test_df['Target']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test