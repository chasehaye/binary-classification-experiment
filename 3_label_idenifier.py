import pandas as pd
import os

data_dir = "data_storage"

print("************************************************")
default_input = "GSPC_engineered_features.csv"
input_prompt = input(
    f"Enter the engineered filename to verify (default: {default_input}): "
).strip()

file_name = input_prompt if input_prompt else default_input

if not file_name.lower().endswith(".csv"):
    file_name += ".csv"

input_path = os.path.join(data_dir, file_name)

try:
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f"Successfully loaded: {input_path}")
except FileNotFoundError:
    print(f"Error: file not found -> {input_path}")
    exit()

print("************************************************")

print("\n--- Dataset Summary ---")
print("************************************************")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print("************************************************")

print("\n--- Target Distribution ---")
print("************************************************")

direction_labels = {
    0: "Down",
    1: "Up"
}

df["Direction"] = df["Target"].map(direction_labels)

for k, v in direction_labels.items():
    count = (df["Target"] == k).sum()
    pct = (count / len(df)) * 100
    print(f"{v:5} | {count:6} ({pct:.2f}%)")

print("************************************************")
print("\n--- Sample Rows ---")
print(df[["Target", "Direction"]].head())
print("************************************************")

print("\n--- Feature Verification ---")

expected_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_20", "MA_50", "Price_to_MA20_Ratio", 
    "Price_to_MA50_Ratio", "trend_strength",
    "return_1d", "log_return", "return_5d", 
    "return_10d", "momentum_5", "momentum_10", 
    "momentum_20", "vol_5", "vol_20", "vol_ratio",
    "Daily_Range_Pct", "body_size", "gap",
    "Vol_Ratio", "volume_z", "acceleration",
    "Target"
]

missing = [c for c in expected_cols if c not in df.columns]

print("************************************************")

if len(missing) == 0:
    print("All engineered features are present.")
else:
    print("Missing features:")
    for m in missing:
        print(" -", m)

print("************************************************")
print("\n--- Feature Groups ---")
print("************************************************")

feature_groups = {
    "Trend": ["MA_20", "MA_50", "Price_to_MA20_Ratio", "Price_to_MA50_Ratio", "trend_strength"],
    "Returns": ["return_1d", "log_return", "return_5d", "return_10d"],
    "Momentum": ["momentum_5", "momentum_10", "momentum_20"],
    "Volatility": ["vol_5", "vol_20", "vol_ratio"],
    "Price Action": ["Daily_Range_Pct", "body_size", "gap"],
    "Volume": ["Vol_Ratio", "volume_z"],
    "Acceleration": ["acceleration"]
}

for group, cols in feature_groups.items():
    present = [c for c in cols if c in df.columns]
    print(f"{group}:")
    print(" -", present)

print("************************************************")