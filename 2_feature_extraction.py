import pandas as pd
import os
import numpy as np

data_dir = "data_storage"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
default_input = "GSPC_raw_data.csv"
print("************************************************")
input_prompt = input(f"Enter the RAW data filename to load (default: {default_input}): ").strip()
input_name = input_prompt if input_prompt else default_input

input_path = os.path.join(data_dir, input_name)

try:
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f"Successfully loaded {input_path}")
    print("************************************************")
except FileNotFoundError:
    print(f"Error: The file '{input_path}' was not found.")
    print("************************************************")
    exit()

print("Processing local data...")


print("Engineering trend + momentum features...")
df["MA_20"] = df["Close"].rolling(20).mean()
df["MA_50"] = df["Close"].rolling(50).mean()

df["Price_to_MA20_Ratio"] = df["Close"] / df["MA_20"]
df["Price_to_MA50_Ratio"] = df["Close"] / df["MA_50"]

df["trend_strength"] = df["MA_20"] - df["MA_50"]


print("Engineering returns...")
df["return_1d"] = df["Close"].pct_change()
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

df["return_5d"] = df["Close"].pct_change(5)
df["return_10d"] = df["Close"].pct_change(10)


print("Engineering momentum...")
df["momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
df["momentum_20"] = df["Close"] / df["Close"].shift(20) - 1


print("Engineering volatility...")
df["vol_5"] = df["return_1d"].rolling(5).std()
df["vol_20"] = df["return_1d"].rolling(20).std()

df["vol_ratio"] = df["vol_5"] / df["vol_20"]


print("Engineering price action...")
df["Daily_Range_Pct"] = (df["High"] - df["Low"]) / df["Close"]
df["body_size"] = (df["Close"] - df["Open"]) / df["Close"]

df["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)


print("Engineering volume...")
df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(5).mean()

df["volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()


print("Engineering acceleration...")
df["acceleration"] = df["return_1d"].diff()


print("Engineering time features...")
df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
df["Day_of_Week"] = df.index.dayofweek
df = pd.get_dummies(df, columns=["Day_of_Week"], prefix="Day")


print("Engineering target...")
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)


df.dropna(inplace=True)

default_output = "GSPC_engineered_features.csv"

print("************************************************")
output_prompt = input(f"Enter filename to SAVE as (default: {default_output}): ").strip()
output_name = output_prompt if output_prompt else default_output

if not output_name.lower().endswith(".csv"):
    output_name += ".csv"

output_path = os.path.join(data_dir, output_name)

df.to_csv(output_path)

print(f"File saved successfully as {output_path}")
print("************************************************")