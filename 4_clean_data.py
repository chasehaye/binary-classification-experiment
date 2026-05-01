import pandas as pd
import os


data_dir = "data_storage"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print("************************************************")


default_input = "GSPC_engineered_features.csv"
input_prompt = input(f"Enter the engineered filename to clean (default: {default_input}): ").strip()
input_name = input_prompt if input_prompt else default_input

if not input_name.lower().endswith('.csv'):
    input_name += '.csv'

input_path = os.path.join(data_dir, input_name)

try:
    df = pd.read_csv(input_path, index_col=0)
    print(f"Loaded: {input_path}")
except FileNotFoundError:
    print(f"Error: File '{input_path}' not found.")
    exit()

print("************************************************")


trend_features = [
    "MA_20",
    "MA_50",
    "Price_to_MA20_Ratio",
    "Price_to_MA50_Ratio",
    "trend_strength"
]

return_features = [
    "return_1d",
    "log_return",
    "return_5d",
    "return_10d"
]

momentum_features = [
    "momentum_5",
    "momentum_10",
    "momentum_20"
]

volatility_features = [
    "vol_5",
    "vol_20",
    "vol_ratio",
    "Daily_Range_Pct"
]

price_action_features = [
    "body_size",
    "gap",
    "acceleration"
]

volume_features = [
    "Vol_Ratio",
    "volume_z"
]


calendar_features = [
    col for col in [
        "Day_0", "Day_1", "Day_2", "Day_3", "Day_4",
    ]
]

target = ["Target"]

final_columns = (trend_features + return_features + momentum_features + volatility_features + price_action_features + volume_features + calendar_features + target)

final_df = df[final_columns].copy()

initial_count = len(final_df)
final_df.dropna(inplace=True)
dropped_count = initial_count - len(final_df)


default_output = "GSPC_clean_data.csv"
output_prompt = input(f"Enter filename to SAVE cleaned data (default: {default_output}): ").strip()
output_name = output_prompt if output_prompt else default_output

if not output_name.lower().endswith('.csv'):
    output_name += '.csv'
    print("Note: .csv extension was added.")


output_path = os.path.join(data_dir, output_name)
final_df.to_csv(output_path)

print("************************************************")
print(f"--- Cleaning Complete ---")
print(f"Rows processed: {initial_count}")
print(f"Rows dropped (due to NaNs): {dropped_count}")
print(f"Final feature count: {len(final_columns) - 1} (plus target)")
print(f"Saved to: {output_path}")
print("************************************************")