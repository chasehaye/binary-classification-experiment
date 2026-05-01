import yfinance as yf
import os

data_dir = "data_storage"
os.makedirs(data_dir, exist_ok=True)

print("************************************************")

default_ticker = "^GSPC"

ticker_input = input(
    f"Enter ticker symbol (default: {default_ticker}): "
).strip().upper()

ticker_symbol = ticker_input if ticker_input else default_ticker

print("************************************************")

default_period = "12y"

period_input = input(
    "Enter data period (1d, 5d, 1mo, 6mo, 1y, 5y, 10y, 12y, max) "
    f"[default: {default_period}]: "
).strip().lower()

period = period_input if period_input else default_period

print("************************************************")

default_interval = "1d"

interval_input = input(
    "Enter interval (1d, 1h, 5m, etc.) "
    f"[default: {default_interval}]: "
).strip().lower()

interval = interval_input if interval_input else default_interval

print("************************************************")

default_filename = f"{ticker_symbol.replace('^', '')}_raw_data.csv"

file_prompt = input(
    f"Enter filename to save (default: {default_filename}): "
).strip()

file_name = file_prompt if file_prompt else default_filename

if not file_name.lower().endswith(".csv"):
    file_name += ".csv"

output_path = os.path.join(data_dir, file_name)

print("************************************************")
print(f"Downloading {ticker_symbol} | Period: {period} | Interval: {interval}")
print("************************************************")

try:
    ticker = yf.Ticker(ticker_symbol)

    df = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=True
    )

    if df.empty:
        print(f"No data found for {ticker_symbol}")
    else:
        df.to_csv(output_path)
        print(f"Success! Saved to: {output_path}")
        print(f"Rows downloaded: {len(df)}")

except Exception as e:
    print(f"Error occurred: {e}")

print("************************************************")