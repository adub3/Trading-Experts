import pandas as pd

path = "simulation/archive/sp500_stocks.csv"
symbol = "AXP"
start = "2018-01-01"   # try None first to confirm range

df = pd.read_csv(path)
print(f"[dbg] total rows: {len(df)} | cols: {list(df.columns)}")

# normalize symbol column
sym_col = "Symbol" if "Symbol" in df.columns else "Ticker"
df[sym_col] = df[sym_col].astype(str).str.upper().str.strip()
target = symbol.upper().strip()

before = len(df)
df = df[df[sym_col] == target]
print(f"[dbg] symbol '{target}': {before} -> {len(df)} rows")

# force date parse (your samples are YYYY-MM-DD)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
print("[dbg] Date NaT count:", df["Date"].isna().sum())

# numeric coercion + NaN check
for c in ["Open","High","Low","Close","Adj Close","Volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"[dbg] {c} NaN count:", df[c].isna().sum())

pre_drop = len(df)
df = df.dropna(subset=["Date","Close"]).sort_values("Date")
print(f"[dbg] dropna(Date,Close): {pre_drop} -> {len(df)} rows")
print("[dbg] range:", df["Date"].min(), "→", df["Date"].max())

# (only after confirming range) apply start filter
if start:
    before = len(df)
    df = df[df["Date"] >= pd.to_datetime(start)]
    print(f"[dbg] start >= {start}: {before} -> {len(df)} rows")

# final OHLCV
out = df.set_index("Date")[["Open","High","Low","Close","Adj Close","Volume"]]
print(f"[dbg] final rows: {len(out)} | final range: {out.index.min()} → {out.index.max()}")
