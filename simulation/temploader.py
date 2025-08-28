# investing_bot/data/smp_loader.py (or your temploader.py)
import os, re
import pandas as pd

REQ_PRICE_COLS = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
SYMBOL_CANDS = ["Symbol", "Ticker", "symbol", "ticker", "SYMBOL", "TICKER"]

def _read_csv_flexible(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python", sep=None)

def _find_symbol_col(df: pd.DataFrame) -> str | None:
    for c in SYMBOL_CANDS:
        if c in df.columns:
            return c
    return None

def _try_parse_dates(series: pd.Series, date_format: str | None):
    s = series.astype(str)
    raw_preview = s.head(6).tolist()
    print(f"[loader] raw Date samples: {raw_preview}")

    # 1) If user supplied a format, try it first
    if date_format:
        d = pd.to_datetime(s, format=date_format, errors="coerce")
        print(f"[loader] using format={date_format} -> NaT count: {d.isna().sum()} / {len(d)}")
        return d

    # 2) Heuristic: if looks like M-D-YY or M/D/YY, use %m-%d-%y
    if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2}$", s.iloc[0].strip()):
        # Two-digit years: pandas maps 00–68 → 2000–2068, 69–99 → 1969–1999
        d = pd.to_datetime(s, format="%m-%d-%y", errors="coerce")
        if d.isna().mean() > 0.5:
            d = pd.to_datetime(s, format="%m/%d/%y", errors="coerce")
        print(f"[loader] heuristic %%m-%%d-%%y/%%m/%%d/%%y -> NaT count: {d.isna().sum()} / {len(d)}")
        return d

    # 3) Try a cascade of common formats
    candidates = [
        None,               # auto / infer
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%m/%d/%y",
        "%m-%d-%y",
    ]
    best = None
    best_nan = 10**9
    for fmt in candidates:
        try:
            d = pd.to_datetime(s, format=fmt, errors="coerce") if fmt else pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
            nat = d.isna().sum()
            print(f"[loader] try format={fmt or 'infer'} -> NaT: {nat}")
            if nat < best_nan:
                best, best_nan = d, nat
            if nat == 0:
                break
        except Exception as e:
            print(f"[loader] format={fmt} failed: {e}")
    return best

def load_smp_csv(
    path: str,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    date_format: str | None = None,   # <-- NEW: force a format if you know it
) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = _read_csv_flexible(path)

    sym_col = _find_symbol_col(df)
    if sym_col is None:
        raise ValueError("No symbol column found (Symbol/Ticker).")

    df[sym_col] = df[sym_col].astype(str).str.upper().str.strip()
    target = str(symbol).upper().strip()
    before = len(df)
    df = df[df[sym_col] == target]
    if df.empty:
        raise ValueError(f"No rows for symbol '{target}' in {path}")

    # Required columns
    missing = [c for c in REQ_PRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    # Parse Date robustly
    raw_dates = df["Date"].astype(str).head(6).tolist()
    dts = _try_parse_dates(df["Date"], date_format=date_format)
    na_date_count = dts.isna().sum()
    df["Date"] = dts

    # Numeric coercion
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    pre_drop = len(df)
    df = df.dropna(subset=["Date","Close"])
    if df.empty:
        raise ValueError("All rows dropped after Date/Close parsing. Check date formats & numeric coercion.")

    df = df.sort_values("Date")

    # Date slice
    if start:
        before = len(df); df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        before = len(df); df = df[df["Date"] <= pd.to_datetime(end)]
    if df.empty:
        raise ValueError("No rows remain after start/end filtering. Widen the date range or remove filters.")

    out = df.set_index("Date")[["Open","High","Low","Close","Adj Close","Volume"]]
    return out
