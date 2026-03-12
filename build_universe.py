import io
import os
import json
from typing import Dict, List

import pandas as pd
import requests


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")
META_FILE = os.path.join(OUTPUT_DIR, "universe_build_meta.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,text/plain,text/html,*/*",
}

# iShares holdings CSV endpoints
# Russell 3000 / S&P 500 / S&P 400 / S&P 600
SOURCES = [
    {
        "name": "IWV_Russell_3000",
        "url": "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund",
    },
    {
        "name": "IVV_SP500",
        "url": "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund",
    },
    {
        "name": "IJH_SP400",
        "url": "https://www.ishares.com/us/products/239763/ishares-core-sp-midcap-etf/1467271812596.ajax?fileType=csv&fileName=IJH_holdings&dataType=fund",
    },
    {
        "name": "IJR_SP600",
        "url": "https://www.ishares.com/us/products/239774/ishares-core-sp-smallcap-etf/1467271812596.ajax?fileType=csv&fileName=IJR_holdings&dataType=fund",
    },
]

NAME_BLOCK_KEYWORDS = [
    "etf",
    "etn",
    "fund",
    "trust",
    "closed-end",
    "adr",
    "ads",
    "depositary",
    "warrant",
    "rights",
    "unit",
    "acquisition",
    "shell company",
    "preferred",
    "preferred stock",
    "reit",
    "real estate investment trust",
    "biotech",
    "biotechnology",
    "pharmaceutical",
    "drug manufacturers",
    "oil",
    "gas",
    "energy",
    "midstream",
    "upstream",
    "downstream",
    "coal",
    "metals",
    "mining",
    "materials",
    "steel",
    "aluminum",
    "gold",
    "silver",
    "utility",
    "utilities",
    "insurance",
    "food",
    "beverage",
    "tobacco",
]

TICKER_BLOCK_SUFFIXES = [
    "W",   # warrants often end with W on some feeds
    "R",   # rights sometimes
]

MIN_EXPECTED_UNIVERSE = 1500


def normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    t = t.replace(".", "-")
    return t


def find_header_line(text: str) -> int:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        normalized = line.replace('"', "").strip().lower()
        if normalized.startswith("ticker,") and "name" in normalized:
            return i
    raise ValueError("holdings csv header not found")


def fetch_ishares_holdings(source: Dict[str, str]) -> pd.DataFrame:
    resp = requests.get(source["url"], headers=HEADERS, timeout=60)
    resp.raise_for_status()
    text = resp.text

    start_idx = find_header_line(text)
    csv_text = "\n".join(text.splitlines()[start_idx:])
    df = pd.read_csv(io.StringIO(csv_text))

    df.columns = [str(c).strip() for c in df.columns]

    # keep only essential columns
    if "Ticker" not in df.columns or "Name" not in df.columns:
        raise ValueError(f"{source['name']} missing Ticker/Name columns")

    # iShares holdings files usually include Asset Class
    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.lower() == "equity"].copy()

    df = df[["Ticker", "Name"]].copy()
    df = df.rename(columns={"Ticker": "ticker", "Name": "name"})
    df["source"] = source["name"]
    return df


def basic_text_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ticker"] = out["ticker"].map(normalize_ticker)
    out["name"] = out["name"].astype(str).str.strip()

    out = out.dropna(subset=["ticker", "name"])
    out = out[out["ticker"].str.len() > 0]
    out = out[~out["ticker"].str.contains(r"\s", regex=True)]

    # remove obviously weird share classes / rights / warrants
    for suffix in TICKER_BLOCK_SUFFIXES:
        out = out[~out["ticker"].str.endswith(suffix)]

    name_text = out["name"].str.lower()

    for kw in NAME_BLOCK_KEYWORDS:
        out = out[~name_text.str.contains(kw, regex=False)]

    # remove duplicate lines
    out = out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return out


def build_universe() -> Dict[str, object]:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames: List[pd.DataFrame] = []
    source_counts: Dict[str, int] = {}
    source_status: Dict[str, str] = {}

    for source in SOURCES:
        try:
            df = fetch_ishares_holdings(source)
            source_counts[source["name"]] = int(len(df))
            source_status[source["name"]] = "ok"
            frames.append(df)
        except Exception as e:
            source_counts[source["name"]] = 0
            source_status[source["name"]] = f"fail: {type(e).__name__}"
            continue

    if not frames:
        raise RuntimeError("all universe sources failed")

    raw = pd.concat(frames, ignore_index=True)
    raw_count = len(raw)

    deduped = raw.drop_duplicates(subset=["ticker"]).copy()
    deduped_count = len(deduped)

    filtered = basic_text_filters(deduped)
    filtered_count = len(filtered)

    filtered = filtered.sort_values("ticker").reset_index(drop=True)
    filtered[["ticker", "name"]].to_csv(UNIVERSE_FILE, index=False, encoding="utf-8-sig")

    meta = {
        "built_at": pd.Timestamp.utcnow().isoformat(),
        "raw_count": int(raw_count),
        "deduped_count": int(deduped_count),
        "filtered_count": int(filtered_count),
        "source_counts": source_counts,
        "source_status": source_status,
        "universe_file": UNIVERSE_FILE,
        "below_expected_minimum": bool(filtered_count < MIN_EXPECTED_UNIVERSE),
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def main() -> None:
    meta = build_universe()
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
