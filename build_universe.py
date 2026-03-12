import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")


def fetch_wikipedia_table(url: str) -> pd.DataFrame:
    tables = pd.read_html(url)
    if not tables:
        return pd.DataFrame()
    return tables[0]


def normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    t = t.replace(".", "-")
    return t


def build_universe() -> pd.DataFrame:
    dfs = []

    # S&P 500
    try:
        sp500 = fetch_wikipedia_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp500 = sp500.rename(columns={"Symbol": "ticker", "Security": "name"})
        sp500 = sp500[["ticker", "name"]].copy()
        dfs.append(sp500)
    except Exception:
        pass

    # Nasdaq-100
    try:
        ndx = fetch_wikipedia_table("https://en.wikipedia.org/wiki/Nasdaq-100")
        possible_cols = ndx.columns.tolist()

        ticker_col = None
        name_col = None

        for c in possible_cols:
            cl = str(c).lower()
            if ticker_col is None and ("ticker" in cl or "symbol" in cl):
                ticker_col = c
            if name_col is None and ("company" in cl or "name" in cl):
                name_col = c

        if ticker_col is not None and name_col is not None:
            ndx = ndx.rename(columns={ticker_col: "ticker", name_col: "name"})
            ndx = ndx[["ticker", "name"]].copy()
            dfs.append(ndx)
    except Exception:
        pass

    # S&P MidCap 400
    try:
        mid = fetch_wikipedia_table("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
        mid = mid.rename(columns={"Symbol": "ticker", "Security": "name"})
        mid = mid[["ticker", "name"]].copy()
        dfs.append(mid)
    except Exception:
        pass

    # fallback minimal universe
    if not dfs:
        fallback = pd.DataFrame(
            {
                "ticker": [
                    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA",
                    "ANET", "NOW", "CRWD", "PANW", "AMD", "NFLX", "UBER", "SHOP",
                    "TTD", "MDB", "SNOW", "DDOG", "APP", "PLTR", "CFLT", "CSGS"
                ],
                "name": [
                    "Apple", "Microsoft", "NVIDIA", "Amazon", "Meta Platforms", "Alphabet",
                    "Broadcom", "Tesla", "Arista Networks", "ServiceNow", "CrowdStrike",
                    "Palo Alto Networks", "Advanced Micro Devices", "Netflix", "Uber",
                    "Shopify", "The Trade Desk", "MongoDB", "Snowflake", "Datadog",
                    "AppLovin", "Palantir", "Confluent", "CSG Systems"
                ],
            }
        )
        dfs.append(fallback)

    df = pd.concat(dfs, ignore_index=True)
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["name"] = df["name"].astype(str).str.strip()
    df = df.dropna(subset=["ticker"])
    df = df.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    return df


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = build_universe()
    df.to_csv(UNIVERSE_FILE, index=False, encoding="utf-8-sig")
    print(f"Universe saved: {UNIVERSE_FILE} | rows={len(df)}")


if __name__ == "__main__":
    main()
