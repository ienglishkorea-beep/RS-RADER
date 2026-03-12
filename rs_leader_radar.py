import os
import json
import runpy
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")
RESULT_FILE = os.path.join(OUTPUT_DIR, "rs_leader_radar_results.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "rs_leader_radar_summary.json")
SECTOR_CACHE_FILE = os.path.join(OUTPUT_DIR, "rs_leader_sector_cache.json")

BENCHMARK = "SPY"
DOWNLOAD_PERIOD = "2y"
MIN_HISTORY = 260

MIN_PRICE = 10.0
MIN_DOLLAR_VOL_20 = 10_000_000
MIN_3M_RETURN = 0.20

RS_LOOKBACK = 252
RS_PERCENTILE_MIN = 95.0
RS_NEW_HIGH_TOL = 0.995

READY_MAX_DISTANCE = -0.05     # 52주 고점 -5% 이내
WATCH_MIN_DISTANCE = -0.20     # -20% 이상
WATCH_MAX_DISTANCE = -0.05     # -5% 미만

REQUIRE_MA_ALIGNMENT = True

BLOCKED_KEYWORDS = [
    "energy", "oil", "gas", "midstream", "upstream", "downstream",
    "utilities", "utility",
    "insurance",
    "food", "beverage", "tobacco",
    "biotech", "biotechnology", "drug manufacturers", "pharmaceutical", "pharma",
    "metals", "mining", "coal", "materials", "basic materials", "steel", "aluminum",
    "precious metals", "gold", "silver",
]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MAX_TELEGRAM_MESSAGE_LEN = 3500
MAX_TELEGRAM_ROWS_READY = 30
MAX_TELEGRAM_ROWS_WATCH = 50


@dataclass
class RSLeaderResult:
    bucket: str
    ticker: str
    name: str
    as_of_date: str
    sector: str
    industry: str
    rs_grade: str
    rs_percentile: float
    rs_current_vs_high: float
    ret_3m: float
    close: float
    high_52w: float
    distance_to_52w_high: float
    dollar_vol_20: float
    ma50: float
    ma150: float
    ma200: float


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def format_price(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):,.2f}"


def format_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.1f}%"


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram(text: str) -> None:
    if not telegram_enabled():
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=20,
    )


def send_telegram_chunked(lines: List[str]) -> None:
    if not telegram_enabled() or not lines:
        return

    chunks: List[str] = []
    current = ""

    for line in lines:
        candidate = f"{current}\n{line}".strip() if current else line
        if len(candidate) > MAX_TELEGRAM_MESSAGE_LEN:
            if current:
                chunks.append(current)
            current = line
        else:
            current = candidate

    if current:
        chunks.append(current)

    for chunk in chunks:
        send_telegram(chunk)


def ensure_universe_file() -> None:
    if os.path.exists(UNIVERSE_FILE):
        return
    runpy.run_path(os.path.join(BASE_DIR, "build_universe.py"), run_name="__main__")


def load_universe() -> pd.DataFrame:
    ensure_universe_file()

    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(f"유니버스 파일 없음: {UNIVERSE_FILE}")

    df = pd.read_csv(UNIVERSE_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "ticker" not in df.columns:
        raise ValueError("universe.csv에 ticker 컬럼 필요")
    if "name" not in df.columns:
        df["name"] = df["ticker"]

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str)

    return df[["ticker", "name"]].drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def normalize_downloaded(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    rename_map = {}

    for c in df.columns:
        cl = str(c).lower()
        if cl == "date":
            rename_map[c] = "Date"
        elif cl == "open":
            rename_map[c] = "Open"
        elif cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl == "volume":
            rename_map[c] = "Volume"

    df = df.rename(columns=rename_map)
    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[needed].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna().sort_values("Date").reset_index(drop=True)


def download_history(ticker: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_downloaded(raw)


def load_sector_cache() -> Dict[str, Dict[str, str]]:
    if not os.path.exists(SECTOR_CACHE_FILE):
        return {}
    try:
        with open(SECTOR_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_sector_cache(cache: Dict[str, Dict[str, str]]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SECTOR_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_sector_info(ticker: str, cache: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    if ticker in cache:
        item = cache[ticker]
        return item.get("sector", ""), item.get("industry", "")

    sector = ""
    industry = ""

    try:
        info = yf.Ticker(ticker).info
        sector = str(info.get("sector", "") or "")
        industry = str(info.get("industry", "") or "")
    except Exception:
        sector = ""
        industry = ""

    cache[ticker] = {"sector": sector, "industry": industry}
    return sector, industry


def is_blocked_sector(sector: str, industry: str) -> bool:
    text = f"{sector} {industry}".lower().strip()
    if not text:
        return False
    return any(keyword in text for keyword in BLOCKED_KEYWORDS)


def get_rs_grade(rs_current_vs_high: float) -> str:
    if rs_current_vs_high >= RS_NEW_HIGH_TOL:
        return "S"
    return "A"


def rolling_return(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1.0


def build_result_lines(r: RSLeaderResult) -> List[str]:
    return [
        f"- {r.ticker} {r.name}",
        f"  RS: {r.rs_grade} | RS Percentile: {r.rs_percentile:.1f} | RS 고점 거리: {format_pct(r.rs_current_vs_high - 1.0)}",
        f"  3개월 수익률: {format_pct(r.ret_3m)}",
        f"  종가: {format_price(r.close)}",
        f"  52주 고점 거리: {format_pct(r.distance_to_52w_high)}",
        f"  섹터: {r.sector or '-'} | 산업: {r.industry or '-'}",
    ]


def notify_bucket(results: List[RSLeaderResult], bucket: str) -> None:
    if not telegram_enabled():
        return

    if bucket == "ready":
        title = "[RS 리더 — 돌파 준비]"
        max_rows = MAX_TELEGRAM_ROWS_READY
    else:
        title = "[RS 리더 — Watchlist]"
        max_rows = MAX_TELEGRAM_ROWS_WATCH

    lines = [title, f"전체 후보: {len(results)}"]

    for r in results[:max_rows]:
        lines.extend(build_result_lines(r))

    send_telegram_chunked(lines)


def save_outputs(results: List[RSLeaderResult]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cols = [f.name for f in RSLeaderResult.__dataclass_fields__.values()]

    if not results:
        pd.DataFrame(columns=cols).to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_at": datetime.now().isoformat(),
                    "total": 0,
                    "ready_count": 0,
                    "watch_count": 0,
                    "rs_s": 0,
                    "rs_a": 0,
                    "as_of_date": None,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return

    df = pd.DataFrame([asdict(r) for r in results])
    df["bucket_rank"] = df["bucket"].map({"ready": 0, "watch": 1}).fillna(9)
    df["rs_rank"] = df["rs_grade"].map({"S": 0, "A": 1}).fillna(9)

    df = df.sort_values(
        ["bucket_rank", "rs_rank", "rs_percentile", "distance_to_52w_high", "ticker"],
        ascending=[True, True, False, False, True],
    ).drop(columns=["bucket_rank", "rs_rank"])

    df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

    summary = {
        "run_at": datetime.now().isoformat(),
        "total": int(len(df)),
        "ready_count": int((df["bucket"] == "ready").sum()),
        "watch_count": int((df["bucket"] == "watch").sum()),
        "rs_s": int((df["rs_grade"] == "S").sum()),
        "rs_a": int((df["rs_grade"] == "A").sum()),
        "as_of_date": str(df["as_of_date"].max()),
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def scan_one(
    ticker: str,
    name: str,
    spy_df: pd.DataFrame,
    sector_cache: Dict[str, Dict[str, str]],
) -> Optional[RSLeaderResult]:
    df = download_history(ticker)
    if df.empty or len(df) < MIN_HISTORY:
        return None

    close = float(df["Close"].iloc[-1])
    if close < MIN_PRICE:
        return None

    df["dollar_vol_20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    dollar_vol_20 = safe_float(df["dollar_vol_20"].iloc[-1])
    if dollar_vol_20 is None or dollar_vol_20 < MIN_DOLLAR_VOL_20:
        return None

    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma150"] = df["Close"].rolling(150).mean()
    df["ma200"] = df["Close"].rolling(200).mean()

    ma50 = safe_float(df["ma50"].iloc[-1])
    ma150 = safe_float(df["ma150"].iloc[-1])
    ma200 = safe_float(df["ma200"].iloc[-1])

    if ma50 is None or ma150 is None or ma200 is None:
        return None

    if REQUIRE_MA_ALIGNMENT and not (close > ma50 > ma150 > ma200):
        return None

    ret_3m = safe_float(rolling_return(df["Close"], 63).iloc[-1])
    if ret_3m is None or ret_3m < MIN_3M_RETURN:
        return None

    stock = df.copy().set_index("Date")
    spy = spy_df.copy().set_index("Date")
    spy_close = spy["Close"].reindex(stock.index).ffill()

    rs_line = stock["Close"] / spy_close
    if len(rs_line) < RS_LOOKBACK:
        return None

    rs_high = float(rs_line.tail(RS_LOOKBACK).max())
    rs_low = float(rs_line.tail(RS_LOOKBACK).min())
    rs_now = float(rs_line.iloc[-1])

    if rs_high <= 0 or rs_high <= rs_low:
        return None

    rs_current_vs_high = rs_now / rs_high
    rs_percentile = ((rs_now - rs_low) / (rs_high - rs_low)) * 100.0

    if rs_percentile < RS_PERCENTILE_MIN:
        return None

    high_52w = float(df["High"].rolling(252).max().iloc[-1])
    if high_52w <= 0:
        return None

    distance_to_52w_high = close / high_52w - 1.0

    if distance_to_52w_high >= READY_MAX_DISTANCE:
        bucket = "ready"
    elif WATCH_MIN_DISTANCE <= distance_to_52w_high < WATCH_MAX_DISTANCE:
        bucket = "watch"
    else:
        return None

    sector, industry = get_sector_info(ticker, sector_cache)
    if is_blocked_sector(sector, industry):
        return None

    rs_grade = get_rs_grade(rs_current_vs_high)

    return RSLeaderResult(
        bucket=bucket,
        ticker=ticker,
        name=name,
        as_of_date=str(df["Date"].iloc[-1].date()),
        sector=sector,
        industry=industry,
        rs_grade=rs_grade,
        rs_percentile=round(rs_percentile, 2),
        rs_current_vs_high=round(rs_current_vs_high, 4),
        ret_3m=round(ret_3m, 4),
        close=round(close, 2),
        high_52w=round(high_52w, 2),
        distance_to_52w_high=round(distance_to_52w_high, 4),
        dollar_vol_20=round(float(dollar_vol_20), 2),
        ma50=round(float(ma50), 2),
        ma150=round(float(ma150), 2),
        ma200=round(float(ma200), 2),
    )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    universe = load_universe()
    spy = download_history(BENCHMARK)
    if spy.empty or len(spy) < MIN_HISTORY:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    sector_cache = load_sector_cache()
    results: List[RSLeaderResult] = []

    for row in universe.itertuples(index=False):
        try:
            result = scan_one(row.ticker, row.name, spy, sector_cache)
            if result is not None:
                results.append(result)
        except Exception:
            continue

    save_sector_cache(sector_cache)
    save_outputs(results)

    ready_results = [r for r in results if r.bucket == "ready"]
    watch_results = [r for r in results if r.bucket == "watch"]

    ready_results = sorted(
        ready_results,
        key=lambda x: (
            {"S": 0, "A": 1}.get(x.rs_grade, 9),
            -x.rs_percentile,
            -x.distance_to_52w_high,
            x.ticker,
        ),
    )
    watch_results = sorted(
        watch_results,
        key=lambda x: (
            {"S": 0, "A": 1}.get(x.rs_grade, 9),
            -x.rs_percentile,
            -x.ret_3m,
            x.ticker,
        ),
    )

    notify_bucket(ready_results, "ready")
    notify_bucket(watch_results, "watch")


if __name__ == "__main__":
    main()
