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
PROFILE_CACHE_FILE = os.path.join(OUTPUT_DIR, "rs_leader_profile_cache.json")
UNIVERSE_META_FILE = os.path.join(OUTPUT_DIR, "universe_build_meta.json")

BENCHMARK = "SPY"
DOWNLOAD_PERIOD = "2y"
MIN_HISTORY = 260

# -----------------------------
# Hard cuts
# -----------------------------
MIN_PRICE = 20.0
MIN_DOLLAR_VOL_20 = 25_000_000
MIN_3M_RETURN = 0.15
MIN_MARKET_CAP = 3_000_000_000

RS_LOOKBACK = 252
RS_PERCENTILE_MIN = 95.0
RS_NEW_HIGH_TOL = 0.995

READY_MAX_DISTANCE = -0.05
WATCH_MIN_DISTANCE = -0.30
WATCH_MAX_DISTANCE = -0.05

REQUIRE_MA_ALIGNMENT = True
MIN_ACCEPTABLE_UNIVERSE = 1500

# -----------------------------
# Event / M&A filters
# -----------------------------
MAX_20D_RETURN = 0.35
MAX_SINGLE_DAY_SPIKE = 0.18
MIN_10D_RANGE = 0.018
MAX_15D_RETURN = 0.35
MAX_GAP_DAY = 0.15

# -----------------------------
# Blocked keywords
# -----------------------------
BLOCKED_SECTOR_INDUSTRY_KEYWORDS = [
    "energy", "oil", "gas", "midstream", "upstream", "downstream",
    "utilities", "utility",
    "insurance",
    "food", "beverage", "tobacco",
    "biotech", "biotechnology", "drug manufacturers", "pharmaceutical", "pharma",
    "metals", "mining", "coal", "materials", "basic materials", "steel", "aluminum",
    "precious metals", "gold", "silver",
    "reit", "real estate investment trust", "real estate",
    "residential construction", "homebuilding", "home builders",
    "engineering & construction", "engineering and construction",
    "farm products", "agricultural inputs", "agricultural products",
    "medical care facilities",
    "marine shipping", "shipping", "water transportation",
    "specialty finance", "mortgage", "regional banks",
    "holding companies", "asset management",
    "casinos", "gaming", "gambling",
    "staffing", "employment services",
    "solar", "uranium", "cannabis", "crypto", "bitcoin",
    "airlines", "air freight", "trucking",
]

BLOCKED_NAME_KEYWORDS = [
    "holdings",
    "acquisition",
    "royalty",
    "partnership",
    "l.p.",
    "lp",
    "adr",
    "ads",
    "depositary",
    "trust",
]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MAX_TELEGRAM_MESSAGE_LEN = 3500
MAX_TELEGRAM_ROWS_READY = 25
MAX_TELEGRAM_ROWS_WATCH = 40


@dataclass
class RSLeaderResult:
    bucket: str
    ticker: str
    name: str
    as_of_date: str
    sector: str
    industry: str
    country: str
    rs_grade: str
    rs_percentile: float
    rs_current_vs_high: float
    ret_3m: float
    ret_20d: float
    close: float
    high_52w: float
    distance_to_52w_high: float
    dollar_vol_20: float
    market_cap: float
    trailing_pe: float
    profit_margin: float
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


def format_cap(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    v = float(x)
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.0f}M"
    return f"${v:,.0f}"


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
        candidate = f"{current}\n{line}".rstrip() if current else line.rstrip()
        if len(candidate) > MAX_TELEGRAM_MESSAGE_LEN:
            if current:
                chunks.append(current)
            current = line.rstrip()
        else:
            current = candidate

    if current:
        chunks.append(current)

    for chunk in chunks:
        send_telegram(chunk)


def ensure_universe_file() -> None:
    need_rebuild = False

    if not os.path.exists(UNIVERSE_FILE):
        need_rebuild = True
    else:
        try:
            df = pd.read_csv(UNIVERSE_FILE)
            if len(df) < MIN_ACCEPTABLE_UNIVERSE:
                need_rebuild = True
        except Exception:
            need_rebuild = True

    if need_rebuild:
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


def load_profile_cache() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(PROFILE_CACHE_FILE):
        return {}
    try:
        with open(PROFILE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_profile_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROFILE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_universe_meta() -> Dict[str, Any]:
    if not os.path.exists(UNIVERSE_META_FILE):
        return {}
    try:
        with open(UNIVERSE_META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_profile_info(ticker: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if ticker in cache:
        return cache[ticker]

    profile = {
        "sector": "",
        "industry": "",
        "country": "",
        "quoteType": "",
        "marketCap": None,
        "trailingPE": None,
        "profitMargins": None,
    }

    try:
        info = yf.Ticker(ticker).info
        profile["sector"] = str(info.get("sector", "") or "")
        profile["industry"] = str(info.get("industry", "") or "")
        profile["country"] = str(info.get("country", "") or "")
        profile["quoteType"] = str(info.get("quoteType", "") or "")
        profile["marketCap"] = safe_float(info.get("marketCap"))
        profile["trailingPE"] = safe_float(info.get("trailingPE"))
        profile["profitMargins"] = safe_float(info.get("profitMargins"))
    except Exception:
        pass

    cache[ticker] = profile
    return profile


def is_blocked_profile(profile: Dict[str, Any], company_name: str) -> bool:
    quote_type = str(profile.get("quoteType", "") or "").lower()
    country = str(profile.get("country", "") or "").lower()
    sector = str(profile.get("sector", "") or "")
    industry = str(profile.get("industry", "") or "")
    company_name = str(company_name or "")

    text = f"{sector} {industry} {company_name}".lower().strip()

    if quote_type and quote_type != "equity":
        return True

    if country and "united states" not in country and "usa" not in country:
        return True

    if any(keyword in text for keyword in BLOCKED_SECTOR_INDUSTRY_KEYWORDS):
        return True

    if any(keyword in text for keyword in BLOCKED_NAME_KEYWORDS):
        return True

    return False


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
        f"  3개월 수익률: {format_pct(r.ret_3m)} | 20일 수익률: {format_pct(r.ret_20d)}",
        f"  종가: {format_price(r.close)} | 52주 고점 거리: {format_pct(r.distance_to_52w_high)}",
        f"  시총: {format_cap(r.market_cap)} | 거래대금20D: {format_cap(r.dollar_vol_20)}",
        f"  섹터: {r.sector or '-'} | 산업: {r.industry or '-'} | 국가: {r.country or '-'}",
        "",
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

    lines = [title, f"전체 후보: {len(results)}", ""]

    for r in results[:max_rows]:
        lines.extend(build_result_lines(r))

    send_telegram_chunked(lines)


def send_summary(stats: Dict[str, Any], universe_meta: Dict[str, Any]) -> None:
    if not telegram_enabled():
        return

    source_counts = universe_meta.get("source_counts", {})
    msg = (
        "[RS-RADAR 요약]\n"
        f"실행 시각: {stats['run_at']}\n"
        f"유니버스 수: {stats['universe_count']}\n"
        f"IWV Russell3000: {source_counts.get('IWV_Russell_3000', 0)}\n"
        f"IVV S&P500: {source_counts.get('IVV_SP500', 0)}\n"
        f"IJH S&P400: {source_counts.get('IJH_SP400', 0)}\n"
        f"IJR S&P600: {source_counts.get('IJR_SP600', 0)}\n"
        f"데이터 정상 수집: {stats['download_ok']}\n"
        f"데이터 부족/실패: {stats['download_fail_or_short']}\n"
        f"가격/거래대금 탈락: {stats['liquidity_fail']}\n"
        f"이평 정렬 탈락: {stats['ma_fail']}\n"
        f"3개월 수익률 15% 미만 탈락: {stats['ret3m_fail']}\n"
        f"RS 95 미만 탈락: {stats['rs_fail']}\n"
        f"프로필/국가/차단섹터 탈락: {stats['profile_block_fail']}\n"
        f"품질/수익성 탈락: {stats['quality_fail']}\n"
        f"M&A/이벤트 패턴 탈락: {stats['event_fail']}\n"
        f"거리 조건 탈락: {stats['distance_fail']}\n"
        f"RS 리더 최종 후보: {stats['final_total']}\n"
        f"돌파 준비(-5% 이내): {stats['ready_count']}\n"
        f"Watchlist(-30%~-5%): {stats['watch_count']}\n"
        f"RS S: {stats['rs_s']}\n"
        f"RS A: {stats['rs_a']}"
    )
    send_telegram(msg)


def save_outputs(results: List[RSLeaderResult], stats: Dict[str, Any]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cols = [f.name for f in RSLeaderResult.__dataclass_fields__.values()]

    if not results:
        pd.DataFrame(columns=cols).to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    else:
        df = pd.DataFrame([asdict(r) for r in results])
        df["bucket_rank"] = df["bucket"].map({"ready": 0, "watch": 1}).fillna(9)
        df["rs_rank"] = df["rs_grade"].map({"S": 0, "A": 1}).fillna(9)

        df = df.sort_values(
            ["bucket_rank", "rs_rank", "rs_percentile", "distance_to_52w_high", "ticker"],
            ascending=[True, True, False, False, True],
        ).drop(columns=["bucket_rank", "rs_rank"])

        df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def is_event_or_ma_pattern(df: pd.DataFrame) -> bool:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]

    ret_20d = safe_float(rolling_return(close, 20).iloc[-1])
    ret_15d = safe_float(rolling_return(close, 15).iloc[-1])

    if ret_20d is not None and ret_20d > MAX_20D_RETURN:
        return True

    if ret_15d is not None and ret_15d > MAX_15D_RETURN:
        return True

    daily_ret = close.pct_change().tail(20)
    if not daily_ret.empty and daily_ret.max() > MAX_SINGLE_DAY_SPIKE:
        return True

    gap_ret = (open_ / close.shift(1) - 1.0).tail(20)
    if not gap_ret.empty and gap_ret.max() > MAX_GAP_DAY:
        return True

    recent_high = safe_float(high.tail(10).max())
    recent_low = safe_float(low.tail(10).min())
    current_close = safe_float(close.iloc[-1])

    if recent_high and recent_low and current_close and current_close > 0:
        range_10d = (recent_high - recent_low) / current_close
        if range_10d < MIN_10D_RANGE:
            return True

    return False


def passes_quality_filters(profile: Dict[str, Any]) -> bool:
    market_cap = safe_float(profile.get("marketCap"))
    trailing_pe = safe_float(profile.get("trailingPE"))
    profit_margin = safe_float(profile.get("profitMargins"))

    if market_cap is None or market_cap < MIN_MARKET_CAP:
        return False

    if trailing_pe is None or trailing_pe <= 0:
        return False

    if profit_margin is None or profit_margin <= 0:
        return False

    return True


def scan_one(
    ticker: str,
    name: str,
    spy_df: pd.DataFrame,
    profile_cache: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[RSLeaderResult], str]:
    df = download_history(ticker)
    if df.empty or len(df) < MIN_HISTORY:
        return None, "download_fail_or_short"

    close = float(df["Close"].iloc[-1])
    if close < MIN_PRICE:
        return None, "liquidity_fail"

    df["dollar_vol_20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    dollar_vol_20 = safe_float(df["dollar_vol_20"].iloc[-1])
    if dollar_vol_20 is None or dollar_vol_20 < MIN_DOLLAR_VOL_20:
        return None, "liquidity_fail"

    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma150"] = df["Close"].rolling(150).mean()
    df["ma200"] = df["Close"].rolling(200).mean()

    ma50 = safe_float(df["ma50"].iloc[-1])
    ma150 = safe_float(df["ma150"].iloc[-1])
    ma200 = safe_float(df["ma200"].iloc[-1])

    if ma50 is None or ma150 is None or ma200 is None:
        return None, "download_fail_or_short"

    if REQUIRE_MA_ALIGNMENT and not (close > ma50 > ma150 > ma200):
        return None, "ma_fail"

    ret_3m = safe_float(rolling_return(df["Close"], 63).iloc[-1])
    if ret_3m is None or ret_3m < MIN_3M_RETURN:
        return None, "ret3m_fail"

    stock = df.copy().set_index("Date")
    spy = spy_df.copy().set_index("Date")
    spy_close = spy["Close"].reindex(stock.index).ffill()

    rs_line = stock["Close"] / spy_close
    if len(rs_line) < RS_LOOKBACK:
        return None, "download_fail_or_short"

    rs_high = float(rs_line.tail(RS_LOOKBACK).max())
    rs_low = float(rs_line.tail(RS_LOOKBACK).min())
    rs_now = float(rs_line.iloc[-1])

    if rs_high <= 0 or rs_high <= rs_low:
        return None, "download_fail_or_short"

    rs_current_vs_high = rs_now / rs_high
    rs_percentile = ((rs_now - rs_low) / (rs_high - rs_low)) * 100.0

    if rs_percentile < RS_PERCENTILE_MIN:
        return None, "rs_fail"

    profile = get_profile_info(ticker, profile_cache)

    if is_blocked_profile(profile, name):
        return None, "profile_block_fail"

    if not passes_quality_filters(profile):
        return None, "quality_fail"

    if is_event_or_ma_pattern(df):
        return None, "event_fail"

    high_52w = float(df["High"].rolling(252).max().iloc[-1])
    if high_52w <= 0:
        return None, "download_fail_or_short"

    distance_to_52w_high = close / high_52w - 1.0

    if distance_to_52w_high >= READY_MAX_DISTANCE:
        bucket = "ready"
    elif WATCH_MIN_DISTANCE <= distance_to_52w_high < WATCH_MAX_DISTANCE:
        bucket = "watch"
    else:
        return None, "distance_fail"

    rs_grade = get_rs_grade(rs_current_vs_high)
    ret_20d = safe_float(rolling_return(df["Close"], 20).iloc[-1])

    result = RSLeaderResult(
        bucket=bucket,
        ticker=ticker,
        name=name,
        as_of_date=str(df["Date"].iloc[-1].date()),
        sector=str(profile.get("sector", "") or ""),
        industry=str(profile.get("industry", "") or ""),
        country=str(profile.get("country", "") or ""),
        rs_grade=rs_grade,
        rs_percentile=round(rs_percentile, 2),
        rs_current_vs_high=round(rs_current_vs_high, 4),
        ret_3m=round(ret_3m, 4),
        ret_20d=round(ret_20d, 4) if ret_20d is not None else 0.0,
        close=round(close, 2),
        high_52w=round(high_52w, 2),
        distance_to_52w_high=round(distance_to_52w_high, 4),
        dollar_vol_20=round(float(dollar_vol_20), 2),
        market_cap=round(float(profile.get("marketCap")), 2),
        trailing_pe=round(float(profile.get("trailingPE")), 2),
        profit_margin=round(float(profile.get("profitMargins")), 4),
        ma50=round(float(ma50), 2),
        ma150=round(float(ma150), 2),
        ma200=round(float(ma200), 2),
    )
    return result, "ok"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    universe = load_universe()
    universe_meta = load_universe_meta()

    spy = download_history(BENCHMARK)
    if spy.empty or len(spy) < MIN_HISTORY:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    profile_cache = load_profile_cache()
    results: List[RSLeaderResult] = []

    stats: Dict[str, Any] = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "universe_count": int(len(universe)),
        "download_ok": 0,
        "download_fail_or_short": 0,
        "liquidity_fail": 0,
        "ma_fail": 0,
        "ret3m_fail": 0,
        "rs_fail": 0,
        "profile_block_fail": 0,
        "quality_fail": 0,
        "event_fail": 0,
        "distance_fail": 0,
        "final_total": 0,
        "ready_count": 0,
        "watch_count": 0,
        "rs_s": 0,
        "rs_a": 0,
        "as_of_date": None,
    }

    for row in universe.itertuples(index=False):
        try:
            result, status = scan_one(row.ticker, row.name, spy, profile_cache)
            if status == "ok" and result is not None:
                stats["download_ok"] += 1
                results.append(result)
            else:
                stats[status] += 1
        except Exception:
            stats["download_fail_or_short"] += 1
            continue

    save_profile_cache(profile_cache)

    ready_results = [r for r in results if r.bucket == "ready"]
    watch_results = [r for r in results if r.bucket == "watch"]

    ready_results = sorted(
        ready_results,
        key=lambda x: (
            {"S": 0, "A": 1}.get(x.rs_grade, 9),
            -x.rs_percentile,
            -x.distance_to_52w_high,
            -x.ret_3m,
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

    stats["final_total"] = len(results)
    stats["ready_count"] = len(ready_results)
    stats["watch_count"] = len(watch_results)
    stats["rs_s"] = sum(1 for r in results if r.rs_grade == "S")
    stats["rs_a"] = sum(1 for r in results if r.rs_grade == "A")
    stats["as_of_date"] = results[0].as_of_date if results else None

    save_outputs(results, stats)
    notify_bucket(ready_results, "ready")
    notify_bucket(watch_results, "watch")
    send_summary(stats, universe_meta)


if __name__ == "__main__":
    main()
