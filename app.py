import streamlit as st
import pandas as pd
import pandas_ta_classic as ta
from concurrent.futures import ThreadPoolExecutor, as_completed

from providers import fetch_ohlcv_with_failover, provider_health_check

# =========================================================
# APP
# =========================================================
APP_VERSION = "CryptoOS Hybrid Breakout V7"
st.set_page_config(page_title="CryptoOS", layout="wide")

# =========================================================
# CONFIG
# =========================================================
TIMEFRAME_MAP = {
    "4h": ("4h", 400),
    "1d": ("1d", 400),
    "1w": ("1w", 400),
    "1M": ("1M", 400),
}

MAX_WORKERS = 8
MACRO_TIMEFRAME = "1d"
MACRO_RECENT_PIVOT_MAX_AGE = 90

# =========================================================
# UNIVERSE
# =========================================================
RAW_UNIVERSE = [
    ("BTCUSDT", "Bitcoin"),
    ("ETHUSDT", "Ethereum"),
    ("XRPUSDT", "XRP"),
    ("BNBUSDT", "BNB"),
    ("SOLUSDT", "Solana"),
    ("TRXUSDT", "TRON"),
    ("DOGEUSDT", "Dogecoin"),
    ("HYPEUSDT", "Hyperliquid"),
    ("LEOUSDT", "UNUS SED LEO"),
    ("ADAUSDT", "Cardano"),
    ("BCHUSDT", "Bitcoin Cash"),
    ("LINKUSDT", "Chainlink"),
    ("XMRUSDT", "Monero"),
    ("CCUSDT", "Canton"),
    ("XLMUSDT", "Stellar"),
    ("ZECUSDT", "Zcash"),
    ("MUSDT", "MemeCore"),
    ("LTCUSDT", "Litecoin"),
    ("AVAXUSDT", "Avalanche"),
    ("HBARUSDT", "Hedera"),
    ("SUIUSDT", "Sui"),
    ("SHIBUSDT", "Shiba Inu"),
    ("TONUSDT", "Toncoin"),
    ("TAOUSDT", "Bittensor"),
    ("DOTUSDT", "Polkadot"),
    ("UNIUSDT", "Uniswap"),
    ("NEARUSDT", "NEAR Protocol"),
    ("APTUSDT", "Aptos"),
    ("PEPEUSDT", "Pepe"),
    ("OPUSDT", "Optimism"),
    ("ARBUSDT", "Arbitrum"),
    ("KASUSDT", "Kaspa"),
    ("ICPUSDT", "Internet Computer"),
    ("STXUSDT", "Stacks"),
    ("RENDERUSDT", "Render"),
    ("INJUSDT", "Injective"),
    ("IMXUSDT", "Immutable"),
    ("GRTUSDT", "The Graph"),
    ("FTMUSDT", "Fantom"),
    ("RUNEUSDT", "Thorchain"),
    ("FETUSDT", "Fetch.ai"),
    ("ARUSDT", "Arweave"),
    ("SEIUSDT", "Sei"),
    ("ALGOUSDT", "Algorand"),
    ("QNTUSDT", "Quant"),
    ("MNTUSDT", "Mantle"),
    ("FLOWUSDT", "Flow"),
    ("AAVEUSDT", "Aave"),
    ("LDOUSDT", "Lido DAO"),
    ("THETAUSDT", "Theta Network"),
    ("POLUSDT", "Polygon"),
    ("PYTHUSDT", "Pyth Network"),
    ("JUPUSDT", "Jupiter"),
    ("BONKUSDT", "Bonk"),
    ("TIAUSDT", "Celestia"),
    ("ONDOUSDT", "Ondo"),
    ("MKRUSDT", "Maker"),
    ("JASMYUSDT", "JasmyCoin"),
    ("GALAUSDT", "Gala"),
    ("ENAUSDT", "Ethena"),
    ("WLDUSDT", "Worldcoin"),
    ("AGIXUSDT", "SingularityNET"),
    ("SNXUSDT", "Synthetix"),
    ("EOSUSDT", "EOS"),
    ("FLRUSDT", "Flare"),
    ("CRVUSDT", "Curve DAO Token"),
    ("RPLUSDT", "Rocket Pool"),
    ("MINAUSDT", "Mina"),
    ("XTZUSDT", "Tezos"),
    ("AXSUSDT", "Axie Infinity"),
    ("CFXUSDT", "Conflux"),
    ("CHZUSDT", "Chiliz"),
    ("NEOUSDT", "Neo"),
    ("CAKEUSDT", "PancakeSwap"),
    ("SANDUSDT", "Sandbox"),
    ("MANAUSDT", "Decentraland"),
    ("AKTUSDT", "Akash Network"),
    ("BGBUSDT", "Bitget Token"),
    ("STRKUSDT", "Starknet"),
    ("DYMUSDT", "Dymension"),
    ("AEVOUSDT", "Aevo"),
    ("SATSUSDT", "Satoshi"),
    ("CKBUSDT", "Nervos Network"),
    ("IOTAUSDT", "IOTA"),
    ("HNTUSDT", "Helium"),
    ("RONINUSDT", "Ronin"),
    ("WEMIXUSDT", "WEMIX"),
    ("BLURUSDT", "Blur"),
    ("ETHFIUSDT", "Ether.fi"),
    ("METISUSDT", "Metis"),
    ("FXSUSDT", "Frax Share"),
    ("ZILUSDT", "Zilliqa"),
    ("GASUSDT", "Gas"),
    ("GNOUSDT", "Gnosis"),
    ("KAVAUSDT", "Kava"),
    ("ASTRUSDT", "Astar"),
    ("BEAMUSDT", "Beam"),
    ("LPTUSDT", "Livepeer"),
]

def build_custom_universe() -> pd.DataFrame:
    seen = set()
    rows = []
    rank = 1
    for symbol, name in RAW_UNIVERSE:
        if symbol in seen:
            continue
        seen.add(symbol)
        rows.append({"rank": rank, "name": name, "symbol": symbol})
        rank += 1
    return pd.DataFrame(rows)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Kontrol Paneli")
st.sidebar.caption(APP_VERSION)

view_mode = st.sidebar.radio("Görünüm", ["Tekil Analiz", "Piyasa Taraması"], index=1)
symbol_input = st.sidebar.text_input("Varlık (Örn: BTC-USD):", "BTC-USD").upper().strip()
mode = st.sidebar.radio("Mod", ["Spot", "Swing"], index=1)
timeframe = st.sidebar.selectbox("Zaman Dilimi", list(TIMEFRAME_MAP.keys()), index=0)

capital = st.sidebar.number_input("Toplam Kasa ($)", min_value=100.0, value=10000.0, step=100.0)
risk_per_trade = st.sidebar.slider("İşlem Başı Risk (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
min_score = st.sidebar.slider("Minimum Skor", min_value=0, max_value=100, value=40, step=5)

trendline_module = st.sidebar.checkbox("Otomatik Trend Takibi", value=True)
trend_lookback = st.sidebar.slider("Trend Lookback", min_value=20, max_value=160, value=100, step=5)
pivot_window = st.sidebar.slider("Pivot Hassasiyeti", min_value=1, max_value=10, value=3, step=1)
breakout_tolerance_pct = st.sidebar.slider("Breakout Toleransı (%)", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

macro_module = st.sidebar.checkbox("Makro Trend Kırılımı", value=True)
macro_lookback = st.sidebar.slider("Makro Lookback", min_value=150, max_value=600, value=400, step=25)
macro_pivot_window = st.sidebar.slider("Makro Pivot Hassasiyeti", min_value=4, max_value=20, value=8, step=1)
macro_breakout_max_pct = st.sidebar.slider("Makro Breakout Üst Sınır (%)", min_value=1.0, max_value=10.0, value=6.0, step=0.5)

show_trend_debug = st.sidebar.checkbox("Trend Debug Göster", value=True)

# =========================================================
# UTILS
# =========================================================
def normalize_symbol(user_symbol: str) -> str:
    s = user_symbol.strip().upper()
    if s.endswith("USDT"):
        return s
    if s.endswith("-USD"):
        return s.replace("-USD", "USDT")
    if s.endswith("USD"):
        return s[:-3] + "USDT"
    return s + "USDT"

@st.cache_data(ttl=300)
def fetch_klines(symbol: str, timeframe_key: str):
    df, err, source, provider_errors = fetch_ohlcv_with_failover(symbol, timeframe_key)
    return df, err, source, provider_errors

# =========================================================
# INDICATORS
# =========================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 220:
        return pd.DataFrame()

    out = df.copy()
    out["EMA20"] = ta.ema(out["close"], length=20)
    out["EMA50"] = ta.ema(out["close"], length=50)
    out["EMA100"] = ta.ema(out["close"], length=100)
    out["EMA200"] = ta.ema(out["close"], length=200)
    out["RSI"] = ta.rsi(out["close"], length=14)
    out["ATR"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["VOL_MA20"] = ta.sma(out["volume"], length=20)
    out = out.dropna()
    return out

def compute_btc_context(timeframe_key: str):
    btc_raw, _, _, _ = fetch_klines("BTCUSDT", timeframe_key)
    if btc_raw.empty:
        return None

    btc_ind = add_indicators(btc_raw)
    if btc_ind.empty:
        return None

    last = btc_ind.iloc[-1]
    price = float(last["close"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    ema200 = float(last["EMA200"])
    rsi = float(last["RSI"])

    if price > ema200 and ema20 > ema50 > ema200 and rsi >= 58:
        market_phase = "Risk On"
    elif price > ema200 and rsi >= 52:
        market_phase = "Neutral+"
    elif price < ema200 and rsi < 45:
        market_phase = "Risk Off"
    else:
        market_phase = "Neutral"

    return {
        "price": price,
        "rsi": rsi,
        "trend": "Bullish" if price > ema200 else "Bearish",
        "market_phase": market_phase,
        "ind_df": btc_ind,
    }

def compute_rs_vs_btc(symbol_ind_df: pd.DataFrame, btc_ind_df: pd.DataFrame) -> float:
    s = symbol_ind_df.set_index("open_time")
    b = btc_ind_df.set_index("open_time")
    common = s.index.intersection(b.index)

    if len(common) < 15:
        return 0.0

    s_close = s.loc[common, "close"]
    b_close = b.loc[common, "close"]

    try:
        s_ret = (float(s_close.iloc[-1]) / float(s_close.iloc[-15]) - 1) * 100
        b_ret = (float(b_close.iloc[-1]) / float(b_close.iloc[-15]) - 1) * 100
        return s_ret - b_ret
    except Exception:
        return 0.0

# =========================================================
# MICRO TREND
# =========================================================
def build_peak_candidates(raw_df: pd.DataFrame, lookback: int, pivot_window_value: int):
    local_df = raw_df.tail(lookback).copy().reset_index(drop=True)
    highs = local_df["high"].tolist()

    if len(local_df) < 40:
        return local_df, [], f"local_df={len(local_df)}<40"

    pivots = []
    w = max(2, pivot_window_value)

    for i in range(w, len(highs) - w):
        left = highs[i - w:i]
        right = highs[i + 1:i + w + 1]
        center = highs[i]
        if center >= max(left) and center >= max(right):
            pivots.append((i, float(center)))

    if len(pivots) >= 3:
        return local_df, pivots, f"pivot bulundu ({len(pivots)})"

    ranked = sorted([(i, float(v)) for i, v in enumerate(highs)], key=lambda x: x[1], reverse=True)
    filtered = []
    min_gap = max(5, w * 3)

    for idx, price in ranked:
        too_close = any(abs(idx - ex_idx) < min_gap for ex_idx, _ in filtered)
        if not too_close:
            filtered.append((idx, price))
        if len(filtered) >= 10:
            break

    filtered = sorted(filtered, key=lambda x: x[0])

    if len(filtered) >= 3:
        return local_df, filtered, f"fallback tepe bulundu ({len(filtered)})"

    return local_df, [], "yeterli tepe yok"

def compute_desc_trendline_breakout(raw_df: pd.DataFrame, lookback=100, pivot_window_value=3, tolerance_pct=0.3):
    local_df, peaks, prep_note = build_peak_candidates(raw_df, lookback, pivot_window_value)

    if len(local_df) < 40:
        return {
            "is_breakout": False,
            "trendline_value": None,
            "breakout_pct": None,
            "trend_note": "veri yetersiz",
            "trend_debug": prep_note,
            "trend_input_bars": len(local_df),
        }

    if len(peaks) < 3:
        return {
            "is_breakout": False,
            "trendline_value": None,
            "breakout_pct": None,
            "trend_note": "yeterli pivot yok",
            "trend_debug": prep_note,
            "trend_input_bars": len(local_df),
        }

    current_idx = len(local_df) - 1
    current_close = float(local_df.iloc[-1]["close"])

    ema20 = ta.ema(local_df["close"], length=20)
    ema50 = ta.ema(local_df["close"], length=50)

    if ema20.isna().all() or ema50.isna().all():
        return {
            "is_breakout": False,
            "trendline_value": None,
            "breakout_pct": None,
            "trend_note": "trend filtresi yok",
            "trend_debug": "ema unavailable",
            "trend_input_bars": len(local_df),
        }

    ema20_last = float(ema20.dropna().iloc[-1])
    ema50_last = float(ema50.dropna().iloc[-1])

    downtrend_bias = (current_close < ema20_last * 1.03 or ema20_last <= ema50_last * 1.02)
    if not downtrend_bias:
        return {
            "is_breakout": False,
            "trendline_value": None,
            "breakout_pct": None,
            "trend_note": "uygun trend yapısı yok",
            "trend_debug": "downtrend_bias=false",
            "trend_input_bars": len(local_df),
        }

    min_gap = max(6, pivot_window_value * 3)
    max_gap = min(60, max(25, lookback // 2))
    recent_pivot_threshold = current_idx - min(35, max(18, lookback // 3))

    best_candidate = None
    candidates_checked = 0

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            x1, y1 = peaks[i]
            x2, y2 = peaks[j]
            candidates_checked += 1

            gap = x2 - x1
            if y2 >= y1:
                continue
            if gap < min_gap or gap > max_gap:
                continue
            if x2 < recent_pivot_threshold:
                continue

            slope = (y2 - y1) / (x2 - x1)
            projected = y1 + slope * (current_idx - x1)

            if projected <= 0:
                continue
            if projected < current_close * 0.95:
                continue
            if projected > current_close * 1.20:
                continue

            violations = 0
            for k, hk in peaks:
                if x1 < k < current_idx:
                    line_k = y1 + slope * (k - x1)
                    if hk > line_k * 1.02:
                        violations += 1

            if violations > 2:
                continue

            distance_pct = abs((projected / current_close) - 1) * 100
            candidate = {
                "projected": projected,
                "violations": violations,
                "distance_pct": distance_pct,
                "gap": gap,
                "x1": x1,
                "x2": x2,
            }

            if best_candidate is None:
                best_candidate = candidate
            else:
                if candidate["distance_pct"] < best_candidate["distance_pct"]:
                    best_candidate = candidate
                elif candidate["distance_pct"] == best_candidate["distance_pct"] and candidate["x2"] > best_candidate["x2"]:
                    best_candidate = candidate

    if best_candidate is None:
        return {
            "is_breakout": False,
            "trendline_value": None,
            "breakout_pct": None,
            "trend_note": "geçerli lower high trendi yok",
            "trend_debug": f"{prep_note} | checked={candidates_checked}",
            "trend_input_bars": len(local_df),
        }

    trendline_value = best_candidate["projected"]
    breakout_threshold = trendline_value * (1 + tolerance_pct / 100)
    breakout_pct = ((current_close / trendline_value) - 1) * 100
    is_breakout = current_close > breakout_threshold
    trend_note = "kırılım var" if is_breakout else "kırılım yok"

    return {
        "is_breakout": is_breakout,
        "trendline_value": round(trendline_value, 6),
        "breakout_pct": round(breakout_pct, 2),
        "trend_note": trend_note,
        "trend_debug": (
            f"pivot bulundu ({len(peaks)}) | gap={best_candidate['gap']} | "
            f"viol={best_candidate['violations']} | p1={best_candidate['x1']},p2={best_candidate['x2']}"
        ),
        "trend_input_bars": len(local_df),
    }

# =========================================================
# MACRO TREND
# =========================================================
def build_macro_peaks(raw_df: pd.DataFrame, lookback: int, pivot_window_value: int):
    local_df = raw_df.tail(lookback).copy().reset_index(drop=True)
    highs = local_df["high"].tolist()

    if len(local_df) < 120:
        return local_df, [], f"local_df={len(local_df)}<120"

    pivots = []
    w = max(4, pivot_window_value)
    for i in range(w, len(highs) - w):
        left = highs[i - w:i]
        right = highs[i + 1:i + w + 1]
        center = highs[i]
        if center >= max(left) and center >= max(right):
            pivots.append((i, float(center)))

    if len(pivots) >= 3:
        return local_df, pivots, f"macro pivot bulundu ({len(pivots)})"

    return local_df, [], "macro pivot yetersiz"

def compute_macro_trend_breakout(raw_df: pd.DataFrame, lookback=400, pivot_window_value=8, breakout_max_pct=6.0):
    local_df, peaks, prep_note = build_macro_peaks(raw_df, lookback, pivot_window_value)

    if len(local_df) < 120:
        return {
            "is_macro_breakout": False,
            "macro_trendline_value": None,
            "macro_breakout_pct": None,
            "macro_note": "makro veri yetersiz",
            "macro_debug": prep_note,
            "macro_input_bars": len(local_df),
        }

    if len(peaks) < 3:
        return {
            "is_macro_breakout": False,
            "macro_trendline_value": None,
            "macro_breakout_pct": None,
            "macro_note": "makro pivot yetersiz",
            "macro_debug": prep_note,
            "macro_input_bars": len(local_df),
        }

    current_idx = len(local_df) - 1
    current_close = float(local_df.iloc[-1]["close"])
    prev_close = float(local_df.iloc[-2]["close"])

    best_candidate = None
    checked = 0
    min_gap = 20
    max_gap = 180
    recent_pivot_threshold = current_idx - MACRO_RECENT_PIVOT_MAX_AGE

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            x1, y1 = peaks[i]
            x2, y2 = peaks[j]
            checked += 1

            gap = x2 - x1
            if y2 >= y1:
                continue
            if gap < min_gap or gap > max_gap:
                continue
            if x2 < recent_pivot_threshold:
                continue

            slope = (y2 - y1) / (x2 - x1)
            tl_now = y1 + slope * (current_idx - x1)
            tl_prev = y1 + slope * ((current_idx - 1) - x1)

            if tl_now <= 0 or tl_prev <= 0:
                continue
            if tl_now < current_close * 0.90:
                continue
            if tl_now > current_close * 1.25:
                continue

            touch_count = 0
            violations = 0
            for k, hk in peaks:
                if x1 <= k <= x2:
                    line_k = y1 + slope * (k - x1)
                    if abs(hk - line_k) / max(line_k, 1e-9) <= 0.02:
                        touch_count += 1
                    elif hk > line_k * 1.03:
                        violations += 1

            if touch_count < 2:
                continue
            if violations > 2:
                continue

            breakout_pct = ((current_close / tl_now) - 1) * 100
            candidate = {
                "x1": x1,
                "x2": x2,
                "tl_now": tl_now,
                "tl_prev": tl_prev,
                "breakout_pct": breakout_pct,
                "touch_count": touch_count,
                "violations": violations,
            }

            if best_candidate is None:
                best_candidate = candidate
            else:
                old_dist = abs(best_candidate["tl_now"] - current_close)
                new_dist = abs(candidate["tl_now"] - current_close)
                if new_dist < old_dist:
                    best_candidate = candidate
                elif new_dist == old_dist and candidate["touch_count"] > best_candidate["touch_count"]:
                    best_candidate = candidate

    if best_candidate is None:
        return {
            "is_macro_breakout": False,
            "macro_trendline_value": None,
            "macro_breakout_pct": None,
            "macro_note": "geçerli makro trend yok",
            "macro_debug": f"{prep_note} | checked={checked}",
            "macro_input_bars": len(local_df),
        }

    tl_now = best_candidate["tl_now"]
    tl_prev = best_candidate["tl_prev"]
    breakout_pct = best_candidate["breakout_pct"]

    is_macro_breakout = (
        current_close > tl_now and
        prev_close <= tl_prev and
        breakout_pct > 0 and
        breakout_pct <= breakout_max_pct
    )

    if is_macro_breakout:
        macro_note = "makro kırılım var"
    elif breakout_pct > breakout_max_pct:
        macro_note = "geç kalmış makro breakout"
    else:
        macro_note = "makro kırılım yok"

    return {
        "is_macro_breakout": is_macro_breakout,
        "macro_trendline_value": round(tl_now, 6),
        "macro_breakout_pct": round(breakout_pct, 2),
        "macro_note": macro_note,
        "macro_debug": (
            f"{prep_note} | touch={best_candidate['touch_count']} | "
            f"viol={best_candidate['violations']} | p1={best_candidate['x1']},p2={best_candidate['x2']}"
        ),
        "macro_input_bars": len(local_df),
    }

# =========================================================
# REASON / RISK
# =========================================================
def build_reason_and_risk(setup, score, above_ema20, above_ema50, above_ema200, bullish_alignment,
                          rsi, rel_volume, rs_vs_btc, market_phase, atr_pct):
    reasons = []
    risks = []

    if above_ema200:
        reasons.append("EMA200 üstü")
    else:
        risks.append("EMA200 altı")

    if bullish_alignment:
        reasons.append("trend hizalı")

    if above_ema50:
        reasons.append("EMA50 üstü")
    else:
        risks.append("EMA50 zayıf")

    if 46 <= rsi <= 60:
        reasons.append("RSI dengeli")
    elif rsi > 68:
        risks.append("RSI uzamış")
    elif rsi < 40:
        risks.append("RSI zayıf")

    if rel_volume >= 1.15:
        reasons.append("hacim destekli")
    elif rel_volume < 0.8:
        risks.append("hacim zayıf")

    if rs_vs_btc > 1:
        reasons.append("BTC'ye göre güçlü")
    elif rs_vs_btc < -2:
        risks.append("BTC'ye göre zayıf")

    if setup == "Breakout":
        reasons.append("kırılım yakın")
    elif setup == "Pullback":
        reasons.append("geri çekilme alanı")
    elif setup == "Early Setup":
        reasons.append("erken yapı")
    elif setup == "Range":
        risks.append("range yapı")
    elif setup == "Weak":
        risks.append("zayıf yapı")

    if market_phase == "Risk On":
        reasons.append("market destekli")
    elif market_phase == "Risk Off":
        risks.append("market risk-off")

    if atr_pct > 8:
        risks.append("oynaklık yüksek")

    if score >= 75:
        reasons.append("yüksek kalite")
    elif score < 50:
        risks.append("kalite düşük")

    return " + ".join(reasons[:3]) if reasons else "nötr yapı", " + ".join(risks[:3]) if risks else "belirgin risk yok"

# =========================================================
# ANALYZE
# =========================================================
def analyze_symbol(symbol: str, raw_df: pd.DataFrame, ind_df: pd.DataFrame, mode_value: str, btc_context, data_source: str):
    last = ind_df.iloc[-1]
    price = float(last["close"])
    prev_close = float(ind_df.iloc[-2]["close"])

    if price <= 0 or prev_close <= 0:
        raise ValueError(f"invalid price data: {symbol} | price={price} prev_close={prev_close}")

    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    ema100 = float(last["EMA100"])
    ema200 = float(last["EMA200"])
    rsi = float(last["RSI"])
    atr = float(last["ATR"])
    volume = float(last["volume"])
    vol_ma20 = float(last["VOL_MA20"]) if float(last["VOL_MA20"]) > 0 else 1.0

    rel_volume = volume / vol_ma20
    atr_pct = (atr / price) * 100 if price else 0.0
    price_change_pct = ((price / prev_close) - 1) * 100 if prev_close else 0.0

    above_ema20 = price > ema20
    above_ema50 = price > ema50
    above_ema100 = price > ema100
    above_ema200 = price > ema200

    bullish_alignment = ema20 > ema50 > ema100 > ema200
    rolling_high = float(ind_df["high"].shift(1).rolling(20).max().iloc[-1])
    near_breakout = price >= rolling_high * 0.99 if rolling_high > 0 else False
    close_to_ema20 = abs(price - ema20) / price <= 0.03 if price else False
    close_to_ema50 = abs(price - ema50) / price <= 0.04 if price else False

    if near_breakout and rel_volume >= 1.2 and rsi >= 54 and above_ema50:
        setup = "Breakout"
    elif bullish_alignment and above_ema200 and rsi >= 54:
        setup = "Strong Trend"
    elif above_ema50 and (close_to_ema20 or close_to_ema50) and 43 <= rsi <= 58:
        setup = "Pullback"
    elif above_ema20 and above_ema50 and not above_ema200 and 40 <= rsi <= 56 and rel_volume >= 0.95:
        setup = "Early Setup"
    elif price < ema50 or rsi < 40:
        setup = "Weak"
    else:
        setup = "Range"

    rs_vs_btc = 0.0
    btc_trend = "N/A"
    market_phase = "Unknown"

    if btc_context is not None:
        rs_vs_btc = compute_rs_vs_btc(ind_df, btc_context["ind_df"])
        btc_trend = btc_context["trend"]
        market_phase = btc_context["market_phase"]

    score = 0
    if above_ema200:
        score += 14
    if above_ema100:
        score += 8
    if above_ema50:
        score += 8
    if above_ema20:
        score += 5
    if bullish_alignment:
        score += 8

    if 46 <= rsi <= 60:
        score += 10
    elif 60 < rsi <= 68:
        score += 6
    elif 40 <= rsi < 46:
        score += 5
    elif rsi < 36:
        score -= 8
    elif rsi > 74:
        score -= 4

    if data_source != "coingecko":
        if rel_volume >= 1.45:
            score += 10
        elif rel_volume >= 1.15:
            score += 6
        elif rel_volume >= 0.95:
            score += 2
        elif rel_volume < 0.7:
            score -= 5

    if 1.2 <= atr_pct <= 5.8:
        score += 5
    elif atr_pct > 9:
        score -= 6

    if setup == "Breakout":
        score += 14
    elif setup == "Strong Trend":
        score += 10
    elif setup == "Pullback":
        score += 10
    elif setup == "Early Setup":
        score += 9
    elif setup == "Range":
        score -= 3
    elif setup == "Weak":
        score -= 14

    if btc_context is not None:
        if market_phase == "Risk On":
            score += 7
        elif market_phase == "Neutral+":
            score += 4
        elif market_phase == "Risk Off":
            score -= 12
        else:
            score -= 2

        if rs_vs_btc > 4:
            score += 8
        elif rs_vs_btc > 1:
            score += 4
        elif rs_vs_btc < -4:
            score -= 8
        elif rs_vs_btc < -1:
            score -= 3

    if mode_value == "Spot" and not above_ema200:
        score -= 10

    trend_breakout = {
        "is_breakout": False,
        "trendline_value": None,
        "breakout_pct": None,
        "trend_note": "modül kapalı",
        "trend_debug": "disabled",
        "trend_input_bars": 0,
    }

    if trendline_module:
        trend_breakout = compute_desc_trendline_breakout(
            raw_df=raw_df,
            lookback=trend_lookback,
            pivot_window_value=pivot_window,
            tolerance_pct=breakout_tolerance_pct,
        )

    micro_breakout_pct = trend_breakout["breakout_pct"]
    if micro_breakout_pct is None:
        trend_breakout_flag = False
        breakout_quality = ""
    elif micro_breakout_pct <= 0:
        trend_breakout_flag = False
        breakout_quality = "NO"
    elif micro_breakout_pct < 1:
        trend_breakout_flag = bool(trend_breakout["is_breakout"])
        breakout_quality = "🔥 EARLY"
    elif micro_breakout_pct < 3:
        trend_breakout_flag = bool(trend_breakout["is_breakout"])
        breakout_quality = "✅ IDEAL"
    elif micro_breakout_pct < 5:
        trend_breakout_flag = bool(trend_breakout["is_breakout"])
        breakout_quality = "⚠️ LATE"
    else:
        trend_breakout_flag = False
        breakout_quality = "❌ TRASH"

    if trend_breakout_flag:
        if micro_breakout_pct < 3:
            score += 12
        elif micro_breakout_pct < 5:
            score += 5
    elif trend_breakout["trend_note"] == "kırılım yok":
        if micro_breakout_pct is not None and micro_breakout_pct > -1.5:
            score += 2

    macro_breakout = {
        "is_macro_breakout": False,
        "macro_trendline_value": None,
        "macro_breakout_pct": None,
        "macro_note": "makro modül kapalı",
        "macro_debug": "disabled",
        "macro_input_bars": 0,
    }

    if macro_module:
        macro_raw_df, _, _, _ = fetch_klines(symbol, MACRO_TIMEFRAME)
        if not macro_raw_df.empty:
            macro_breakout = compute_macro_trend_breakout(
                raw_df=macro_raw_df,
                lookback=macro_lookback,
                pivot_window_value=macro_pivot_window,
                breakout_max_pct=macro_breakout_max_pct,
            )

    if macro_breakout["is_macro_breakout"]:
        score += 14
    elif macro_breakout["macro_note"] == "makro kırılım yok":
        if macro_breakout["macro_breakout_pct"] is not None and -2 <= macro_breakout["macro_breakout_pct"] <= 0:
            score += 3

    fake_breakout_risk = False
    fake_breakout_note = ""
    if trend_breakout["breakout_pct"] is not None and data_source != "coingecko":
        if trend_breakout["breakout_pct"] > 4.5 and rel_volume < 1.0:
            fake_breakout_risk = True
            fake_breakout_note = "zayıf hacimli geç kırılım"
        elif rsi > 72 and rel_volume < 1.1:
            fake_breakout_risk = True
            fake_breakout_note = "RSI uzamış, hacim teyidi zayıf"
        elif price_change_pct > 6 and rel_volume < 1.0:
            fake_breakout_risk = True
            fake_breakout_note = "sert fiyat artışı, hacim teyidi yok"

    if fake_breakout_risk:
        score -= 10

    macro_proximity_score = 0
    macro_proximity_note = "makro trend yok"
    if macro_breakout["macro_trendline_value"] is not None and price > 0:
        macro_distance_pct = ((price / macro_breakout["macro_trendline_value"]) - 1) * 100
        if -1.0 <= macro_distance_pct <= 1.5:
            macro_proximity_score = 10
            macro_proximity_note = "makro çizgiye çok yakın"
        elif -2.0 <= macro_distance_pct <= 3.0:
            macro_proximity_score = 6
            macro_proximity_note = "makro çizgiye yakın"
        elif -4.0 <= macro_distance_pct <= 5.0:
            macro_proximity_score = 3
            macro_proximity_note = "makro bölge içinde"

    score += macro_proximity_score

    watchlist = (
        50 <= score < 65 and
        setup in ["Pullback", "Early Setup", "Strong Trend"] and
        rs_vs_btc > -2.5
    )
    early_entry = (
        44 <= score < 68 and
        setup in ["Early Setup", "Pullback", "Range", "Strong Trend"] and
        rsi >= 42 and
        rs_vs_btc >= -4
    )
    momentum = (score >= 50 and rsi >= 50 and price_change_pct >= -0.2)

    combined_setup_score = 0
    combined_setup_note = "normal"
    if trend_breakout_flag and momentum and macro_proximity_score >= 6:
        combined_setup_score = 12
        combined_setup_note = "mikro + momentum + makro yakınlık"
    elif trend_breakout_flag and macro_breakout["is_macro_breakout"]:
        combined_setup_score = 15
        combined_setup_note = "mikro + makro çift teyit"
    elif early_entry and macro_proximity_score >= 6:
        combined_setup_score = 8
        combined_setup_note = "erken giriş + makro yakınlık"
    elif momentum and macro_proximity_score >= 6:
        combined_setup_score = 6
        combined_setup_note = "momentum + makro yakınlık"

    score += combined_setup_score

    breakout_quality_score = 0.0
    if breakout_quality == "✅ IDEAL":
        breakout_quality_score = 1.0
    elif breakout_quality == "🔥 EARLY":
        breakout_quality_score = 0.8
    elif breakout_quality == "⚠️ LATE":
        breakout_quality_score = 0.45
    elif breakout_quality == "NO":
        breakout_quality_score = 0.1
    elif breakout_quality == "❌ TRASH":
        breakout_quality_score = 0.0

    trend_strength_score = 0.0
    if setup == "Strong Trend":
        trend_strength_score = 1.0
    elif setup == "Breakout":
        trend_strength_score = 0.85
    elif setup == "Pullback":
        trend_strength_score = 0.7
    elif setup == "Early Setup":
        trend_strength_score = 0.55
    elif setup == "Range":
        trend_strength_score = 0.3
    else:
        trend_strength_score = 0.1

    momentum_score = 1.0 if momentum else 0.0

    if data_source == "coingecko":
        volume_confirmation = 0.5
    else:
        if rel_volume >= 1.5:
            volume_confirmation = 1.0
        elif rel_volume >= 1.15:
            volume_confirmation = 0.8
        elif rel_volume >= 0.95:
            volume_confirmation = 0.55
        else:
            volume_confirmation = 0.2

    confidence = (
        breakout_quality_score * 0.4 +
        trend_strength_score * 0.3 +
        momentum_score * 0.2 +
        volume_confirmation * 0.1
    )

    score = max(0, min(100, int(round(score))))

    if score >= 75 and setup in ["Breakout", "Strong Trend"]:
        tier = "Tier 3"
    elif score >= 65 and setup in ["Strong Trend", "Pullback", "Breakout"]:
        tier = "Tier 2"
    elif score >= 48 and setup in ["Pullback", "Early Setup"]:
        tier = "Tier 1"
    else:
        tier = "Tier 0"

    if mode_value == "Swing":
        if score >= 80 and (trend_breakout_flag or macro_breakout["is_macro_breakout"]):
            action = "SWING ADAYI"
        elif score >= 65 and setup in ["Pullback", "Strong Trend", "Breakout"]:
            action = "İZLE"
        elif early_entry:
            action = "ERKEN RADAR"
        elif watchlist:
            action = "WATCHLIST"
        else:
            action = "BEKLE"
    else:
        if score >= 72 and setup in ["Strong Trend", "Breakout"]:
            action = "AL / İZLE"
        elif score >= 60 and setup == "Pullback":
            action = "İZLE"
        else:
            action = "BEKLE"

    if combined_setup_score >= 10 and not fake_breakout_risk:
        action = "A+ SETUP"
    elif fake_breakout_risk and action in ["SWING ADAYI", "A+ SETUP"]:
        action = "TEMKİNLİ İZLE"

    stop = max(price - (atr * 1.5), price * 0.90)
    entry_trigger = rolling_high * 1.002 if setup == "Breakout" else ema20
    breakout_trigger = rolling_high * 1.002 if rolling_high > 0 else price

    risk_amount = capital * (risk_per_trade / 100)
    unit_risk = max(price - stop, max(price * 0.005, 0.00000001))
    position_size = risk_amount / unit_risk
    position_value = position_size * price

    reason_text, risk_text = build_reason_and_risk(
        setup=setup,
        score=score,
        above_ema20=above_ema20,
        above_ema50=above_ema50,
        above_ema200=above_ema200,
        bullish_alignment=bullish_alignment,
        rsi=rsi,
        rel_volume=rel_volume if data_source != "coingecko" else 1.0,
        rs_vs_btc=rs_vs_btc,
        market_phase=market_phase,
        atr_pct=atr_pct,
    )

    if macro_proximity_score >= 6:
        reason_text = f"{reason_text} + makro çizgi yakınlığı"
    if combined_setup_score >= 10:
        reason_text = f"{reason_text} + yüksek birleşik kalite"
    if fake_breakout_risk:
        risk_text = f"{risk_text} + fake breakout riski"
    if trend_breakout_flag:
        reason_text = f"{reason_text} + mikro trend kırılımı"
    if macro_breakout["is_macro_breakout"]:
        reason_text = f"{reason_text} + makro trend kırılımı"

    return {
        "Price": round(price, 6),
        "Price Change %": round(price_change_pct, 2),
        "RSI": round(rsi, 2),
        "ATR %": round(atr_pct, 2),
        "Rel Volume": round(rel_volume, 2) if data_source != "coingecko" else None,
        "RS vs BTC": round(rs_vs_btc, 2),
        "BTC Trend": btc_trend,
        "Market Phase": market_phase,
        "Setup": setup,
        "Score": score,
        "Tier": tier,
        "Entry Trigger": round(entry_trigger, 6),
        "Breakout Trigger": round(breakout_trigger, 6),
        "Stop Trigger": round(stop, 6),
        "Action": action,
        "Watchlist": "👀" if watchlist else "",
        "Early Entry": "🟣" if early_entry else "",
        "Momentum": "🔥" if momentum else "",
        "Trend Breakout": "🎈" if trend_breakout_flag else "",
        "Breakout Quality": breakout_quality,
        "Trendline Value": trend_breakout["trendline_value"],
        "Breakout %": trend_breakout["breakout_pct"],
        "Trend Note": trend_breakout["trend_note"],
        "Trend Debug": trend_breakout["trend_debug"],
        "Trend Input Bars": trend_breakout["trend_input_bars"],
        "Macro Breakout": "🚀" if macro_breakout["is_macro_breakout"] else "",
        "Macro Trendline Value": macro_breakout["macro_trendline_value"],
        "Macro Breakout %": macro_breakout["macro_breakout_pct"],
        "Macro Note": macro_breakout["macro_note"],
        "Macro Debug": macro_breakout["macro_debug"],
        "Macro Input Bars": macro_breakout["macro_input_bars"],
        "Fake Breakout Risk": "⚠️" if fake_breakout_risk else "",
        "Fake Breakout Note": fake_breakout_note,
        "Macro Proximity Score": macro_proximity_score,
        "Macro Proximity Note": macro_proximity_note,
        "Combined Setup Score": combined_setup_score,
        "Combined Setup Note": combined_setup_note,
        "Confidence": round(confidence, 3),
        "Breakout Quality Score": round(breakout_quality_score, 3),
        "Trend Strength Score": round(trend_strength_score, 3),
        "Momentum Score": round(momentum_score, 3),
        "Volume Confirmation": round(volume_confirmation, 3),
        "Raw Bars": len(raw_df),
        "Indicator Bars": len(ind_df),
        "Position Value": round(position_value, 2),
        "Reason": reason_text,
        "Risk Flag": risk_text,
        "Data Source": data_source,
        "Drop Reason": "",
    }

# =========================================================
# RENDER
# =========================================================
def render_metric_row(data: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", data["Score"])
    c2.metric("RSI", data["RSI"])
    c3.metric("Setup", data["Setup"])
    c4.metric("Action", data["Action"])

def render_main_title():
    st.title("🛡️ CryptoOS")
    st.caption(APP_VERSION)

def render_table(df_input: pd.DataFrame, title: str):
    st.subheader(title)
    if df_input.empty:
        st.info("Sonuç yok.")
        return
    st.dataframe(df_input, width="stretch")

# =========================================================
# SINGLE
# =========================================================
def run_single():
    render_main_title()

    health = provider_health_check()
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("OKX", "UP" if health["okx"]["ok"] else "DOWN")
    h2.metric("Gate", "UP" if health["gate"]["ok"] else "DOWN")
    h3.metric("Coinbase", "UP" if health["coinbase"]["ok"] else "DOWN")
    h4.metric("CoinGecko", "UP" if health["coingecko"]["ok"] else "DOWN")

    symbol = normalize_symbol(symbol_input)

    raw_df, err, source, provider_errors = fetch_klines(symbol, timeframe)
    if raw_df.empty:
        st.error(f"Veri alınamadı: {err}")
        if provider_errors:
            st.write("Provider errors:", provider_errors)
        return

    if source != "okx":
        st.warning(f"Primary provider yerine fallback kullanıldı: {source}")

    ind_df = add_indicators(raw_df)
    if ind_df.empty:
        st.error("Gösterge hesaplanamadı.")
        return

    btc_context = compute_btc_context(timeframe)
    result = analyze_symbol(symbol, raw_df, ind_df, mode, btc_context, source)

    render_metric_row(result)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fiyat", result["Price"])
    c2.metric("Pozisyon Değeri ($)", result["Position Value"])
    c3.metric("RS vs BTC", result["RS vs BTC"])
    c4.metric("Confidence", result["Confidence"])

    st.markdown("---")
    st.subheader("Analiz Özeti")
    st.dataframe(pd.DataFrame([result]), width="stretch")

# =========================================================
# WORKER
# =========================================================
def process_symbol(row_dict, timeframe_key, mode_value, btc_context):
    symbol = row_dict["symbol"]
    name = row_dict["name"]
    rank = row_dict["rank"]

    raw_df, err, source, provider_errors = fetch_klines(symbol, timeframe_key)
    if raw_df.empty:
        return {
            "Rank": rank,
            "Name": name,
            "Symbol": symbol,
            "Drop Reason": f"{source} | {err[:200]}" if err else f"{source} | no_data",
            "Provider Errors": " | ".join(provider_errors[:4]) if provider_errors else "",
            "kind": "dropped",
        }

    ind_df = add_indicators(raw_df)
    if ind_df.empty:
        return {
            "Rank": rank,
            "Name": name,
            "Symbol": symbol,
            "Raw Bars": len(raw_df),
            "Indicator Bars": 0,
            "Drop Reason": f"{source} | indicator_failed",
            "Provider Errors": " | ".join(provider_errors[:4]) if provider_errors else "",
            "kind": "dropped",
        }

    try:
        result = analyze_symbol(symbol, raw_df, ind_df, mode_value, btc_context, source)
        return {
            "Rank": rank,
            "Name": name,
            "Symbol": symbol,
            **result,
            "Provider Errors": " | ".join(provider_errors[:4]) if provider_errors else "",
            "kind": "analyzed",
        }
    except Exception as e:
        return {
            "Rank": rank,
            "Name": name,
            "Symbol": symbol,
            "Raw Bars": len(raw_df),
            "Indicator Bars": len(ind_df),
            "Drop Reason": f"{source} | {str(e)[:200]}",
            "Provider Errors": " | ".join(provider_errors[:4]) if provider_errors else "",
            "kind": "dropped",
        }

# =========================================================
# SCREENER
# =========================================================
def run_screener():
    render_main_title()

    health = provider_health_check()
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("OKX", "UP" if health["okx"]["ok"] else "DOWN")
    h2.metric("Gate", "UP" if health["gate"]["ok"] else "DOWN")
    h3.metric("Coinbase", "UP" if health["coinbase"]["ok"] else "DOWN")
    h4.metric("CoinGecko", "UP" if health["coingecko"]["ok"] else "DOWN")

    universe_df = build_custom_universe()
    btc_context = compute_btc_context(timeframe)

    if universe_df.empty:
        st.error("Evren üretilemedi.")
        return

    coverage = {
        "Target Universe": len(universe_df),
        "Symbol Match": len(universe_df),
        "Data Loaded": 0,
        "Analyzed": 0,
        "Dropped": 0,
        "Universe Source": "custom",
    }

    rows = []
    matched_rows = [row.to_dict() for _, row in universe_df.iterrows()]

    progress = st.progress(0)
    total = len(matched_rows) if matched_rows else 1

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_symbol, row_dict, timeframe, mode, btc_context) for row_dict in matched_rows]

        completed = 0
        for future in as_completed(futures):
            completed += 1
            progress.progress(completed / total)
            item = future.result()

            if item["kind"] == "dropped":
                coverage["Dropped"] += 1
                rows.append(item)
            else:
                coverage["Data Loaded"] += 1
                coverage["Analyzed"] += 1
                rows.append(item)

    progress.progress(1.0)
    progress.empty()

    df_results = pd.DataFrame(rows)
    analyzed_df = df_results[df_results["kind"].eq("analyzed")].copy() if "kind" in df_results.columns else pd.DataFrame()
    dropped_df = df_results[df_results["kind"].ne("analyzed")].copy() if "kind" in df_results.columns else df_results.copy()

    if not analyzed_df.empty:
        analyzed_df = analyzed_df[
            (analyzed_df["Score"] >= min_score) &
            (analyzed_df["Confidence"] >= 0.65)
        ].sort_values(["Confidence", "Score"], ascending=[False, False])

    st.markdown("---")
    st.subheader("🛰️ Coverage Diagnostics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Hedef Evren", coverage["Target Universe"])
    c2.metric("Symbol Match", coverage["Symbol Match"])
    c3.metric("Veri Gelen", coverage["Data Loaded"])
    c4.metric("Analiz Edilen", coverage["Analyzed"])
    c5.metric("Elenen", coverage["Dropped"])
    c6.metric("Kaynak", coverage["Universe Source"])

    if btc_context is not None:
        st.write(
            f"**BTC Trend:** {btc_context['trend']} | "
            f"**BTC RSI:** {btc_context['rsi']:.2f} | "
            f"**Market Phase:** {btc_context['market_phase']}"
        )

    st.markdown("---")
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Filtre Sonrası Coin", len(analyzed_df) if not analyzed_df.empty else 0)
    top2.metric("En Yüksek Skor", int(analyzed_df["Score"].max()) if not analyzed_df.empty else 0)
    top3.metric("Shortlist", int((analyzed_df["Score"] >= 65).sum()) if not analyzed_df.empty else 0)
    top4.metric("Watchlist", int((analyzed_df["Watchlist"] == "👀").sum()) if not analyzed_df.empty else 0)

    top_trades = pd.DataFrame()
    if not analyzed_df.empty:
        top_trades = analyzed_df.sort_values(
            ["Confidence", "Score"],
            ascending=[False, False]
        ).head(5)

    render_table(top_trades.drop(columns=["kind"], errors="ignore"), "🔥 Top Trade Setups")
    render_table(analyzed_df.drop(columns=["kind"], errors="ignore"), "📋 Ana Sonuçlar")

    if not analyzed_df.empty:
        render_table(
            analyzed_df[analyzed_df["Score"] >= 65].drop(columns=["kind"], errors="ignore"),
            "🔥 Shortlist (65+)"
        )

        render_table(
            analyzed_df[analyzed_df["Watchlist"] == "👀"].drop(columns=["kind"], errors="ignore"),
            "👀 Watchlist"
        )

        render_table(
            analyzed_df[analyzed_df["Early Entry"] == "🟣"].drop(columns=["kind"], errors="ignore"),
            "🟣 Early Entry Radar"
        )

        render_table(
            analyzed_df[analyzed_df["Momentum"] == "🔥"].drop(columns=["kind"], errors="ignore"),
            "🔥 Momentum Patlaması"
        )

        high_quality_combo_df = analyzed_df[analyzed_df["Combined Setup Score"] >= 10].copy()
        render_table(
            high_quality_combo_df.drop(columns=["kind"], errors="ignore"),
            "🧠 Yüksek Kalite Kombinasyon"
        )

        micro_breakout_df = analyzed_df[
            (analyzed_df["Trend Breakout"] == "🎈") &
            (analyzed_df["Breakout %"].notna()) &
            (analyzed_df["Breakout %"] > 0) &
            (analyzed_df["Breakout %"] < 5)
        ].copy()
        render_table(
            micro_breakout_df.drop(columns=["kind"], errors="ignore"),
            "🎈 Mikro Trend Kırılımı"
        )

        macro_breakout_df = analyzed_df[
            (analyzed_df["Macro Breakout %"].notna()) &
            (analyzed_df["Macro Breakout %"] > 0.5) &
            (analyzed_df["Macro Input Bars"] >= 100)
        ].copy()
        render_table(
            macro_breakout_df.drop(columns=["kind"], errors="ignore"),
            "🚀 Makro Trend Kırılımı"
        )

        fake_breakout_df = analyzed_df[
            analyzed_df["Fake Breakout Risk"] == "⚠️"
        ].copy()
        render_table(
            fake_breakout_df.drop(columns=["kind"], errors="ignore"),
            "⚠️ Fake Breakout Riski"
        )

    if show_trend_debug and not analyzed_df.empty:
        render_table(
            analyzed_df[[
                "Name", "Symbol", "Data Source",
                "Raw Bars", "Indicator Bars", "Trend Input Bars",
                "Trend Note", "Trendline Value", "Breakout %", "Trend Debug",
                "Macro Input Bars", "Macro Note", "Macro Trendline Value", "Macro Breakout %", "Macro Debug",
                "Fake Breakout Risk", "Fake Breakout Note",
                "Macro Proximity Score", "Macro Proximity Note",
                "Combined Setup Score", "Combined Setup Note",
                "Confidence", "Breakout Quality Score", "Trend Strength Score",
                "Momentum Score", "Volume Confirmation",
                "Provider Errors",
            ]].head(30),
            "🧪 Trend Debug"
        )

    if not dropped_df.empty:
        render_table(
            dropped_df[[c for c in [
                "Rank", "Name", "Symbol", "Raw Bars", "Indicator Bars", "Drop Reason", "Provider Errors"
            ] if c in dropped_df.columns]],
            "🗑️ Elenen / Veri Alınamayanlar"
        )

# =========================================================
# EXECUTION
# =========================================================
if view_mode == "Tekil Analiz":
    run_single()
else:
    run_screener()
