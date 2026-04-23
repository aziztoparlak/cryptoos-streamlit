import os
import time
import json
import requests
import pandas as pd

REQUEST_HEADERS = {
    "accept": "application/json",
    "user-agent": "Mozilla/5.0",
}

CACHE_DIR = "cache"

TIMEFRAME_CONFIG = {
    "4h": {"bars": 400, "okx": "4H", "gate": "4h", "coinbase": 14400, "coingecko_days": 90},
    "1d": {"bars": 400, "okx": "1D", "gate": "1d", "coinbase": 86400, "coingecko_days": 365},
    "1w": {"bars": 400, "okx": "1W", "gate": "7d", "coinbase": 86400, "coingecko_days": 365},
    "1M": {"bars": 400, "okx": "1M", "gate": "30d", "coinbase": 86400, "coingecko_days": 365},
}

COINGECKO_ID_OVERRIDES = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "XRPUSDT": "ripple",
    "BNBUSDT": "binancecoin",
    "SOLUSDT": "solana",
    "TRXUSDT": "tron",
    "DOGEUSDT": "dogecoin",
    "ADAUSDT": "cardano",
    "BCHUSDT": "bitcoin-cash",
    "LINKUSDT": "chainlink",
    "XMRUSDT": "monero",
    "XLMUSDT": "stellar",
    "LTCUSDT": "litecoin",
    "AVAXUSDT": "avalanche-2",
    "HBARUSDT": "hedera-hashgraph",
    "SUIUSDT": "sui",
    "SHIBUSDT": "shiba-inu",
    "TONUSDT": "the-open-network",
    "DOTUSDT": "polkadot",
    "UNIUSDT": "uniswap",
    "NEARUSDT": "near",
    "APTUSDT": "aptos",
    "PEPEUSDT": "pepe",
    "OPUSDT": "optimism",
    "ARBUSDT": "arbitrum",
    "KASUSDT": "kaspa",
    "ICPUSDT": "internet-computer",
    "STXUSDT": "blockstack",
    "RENDERUSDT": "render-token",
    "INJUSDT": "injective-protocol",
    "IMXUSDT": "immutable-x",
    "GRTUSDT": "the-graph",
    "FTMUSDT": "fantom",
    "RUNEUSDT": "thorchain",
    "ARUSDT": "arweave",
    "SEIUSDT": "sei-network",
    "ALGOUSDT": "algorand",
    "QNTUSDT": "quant-network",
    "MNTUSDT": "mantle",
    "FLOWUSDT": "flow",
    "AAVEUSDT": "aave",
    "LDOUSDT": "lido-dao",
    "THETAUSDT": "theta-token",
    "POLUSDT": "polygon-ecosystem-token",
    "PYTHUSDT": "pyth-network",
    "JUPUSDT": "jupiter-exchange-solana",
    "BONKUSDT": "bonk",
    "TIAUSDT": "celestia",
    "ONDOUSDT": "ondo-finance",
    "MKRUSDT": "maker",
    "JASMYUSDT": "jasmycoin",
    "GALAUSDT": "gala",
    "ENAUSDT": "ethena",
    "WLDUSDT": "worldcoin-wld",
    "SNXUSDT": "havven",
    "EOSUSDT": "eos",
    "FLRUSDT": "flare-networks",
    "CRVUSDT": "curve-dao-token",
    "RPLUSDT": "rocket-pool",
    "MINAUSDT": "mina-protocol",
    "XTZUSDT": "tezos",
    "AXSUSDT": "axie-infinity",
    "CFXUSDT": "conflux-token",
    "CHZUSDT": "chiliz",
    "NEOUSDT": "neo",
    "CAKEUSDT": "pancakeswap-token",
    "SANDUSDT": "the-sandbox",
    "MANAUSDT": "decentraland",
    "AKTUSDT": "akash-network",
    "BGBUSDT": "bitget-token-new",
    "STRKUSDT": "starknet",
    "DYMUSDT": "dymension",
    "AEVOUSDT": "aevo-exchange",
    "SATSUSDT": "sats-ordinals",
    "CKBUSDT": "nervos-network",
    "IOTAUSDT": "iota",
    "HNTUSDT": "helium",
    "RONINUSDT": "ronin",
    "WEMIXUSDT": "wemix-token",
    "BLURUSDT": "blur",
    "ETHFIUSDT": "ether-fi",
    "METISUSDT": "metis-token",
    "FXSUSDT": "frax-share",
    "ZILUSDT": "zilliqa",
    "GASUSDT": "gas",
    "GNOUSDT": "gnosis",
    "KAVAUSDT": "kava",
    "ASTRUSDT": "astar",
    "BEAMUSDT": "beam-2",
    "LPTUSDT": "livepeer",
    "HYPEUSDT": "hyperliquid",
    "LEOUSDT": "leo-token",
    "TAOUSDT": "bittensor",
    "CCUSDT": "canton",
    "MUSDT": "meme-core",
    "AGIXUSDT": "singularitynet",
}

PROVIDER_PRIORITY = ["okx", "gate", "coinbase", "coingecko"]

def _session():
    s = requests.Session()
    s.trust_env = False
    return s

def _safe_get_json(url: str, params=None, timeout: int = 15):
    try:
        s = _session()
        r = s.get(url, params=params, timeout=timeout, headers=REQUEST_HEADERS)
        if r.status_code != 200:
            return None, f"{url} -> HTTP {r.status_code}"
        return r.json(), ""
    except Exception as e:
        return None, f"{url} -> {e}"

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].dropna().sort_values("open_time")
    return df.reset_index(drop=True)

def _cache_path(symbol: str, timeframe: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{symbol}_{timeframe}.csv")

def save_cache(df: pd.DataFrame, symbol: str, timeframe: str):
    if not df.empty:
        df.to_csv(_cache_path(symbol, timeframe), index=False)

def load_cache(symbol: str, timeframe: str):
    path = _cache_path(symbol, timeframe)
    if not os.path.exists(path):
        return pd.DataFrame(), "cache not found"
    try:
        df = pd.read_csv(path)
        df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce")
        df = _normalize_ohlcv(df)
        if df.empty:
            return pd.DataFrame(), "cache empty after normalize"
        return df, ""
    except Exception as e:
        return pd.DataFrame(), str(e)

def usdt_to_usd_dash(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    return f"{base}-USD"

def usdt_to_usd_plain(symbol: str) -> str:
    return symbol.replace("USDT", "-USD")

def usdt_to_okx_inst(symbol: str) -> str:
    return symbol.replace("USDT", "-USDT")

def get_coingecko_id(symbol: str):
    return COINGECKO_ID_OVERRIDES.get(symbol)

def fetch_okx_ohlcv(symbol: str, timeframe: str):
    cfg = TIMEFRAME_CONFIG[timeframe]
    inst_id = usdt_to_okx_inst(symbol)
    url = "https://www.okx.com/api/v5/market/history-candles"
    params = {"instId": inst_id, "bar": cfg["okx"], "limit": "400"}
    data, err = _safe_get_json(url, params=params, timeout=20)
    if data is None or "data" not in data or not data["data"]:
        return pd.DataFrame(), err or f"OKX no data: {symbol}", "okx"

    try:
        rows = []
        for item in data["data"]:
            rows.append({
                "open_time": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
            })
        df = pd.DataFrame(rows)
        df = _normalize_ohlcv(df)
        return df.tail(cfg["bars"]), "", "okx"
    except Exception as e:
        return pd.DataFrame(), str(e), "okx"

def fetch_gate_ohlcv(symbol: str, timeframe: str):
    cfg = TIMEFRAME_CONFIG[timeframe]
    pair = usdt_to_usd_plain(symbol)
    url = "https://api.gateio.ws/api/v4/spot/candlesticks"
    params = {"currency_pair": pair, "interval": cfg["gate"], "limit": 400}
    data, err = _safe_get_json(url, params=params, timeout=20)
    if data is None or not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(), err or f"Gate no data: {symbol}", "gate"

    try:
        rows = []
        for item in data:
            rows.append({
                "open_time": pd.to_datetime(int(item[0]), unit="s", utc=True),
                "volume": item[1],
                "close": item[2],
                "high": item[3],
                "low": item[4],
                "open": item[5],
            })
        df = pd.DataFrame(rows)
        df = _normalize_ohlcv(df)
        return df.tail(cfg["bars"]), "", "gate"
    except Exception as e:
        return pd.DataFrame(), str(e), "gate"

def fetch_coinbase_ohlcv(symbol: str, timeframe: str):
    cfg = TIMEFRAME_CONFIG[timeframe]
    product_id = usdt_to_usd_dash(symbol)
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {"granularity": cfg["coinbase"]}
    data, err = _safe_get_json(url, params=params, timeout=20)
    if data is None or not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(), err or f"Coinbase no data: {symbol}", "coinbase"

    try:
        rows = []
        for item in data:
            rows.append({
                "open_time": pd.to_datetime(int(item[0]), unit="s", utc=True),
                "low": item[1],
                "high": item[2],
                "open": item[3],
                "close": item[4],
                "volume": item[5],
            })
        df = pd.DataFrame(rows)
        df = _normalize_ohlcv(df)
        return df.tail(300), "", "coinbase"
    except Exception as e:
        return pd.DataFrame(), str(e), "coinbase"

def fetch_coingecko_ohlcv(symbol: str, timeframe: str):
    cfg = TIMEFRAME_CONFIG[timeframe]
    coin_id = get_coingecko_id(symbol)
    if not coin_id:
        return pd.DataFrame(), f"CoinGecko id not found: {symbol}", "coingecko"

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": cfg["coingecko_days"]}
    data, err = _safe_get_json(url, params=params, timeout=20)
    if data is None or not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(), err or f"CoinGecko no data: {symbol}", "coingecko"

    try:
        rows = []
        for item in data:
            rows.append({
                "open_time": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": 1.0,
            })
        df = pd.DataFrame(rows)
        df = _normalize_ohlcv(df)
        return df.tail(cfg["bars"]), "", "coingecko"
    except Exception as e:
        return pd.DataFrame(), str(e), "coingecko"

def provider_health_check():
    checks = {
        "okx": ("https://www.okx.com/api/v5/public/time", None),
        "gate": ("https://api.gateio.ws/api/v4/spot/currencies", None),
        "coinbase": ("https://api.exchange.coinbase.com/time", None),
        "coingecko": ("https://api.coingecko.com/api/v3/ping", None),
    }
    result = {}
    for name, (url, params) in checks.items():
        data, err = _safe_get_json(url, params=params, timeout=8)
        result[name] = {"ok": data is not None, "error": err}
    return result

def fetch_ohlcv_with_failover(symbol: str, timeframe: str):
    errors = []

    for provider in PROVIDER_PRIORITY:
        if provider == "okx":
            df, err, source = fetch_okx_ohlcv(symbol, timeframe)
        elif provider == "gate":
            df, err, source = fetch_gate_ohlcv(symbol, timeframe)
        elif provider == "coinbase":
            df, err, source = fetch_coinbase_ohlcv(symbol, timeframe)
        elif provider == "coingecko":
            df, err, source = fetch_coingecko_ohlcv(symbol, timeframe)
        else:
            continue

        if not df.empty:
            save_cache(df, symbol, timeframe)
            return df, "", source, errors

        errors.append(f"{provider}: {err}")

    cache_df, cache_err = load_cache(symbol, timeframe)
    if not cache_df.empty:
        return cache_df, "all providers failed -> cache used", "cache", errors

    return pd.DataFrame(), f"all providers failed | {' | '.join(errors)} | cache: {cache_err}", "none", errors

