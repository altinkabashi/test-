#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################
############################### IMPORTS ################################
########################################################################
import os, time, csv, logging, requests, math, warnings
from collections import deque
from datetime import datetime, timedelta, timezone      #  ← updated
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

from binance.client import Client
from binance.enums import *

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Optional Bayesian optimisation
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    _have_bayes = True
except ImportError:
    _have_bayes = False

########################################################################
######################## USER-TUNABLE CONSTANTS ########################
########################################################################
PAIRS                    = ["ARBUSDT", "ADAUSDT", "WLDUSDT", "GALAUSDT"]
TIMEFRAME                = Client.KLINE_INTERVAL_15MINUTE
HTF_INTERVAL             = Client.KLINE_INTERVAL_1HOUR      # 1-hour bias
TRADE_INTERVAL_SECONDS   = 400
TRADE_AMOUNT_PERCENTAGE  = 0.04      # 4 % equity per trade (×20 ≈ 80 % notional)
LEVERAGE                 = 20
SUMMARY_INTERVAL_SECONDS = 800

# ML / data
ML_MODEL_PATH_TEMPLATE   = "ml_model_{}.pkl"
DATA_CSV                 = "botiri2_data.csv"
TRADES_CSV               = "botiri2trades.csv"
EXTENDED_HISTORICAL_BARS = 1000
CONFIDENCE_THRESHOLD     = 0.55
RETRAIN_INTERVAL_SECONDS = 43200      # 12 h

# Risk management
SL_ATR_MULT        = 1.0
TP1_ATR_MULT       = 1.5
TP2_ATR_MULT       = 3.0
TRAIL_START_MULT   = TP1_ATR_MULT
TRAIL_INCREMENT_ATR= 0.5

# Filters
ADX_MIN            = 25

########################################################################
############################## LOGGING #################################
########################################################################
order_errors = deque(maxlen=100)
load_dotenv("apitini.env")

api_key    = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client     = Client(api_key, api_secret)

handler = RotatingFileHandler("ANAml3.log", maxBytes=5*1024*1024,
                              backupCount=5, encoding="utf-8")
logging.basicConfig(handlers=[handler], level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

########################################################################
######################### RETRY WRAPPER ################################
########################################################################
def retry_call(func, *args, **kwargs):
    """Call `func` with retries -- up to 5 attempts, 60 s wait."""
    for attempt in range(5):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Retry {attempt+1}/5 for {func.__name__}: {e}")
            if attempt < 4:
                time.sleep(60)
            else:
                raise

########################################################################
############### PRECISION / STEP-SIZE TABLES (ONE-OFF) ################
########################################################################
_exchange_info = retry_call(client.futures_exchange_info)

_QTY_PREC   = {s["symbol"]: int(s["quantityPrecision"])
               for s in _exchange_info["symbols"]}
_PRICE_PREC = {s["symbol"]: int(s["pricePrecision"])
               for s in _exchange_info["symbols"]}

_STEP_SIZE = {}
_TICK_SIZE = {}
for s in _exchange_info["symbols"]:
    for f in s["filters"]:
        if f["filterType"] == "LOT_SIZE":
            _STEP_SIZE[s["symbol"]] = float(f["stepSize"])
        elif f["filterType"] == "PRICE_FILTER":
            _TICK_SIZE[s["symbol"]] = float(f["tickSize"])

def get_asset_precision(sym):  return _QTY_PREC.get(sym, 8)
def get_price_precision(sym):  return _PRICE_PREC.get(sym, 8)

def _round_step(value, step, precision, *, up=False):
    if step == 0:  # fallback
        return round(value, precision)
    return round((math.ceil if up else math.floor)(value/step) * step, precision)

def adjust_qty(qty, pair, *, up=False):
    return _round_step(qty, _STEP_SIZE.get(pair, 1.0),
                       get_asset_precision(pair), up=up)

def adjust_price(price, pair, *, up=False):
    return _round_step(price, _TICK_SIZE.get(pair, 0.0001),
                       get_price_precision(pair), up=up)

########################################################################
######################### GLOBAL STATE HOLDERS #########################
########################################################################
trade_active       = {p: False for p in PAIRS}
trade_side         = {p: None  for p in PAIRS}
trade_entry_price  = {}
trade_open_time    = {}
sl_moved           = {p: False for p in PAIRS}
tp1_price_dict     = {}
tp2_price_dict     = {}
trailing_stop_data = {}
last_summary_time  = time.time()

########################################################################
############################ TELEGRAM ##################################
########################################################################
def send_telegram_message(msg, mtype="Info"):
    bot, cid = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not bot or not cid:
        logging.error("Telegram credentials missing")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{bot}/sendMessage",
                          json={"chat_id": cid, "text": msg, "parse_mode":"HTML"},
                          timeout=10)
        if r.status_code != 200:
            logging.error(f"Telegram {mtype} failed: {r.text}")
    except Exception as e:
        logging.error(f"Telegram {mtype} exception: {e}")

########################################################################
##################### BALANCE / LEVERAGE HELPERS #######################
########################################################################
def get_balance(asset="USDT"):
    try:
        bal = next(b for b in retry_call(client.futures_account_balance) if b["asset"]==asset)
        return float(bal["balance"])
    except Exception:
        return 0.0

def ensure_leverage(pair):
    """Cross margin and set leverage = LEVERAGE."""
    try:
        retry_call(client.futures_change_margin_type,
                   symbol=pair, marginType="CROSSED")
    except:
        pass  # ‘No need to change …’
    try:
        retry_call(client.futures_change_leverage,
                   symbol=pair, leverage=LEVERAGE)
    except Exception as e:
        logging.warning(f"{pair}: leverage change failed ({e})")

########################################################################
############################ INDICATORS ################################
########################################################################
def calculate_ema(series, period):
    return pd.Series(series).ewm(span=period, adjust=False).mean()

def calculate_macd(close):
    c = pd.Series(close)
    ema_fast = c.ewm(span=12, adjust=False).mean()
    ema_slow = c.ewm(span=26, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_adx_and_di(high, low, close, period=14):
    df = pd.DataFrame({'high':high,'low':low,'close':close})
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low']  - df['close'].shift(1))
    ])
    df['+dm'] = np.where((df['high']-df['high'].shift(1)) >
                         (df['low'].shift(1)-df['low']),
                         np.maximum(df['high']-df['high'].shift(1),0),0)
    df['-dm'] = np.where((df['low'].shift(1)-df['low']) >
                         (df['high']-df['high'].shift(1)),
                         np.maximum(df['low'].shift(1)-df['low'],0),0)
    tr_sum   = df['tr'].rolling(period).sum()
    plus_sum = df['+dm'].rolling(period).sum()
    minus_sum= df['-dm'].rolling(period).sum()
    plus_di  = (plus_sum  / tr_sum) * 100
    minus_di = (minus_sum / tr_sum) * 100
    adx      = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).rolling(period).mean()
    return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]

def stochastic_rsi(close, rsi_period=14, stoch_period=14):
    c = pd.Series(close)
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - 100/(1+rs)
    rsi = rsi.dropna()
    if len(rsi) < stoch_period: return np.nan
    min_rsi = rsi.rolling(stoch_period).min().iloc[-1]
    max_rsi = rsi.rolling(stoch_period).max().iloc[-1]
    if max_rsi - min_rsi == 0: return np.nan
    return (rsi.iloc[-1] - min_rsi) / (max_rsi - min_rsi) * 100

def calculate_atr(data_dict, period=14):
    h, l, c = data_dict['high'], data_dict['low'], data_dict['close']
    tr  = np.maximum.reduce([h - l, abs(h - np.roll(c,1)), abs(l - np.roll(c,1))])
    atr = pd.Series(tr).rolling(period).mean()
    return atr.iloc[-1]

def calculate_supertrend(high, low, close, period=10, mult=3):
    hl2 = (high + low) / 2
    atr = pd.Series(np.maximum.reduce([high - low,
                                       abs(high - np.roll(close,1)),
                                       abs(low  - np.roll(close,1))])
                   ).rolling(period).mean()
    upperband = hl2 + mult * atr
    lowerband = hl2 - mult * atr
    final_upper, final_lower = upperband.copy(), lowerband.copy()
    trend = [1]  # 1 = bull, -1 = bear
    for i in range(1, len(close)):
        if close[i] > final_upper[i-1]:
            trend.append(1)
        elif close[i] < final_lower[i-1]:
            trend.append(-1)
        else:
            trend.append(trend[-1])
            if trend[-1] == 1 and lowerband[i] < final_lower[i-1]:
                final_lower[i] = final_lower[i-1]
            if trend[-1] == -1 and upperband[i] > final_upper[i-1]:
                final_upper[i] = final_upper[i-1]
    return trend[-1]

########################################################################
##################### DATA FETCH & STORAGE HELPERS #####################
########################################################################
def fetch_candlestick_data(pair, interval, limit=100):
    kl = retry_call(client.futures_klines, symbol=pair, interval=interval, limit=limit)
    return {
        "open" : np.array([float(k[1]) for k in kl]),
        "high" : np.array([float(k[2]) for k in kl]),
        "low"  : np.array([float(k[3]) for k in kl]),
        "close": np.array([float(k[4]) for k in kl]),
        "volume":np.array([float(k[5]) for k in kl]),
        "raw_candles": kl
    }

def store_analysis_data(pair, last_candle, indicators):
    filename   = DATA_CSV
    exists     = os.path.isfile(filename)
    open_time  = datetime.fromtimestamp(int(last_candle[0])/1000, tz=timezone.utc)\
                    .strftime('%Y-%m-%d %H:%M:%S')      #  ← updated
    row = {
        "DateTime": open_time,
        "Pair":     pair,
        "Open":     float(last_candle[1]),
        "High":     float(last_candle[2]),
        "Low":      float(last_candle[3]),
        "Close":    float(last_candle[4]),
        "Volume":   float(last_candle[5]),
        "EMA50":        indicators.get("ema50", np.nan),
        "MACD_Line":    indicators.get("macd_line", np.nan),
        "MACD_Signal":  indicators.get("macd_signal", np.nan),
        "ADX":          indicators.get("adx", np.nan),
        "PlusDI":       indicators.get("plus_di", np.nan),
        "MinusDI":      indicators.get("minus_di", np.nan),
        "StochRSI":     indicators.get("stoch_rsi", np.nan),
        "SuperDir":     indicators.get("super_dir", np.nan),
        "ATR":          indicators.get("atr", np.nan),
        "Signal":       indicators.get("signal", "NONE")
    }
    fields = list(row.keys())
    with open(filename, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)

def save_trade_to_csv(pair, side, entry, exit_, pnl, outcome):
    filename = TRADES_CSV
    exists   = os.path.isfile(filename)
    row = {
        "DateTime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "Pair":     pair,
        "Side":     side,
        "Entry Price": entry,
        "Exit Price":  exit_,
        "PNL":         pnl,
        "Outcome":     outcome
    }
    with open(filename, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)

########################################################################
######################## MARKET ANALYSIS ENGINE ########################
########################################################################
def higher_tf_ok(pair, signal):
    kl = retry_call(client.futures_klines, symbol=pair, interval=HTF_INTERVAL, limit=200)
    closes = [float(k[4]) for k in kl]
    ema200 = calculate_ema(closes, 200).iloc[-1]
    price  = closes[-1]
    return (signal == "LONG" and price > ema200) or \
           (signal == "SHORT" and price < ema200)

def analyze_market(data):
    close, high, low = data['close'], data['high'], data['low']
    if len(close) < 50:
        return {"signal": None}

    indicators = {}
    ema50 = calculate_ema(close, 50)
    indicators["ema50"] = ema50.iloc[-1]
    above50, below50 = close[-1] > ema50.iloc[-1], close[-1] < ema50.iloc[-1]

    macd_line, macd_sig = calculate_macd(close)
    indicators.update(macd_line=macd_line.iloc[-1],
                      macd_signal=macd_sig.iloc[-1])
    macd_bull, macd_bear = macd_line.iloc[-1] > macd_sig.iloc[-1], \
                           macd_line.iloc[-1] < macd_sig.iloc[-1]

    adx_val, plus_di, minus_di = calculate_adx_and_di(high, low, close, 14)
    indicators.update(adx=adx_val, plus_di=plus_di, minus_di=minus_di)
    adx_bull = adx_val >= ADX_MIN and plus_di > minus_di
    adx_bear = adx_val >= ADX_MIN and minus_di > plus_di

    stoch_val = stochastic_rsi(close)
    indicators["stoch_rsi"] = stoch_val
    stoch_bull, stoch_bear = stoch_val < 10, stoch_val > 90

    super_dir = calculate_supertrend(high, low, close)
    indicators["super_dir"] = super_dir
    super_bull, super_bear = super_dir == 1, super_dir == -1

    bull_count = sum([above50, macd_bull, adx_bull, stoch_bull, super_bull])
    bear_count = sum([below50, macd_bear, adx_bear, stoch_bear, super_bear])

    signal = None
    if bull_count >= 4 and bull_count > bear_count:
        signal = "LONG"
    elif bear_count >= 4 and bear_count > bull_count:
        signal = "SHORT"

    indicators["signal"] = signal
    indicators["bull_count"] = bull_count
    indicators["bear_count"] = bear_count
    return indicators

########################################################################
############################# ML MODULE ################################
########################################################################
def build_features_from_ohlc(df):
    df = df.copy().reset_index(drop=True)
    df["ema50"] = calculate_ema(df["c"], 50)
    macd_line, macd_signal = calculate_macd(df["c"].values)
    df["macd_line"], df["macd_signal"] = macd_line.values, macd_signal.values

    adx_lst, plus_lst, minus_lst = [], [], []
    for i in range(len(df)):
        if i < 14:
            adx_lst.append(np.nan); plus_lst.append(np.nan); minus_lst.append(np.nan)
        else:
            a, p, m = calculate_adx_and_di(df["h"][:i+1], df["l"][:i+1], df["c"][:i+1])
            adx_lst.append(a); plus_lst.append(p); minus_lst.append(m)
    df["adx"], df["plus_di"], df["minus_di"] = adx_lst, plus_lst, minus_lst

    df["stoch_rsi"] = [stochastic_rsi(df["c"][:i+1]) for i in range(len(df))]
    df["atr"] = [calculate_atr({"high":df["h"][:i+1],
                                "low": df["l"][:i+1],
                                "close":df["c"][:i+1]}) for i in range(len(df))]
    df["super_dir"] = [calculate_supertrend(df["h"][:i+1],
                                            df["l"][:i+1],
                                            df["c"][:i+1]) if i>=10 else np.nan
                       for i in range(len(df))]
    df["pct_change_close"] = df["c"].pct_change()
    df["rolling_volatility"] = df["pct_change_close"].rolling(14).std()
    return df.dropna().reset_index(drop=True)

def match_signals_and_trades(window_minutes=10):
    if not (os.path.isfile(DATA_CSV) and os.path.isfile(TRADES_CSV)):
        return pd.DataFrame()
    sig = pd.read_csv(DATA_CSV); trd = pd.read_csv(TRADES_CSV)
    if sig.empty or trd.empty: return pd.DataFrame()
    sig = sig[sig["Signal"].isin(["LONG", "SHORT"])]
    sig["DateTime"] = pd.to_datetime(sig["DateTime"])
    trd["DateTime"] = pd.to_datetime(trd["DateTime"])
    sig = sig.sort_values("DateTime")
    trd = trd.sort_values("DateTime")
    merged = pd.merge_asof(sig, trd, by="Pair",
                           left_on="DateTime", right_on="DateTime",
                           direction="nearest",
                           tolerance=pd.Timedelta(minutes=window_minutes))
    merged = merged.dropna(subset=["PNL"])
    merged = merged[merged["Signal"] == merged["Side"]]
    merged["label"] = (merged["PNL"] > 0).astype(int)
    return merged

def fetch_extended_historical_data(pair, interval, limit=500):
    kl = client.futures_klines(symbol=pair, interval=interval, limit=min(limit, 1500))
    df = pd.DataFrame(kl, columns=[
        "open_time","o","h","l","c","v","close_time","quote_asset_vol",
        "trades","taker_base_vol","taker_quote_vol","ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms").dt.tz_localize(None)
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = df[col].astype(float)
    return df

# ----------------------------  **PATCHED**  ----------------------------
def prepare_dataset_for_ml(pair):
    """
    Safely assemble (X, y, w) for the specified pair.
    If the CSV files are empty or still lack data, return (None, None, None)
    so that the caller skips ML training until enough history exists.
    """
    df_merge = match_signals_and_trades(10)
    if df_merge.empty or "Pair" not in df_merge.columns:
        return None, None, None

    df_merge = df_merge[df_merge["Pair"] == pair]
    if df_merge.empty:
        return None, None, None

    df_hist = fetch_extended_historical_data(pair, TIMEFRAME,
                                             EXTENDED_HISTORICAL_BARS)
    if df_hist.empty:
        return None, None, None
    df_feat = build_features_from_ohlc(df_hist)
    df_feat["open_time"] = df_feat["open_time"].dt.tz_localize(None)
    df_feat["Time"] = df_feat["open_time"]
    df_feat["Pair"] = pair

    df_merge = pd.merge_asof(df_merge.sort_values("DateTime"),
                             df_feat.sort_values("Time"),
                             by="Pair",
                             left_on="DateTime", right_on="Time",
                             direction="backward",
                             tolerance=pd.Timedelta(minutes=15))
    if df_merge.empty or "o" not in df_merge.columns:
        return None, None, None

    feature_cols = [
        "o","h","l","c","v",
        "ema50","macd_line","macd_signal","adx",
        "plus_di","minus_di","stoch_rsi","atr","super_dir",
        "pct_change_close","rolling_volatility",
        "EMA50","MACD_Line","MACD_Signal","ADX",
        "PlusDI","MinusDI","StochRSI","ATR","SuperDir",
        "SignalEncoded"
    ]
    df_merge["SignalEncoded"] = df_merge["Signal"].map({"LONG":1,"SHORT":-1})
    for col in feature_cols:
        if col not in df_merge.columns:
            df_merge[col] = np.nan
    df_merge = df_merge.dropna(subset=feature_cols)
    if df_merge.empty:
        return None, None, None

    X = df_merge[feature_cols].copy()
    y = df_merge["label"].astype(int)
    w = df_merge["PNL"].abs() + 1e-6
    return X, y, w
# ----------------------------------------------------------------------

def train_advanced_ml_model_for_pair(pair):
    X, y, w = prepare_dataset_for_ml(pair)
    if X is None or len(X) < 40 or len(np.unique(y)) < 2:
        logging.info(f"[ML] Not enough data for {pair}")
        return None

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    w_train         = w.iloc[:split_idx]

    base_model = GradientBoostingClassifier(random_state=42)
    if _have_bayes:
        param_space = {
            "n_estimators": Integer(50, 300),
            "max_depth": Integer(2, 8),
            "learning_rate": Real(1e-3, 0.2, prior='log-uniform'),
            "subsample": Real(0.5, 1.0)
        }
        opt = BayesSearchCV(base_model, param_space, n_iter=15,
                            cv=TimeSeriesSplit(3), scoring="accuracy",
                            random_state=42, n_jobs=-1)
        opt.fit(X_train, y_train, sample_weight=w_train)
        model = opt.best_estimator_
    else:
        model = base_model
        model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if acc < 0.55:
        logging.info(f"[ML] {pair} walk-fwd acc {acc:.2f} < 0.55  ➜ skipped")
        return None

    model.fit(X, y, sample_weight=w)
    joblib.dump(model, ML_MODEL_PATH_TEMPLATE.format(pair))
    logging.info(f"[ML] Trained model for {pair} ({len(X)} rows, wf acc {acc:.2f})")
    return model

def train_advanced_ml_model_for_all_pairs():
    for p in PAIRS:
        train_advanced_ml_model_for_pair(p)

def load_advanced_ml_model(pair):
    path = ML_MODEL_PATH_TEMPLATE.format(pair)
    return joblib.load(path) if os.path.isfile(path) else None

def ml_predict_if_profitable(indicators, pair, signal):
    model = load_advanced_ml_model(pair)
    if model is None: return True

    df_hist = fetch_extended_historical_data(pair, TIMEFRAME, 50)
    df_feat = build_features_from_ohlc(df_hist)
    if df_feat.empty: return True
    latest = df_feat.iloc[-1]

    single = {
        "o": latest["o"], "h":latest["h"], "l":latest["l"], "c":latest["c"],
        "v": latest["v"], "ema50":latest["ema50"],
        "macd_line":latest["macd_line"], "macd_signal":latest["macd_signal"],
        "adx":latest["adx"], "plus_di":latest["plus_di"], "minus_di":latest["minus_di"],
        "stoch_rsi":latest["stoch_rsi"], "atr":latest["atr"], "super_dir":latest["super_dir"],
        "pct_change_close":latest["pct_change_close"],
        "rolling_volatility":latest["rolling_volatility"],
        "EMA50":indicators.get("ema50",np.nan),
        "MACD_Line":indicators.get("macd_line",np.nan),
        "MACD_Signal":indicators.get("macd_signal",np.nan),
        "ADX":indicators.get("adx",np.nan), "PlusDI":indicators.get("plus_di",np.nan),
        "MinusDI":indicators.get("minus_di",np.nan), "StochRSI":indicators.get("stoch_rsi",np.nan),
        "ATR":indicators.get("atr",np.nan), "SuperDir":indicators.get("super_dir",np.nan),
        "SignalEncoded": 1 if signal=="LONG" else -1
    }
    df_infer = pd.DataFrame([single]).fillna(0)
    proba = model.predict_proba(df_infer)[0][1]
    logging.info(f"[ML] {pair} prob profitable {proba:.2f}")
    return proba >= CONFIDENCE_THRESHOLD

########################################################################
########################## TRADE MANAGEMENT ############################
########################################################################
def calculate_trade_amount(balance, price, pair):
    margin = balance * TRADE_AMOUNT_PERCENTAGE
    position_size = margin * LEVERAGE
    raw_qty = position_size / price
    qty = adjust_qty(raw_qty, pair)
    if qty == 0.0 or qty * price < 5.0:
        logging.info(f"{pair}: qty too small (would be {raw_qty})")
        return 0.0
    return qty

def place_trade(pair, side, full_amount, atr, price, prec):
    """Open a market position and place fully-sized SL + scaled TPs."""
    try:
        ensure_leverage(pair)

        if side == "LONG":
            open_side, close_side = SIDE_BUY, SIDE_SELL
            sl_raw  = price - SL_ATR_MULT*atr
            tp1_raw = price + TP1_ATR_MULT*atr
            tp2_raw = price + TP2_ATR_MULT*atr
            sl  = adjust_price(sl_raw,  pair)
            tp1 = adjust_price(tp1_raw, pair, up=True)
            tp2 = adjust_price(tp2_raw, pair, up=True)
        else:
            open_side, close_side = SIDE_SELL, SIDE_BUY
            sl_raw  = price + SL_ATR_MULT*atr
            tp1_raw = price - TP1_ATR_MULT*atr
            tp2_raw = price - TP2_ATR_MULT*atr
            sl  = adjust_price(sl_raw,  pair, up=True)
            tp1 = adjust_price(tp1_raw, pair)
            tp2 = adjust_price(tp2_raw, pair)

        tp1_price_dict[pair] = tp1
        tp2_price_dict[pair] = tp2

        # ─── OPEN MARKET ORDER ───────────────────────────────────────────
        mkt = retry_call(client.futures_create_order,
                         symbol=pair,
                         side=open_side,
                         type=ORDER_TYPE_MARKET,
                         quantity=full_amount,
                         newOrderRespType="RESULT")
        trade_open_time[pair] = mkt.get("updateTime", 0)

        # fetch the real filled size & entry
        pos = next(p for p in retry_call(client.futures_position_information)
                   if p["symbol"] == pair)
        qty   = abs(float(pos["positionAmt"]))
        entry = float(pos["entryPrice"])
        trade_entry_price[pair]               = entry
        trade_entry_price[f"{pair}_full_qty"] = qty

        # sizing for partial targets
        half_qty    = adjust_qty(qty / 2, pair)
        remain_qty  = adjust_qty(qty - half_qty, pair, up=True)
        if half_qty == 0.0:
            half_qty = remain_qty = qty

        # ─── EXIT ORDERS ─────────────────────────────────────────────────
        # full-size stop-loss
        retry_call(client.futures_create_order,
                   symbol=pair,
                   side=close_side,
                   type="STOP_MARKET",
                   stopPrice=sl,
                   quantity=qty,
                   reduceOnly=True)
        # TP1 – half
        retry_call(client.futures_create_order,
                   symbol=pair,
                   side=close_side,
                   type="TAKE_PROFIT_MARKET",
                   stopPrice=tp1,
                   quantity=half_qty,
                   reduceOnly=True)
        # TP2 – rest
        retry_call(client.futures_create_order,
                   symbol=pair,
                   side=close_side,
                   type="TAKE_PROFIT_MARKET",
                   stopPrice=tp2,
                   quantity=remain_qty,
                   reduceOnly=True)

        log_msg = (f"OPEN {pair} {side} | qty {qty} | entry {entry:.4f} | "
                   f"SL {sl} | TP1 {tp1} | TP2 {tp2}")
        logging.info(log_msg)
        send_telegram_message(
            f"✅ <b>ANAML3 Trade</b>\nPair {pair}\nSide {side}"
            f"\nEntry {entry:.4f}\nQty {qty}\nSL {sl}\nTP1 {tp1}\nTP2 {tp2}",
            "Trade"
        )

    except Exception as e:
        err = f"Place trade error {pair}: {e}"
        logging.error(err)
        send_telegram_message(err, "Error")
        order_errors.append(err)

def compute_exit_price_and_pnl(pair, side, open_time_ms):
    try:
        trades = client.futures_account_trades(symbol=pair, startTime=open_time_ms)
        closers = [t for t in trades if t["side"] ==
                   ("SELL" if side=="LONG" else "BUY")]
        if not closers:
            price = float(client.futures_mark_price(symbol=pair)["markPrice"])
            return price, 0.0
        qty_tot = sum(float(t["qty"]) for t in closers)
        if qty_tot == 0:
            return float(closers[-1]["price"]), 0.0
        avg_exit = sum(float(t["price"])*float(t["qty"]) for t in closers) / qty_tot
        pnl = sum(float(t["realizedPnl"]) - float(t["commission"]) for t in closers
                  if t["commissionAsset"] == "USDT")
        return avg_exit, pnl
    except Exception as e:
        logging.error(f"exit PNL error {pair}: {e}")
        return float(client.futures_mark_price(symbol=pair)["markPrice"]), 0.0

def finalize_closed_trade(pair):
    side  = trade_side.get(pair)
    entry = trade_entry_price.get(pair)
    if not side or entry is None:
        trade_active[pair] = False
        return
    exit_price, pnl_real = compute_exit_price_and_pnl(pair, side,
                                                      trade_open_time[pair])
    qty = trade_entry_price.get(f"{pair}_full_qty",0.0)
    pnl_theo = (exit_price-entry)*qty if side=="LONG" else (entry-exit_price)*qty
    send_telegram_message(
        f"📉 <b>Trade Closed</b>\nPair {pair}\nSide {side}\nEntry {entry:.4f}\n"
        f"Exit {exit_price:.4f}\nReal PNL {pnl_real:.4f}\nTheo PNL {pnl_theo:.4f}",
        "Trade")
    save_trade_to_csv(pair, side, entry, exit_price, pnl_real, "CLOSED")
    for d in (trade_active, trade_side, trade_open_time, sl_moved):
        d[pair] = False if d is trade_active else None
    tp1_price_dict.pop(pair, None)
    tp2_price_dict.pop(pair, None)
    trailing_stop_data.pop(pair, None)

def monitor_and_update_outcome(pair, pos_cache):
    if not trade_active[pair]:
        return
    try:
        pos = next((p for p in pos_cache if p["symbol"] == pair), None)
        if not pos or abs(float(pos["positionAmt"])) < 1e-7:
            finalize_closed_trade(pair)
            return

        side  = trade_side[pair]
        entry = trade_entry_price[pair]
        qty   = abs(float(pos["positionAmt"]))
        mark  = float(client.futures_mark_price(symbol=pair)["markPrice"])
        atr   = calculate_atr(fetch_candlestick_data(pair, TIMEFRAME, 50), 14)

        reached = (mark-entry) if side=="LONG" else (entry-mark)
        if reached >= TRAIL_START_MULT*atr and pair not in trailing_stop_data:
            trailing_stop_data[pair] = {"last": mark}
            logging.info(f"{pair}: trailing activated")

        if pair in trailing_stop_data:
            last = trailing_stop_data[pair]["last"]
            if ((side == "LONG" and mark - last >= TRAIL_INCREMENT_ATR*atr) or
                (side == "SHORT" and last - mark >= TRAIL_INCREMENT_ATR*atr)):
                new_sl_raw = (entry + (reached-TRAIL_INCREMENT_ATR*atr)
                              if side=="LONG"
                              else entry - (reached-TRAIL_INCREMENT_ATR*atr))
                new_sl = adjust_price(new_sl_raw, pair, up=(side=="SHORT"))
                client.futures_cancel_all_open_orders(symbol=pair)
                client.futures_create_order(symbol=pair,
                                            side=SIDE_SELL if side=="LONG" else SIDE_BUY,
                                            type="STOP_MARKET",
                                            stopPrice=new_sl,
                                            quantity=qty,
                                            reduceOnly=True)
                trailing_stop_data[pair]["last"] = mark
                logging.info(f"{pair} SL moved to {new_sl}")
    except Exception as e:
        logging.error(f"monitor error {pair}: {e}")

########################################################################
########################### SUMMARY / LOOP #############################
########################################################################
def send_summary_message():
    bal = get_balance()
    msg = f"<b>ANAML3 Summary</b>\nBalance {bal:.2f} USDT\n"
    for p in PAIRS:
        msg += f"{p}: {'Active' if trade_active[p] else 'Idle'}\n"
    send_telegram_message(msg, "Summary")

train_advanced_ml_model_for_all_pairs()
last_retrain_time = time.time()

while True:
    try:
        now = time.time()
        if now - last_retrain_time > RETRAIN_INTERVAL_SECONDS:
            train_advanced_ml_model_for_all_pairs()
            last_retrain_time = now

        pos_cache = retry_call(client.futures_position_information)
        for pair in PAIRS:
            if trade_active[pair]:
                monitor_and_update_outcome(pair, pos_cache)
                continue

            data = fetch_candlestick_data(pair, TIMEFRAME, 100)
            atr  = calculate_atr(data, 14)
            ind  = analyze_market(data)
            ind["atr"] = atr
            store_analysis_data(pair, data["raw_candles"][-1], ind)

            sig = ind["signal"]
            if sig is None:
                logging.info(f"{pair}: No strong signal "
                             f"(bull={ind['bull_count']}, bear={ind['bear_count']})")
                continue
            if math.isnan(atr):
                logging.info(f"{pair}: ATR NaN – skipping")
                continue
            if not higher_tf_ok(pair, sig):
                logging.info(f"{pair}: higher-TF filter blocked {sig}")
                continue
            if not ml_predict_if_profitable(ind, pair, sig):
                logging.info(f"{pair}: ML model rejected trade (probability below "
                             f"{CONFIDENCE_THRESHOLD})")
                continue

            bal = get_balance()
            qty = calculate_trade_amount(bal, data["close"][-1], pair)
            if qty > 0:
                place_trade(pair, sig, qty, atr,
                            data["close"][-1], get_price_precision(pair))
                trade_active[pair] = True
                trade_side[pair]   = sig
            else:
                logging.info(f"{pair}: qty=0, trade not opened")

            time.sleep(0.05)  # keep under rate-limit

        if now - last_summary_time > SUMMARY_INTERVAL_SECONDS:
            send_summary_message()
            last_summary_time = now
        time.sleep(TRADE_INTERVAL_SECONDS)

    except Exception as e:
        err = f"Main loop error: {e}"
        logging.error(err)
        send_telegram_message(err, "Error")
