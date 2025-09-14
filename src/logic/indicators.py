import pandas as pd
from typing import Dict


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False, min_periods=1).mean()
    hist = dif - dea
    return {"dif": dif, "dea": dea, "hist": hist}


def backtest_ma_cross(df: pd.DataFrame, short: int = 10, long: int = 20, fee: float = 0.0005) -> Dict[str, float]:
    # 简单均线金叉死叉策略回测
    result = {
        "trades": 0,
        "return": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
    }
    if df is None or df.empty or "close" not in df.columns:
        return result
    close = df["close"].astype(float)
    ma_s = sma(close, short)
    ma_l = sma(close, long)
    signal = (ma_s > ma_l).astype(int)
    position = signal.shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    strategy_ret = position * ret - abs(position.diff().fillna(0)) * fee
    equity = (1 + strategy_ret).cumprod()
    # 统计
    result["return"] = equity.iloc[-1] - 1 if len(equity) else 0.0
    result["max_drawdown"] = ((equity.cummax() - equity) / equity.cummax()).max() if len(equity) else 0.0
    # 交易次数与胜率估计
    trades = (position.diff().abs() == 1).sum() // 2  # 进出各一次
    result["trades"] = int(trades)
    wins = (strategy_ret[strategy_ret != 0] > 0).sum()
    total = (strategy_ret != 0).sum()
    result["win_rate"] = float(wins / total) if total > 0 else 0.0
    return result

# --- 新增：RSI 指标与相关回测 ---

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = roll_up / (roll_down.replace(0, 1e-12))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def backtest_macd_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, fee: float = 0.0005) -> Dict[str, float]:
    result = {"trades": 0, "return": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
    if df is None or df.empty or "close" not in df.columns:
        return result
    close = df["close"].astype(float)
    m = macd(close, fast=fast, slow=slow, signal=signal)
    dif, dea = m["dif"], m["dea"]
    signal_line = (dif > dea).astype(int)
    position = signal_line.shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    strategy_ret = position * ret - abs(position.diff().fillna(0)) * fee
    equity = (1 + strategy_ret).cumprod()
    result["return"] = equity.iloc[-1] - 1 if len(equity) else 0.0
    result["max_drawdown"] = ((equity.cummax() - equity) / equity.cummax()).max() if len(equity) else 0.0
    trades = (position.diff().abs() == 1).sum() // 2
    result["trades"] = int(trades)
    wins = (strategy_ret[strategy_ret != 0] > 0).sum()
    total = (strategy_ret != 0).sum()
    result["win_rate"] = float(wins / total) if total > 0 else 0.0
    return result


def backtest_rsi(df: pd.DataFrame, period: int = 14, low: int = 30, high: int = 70, fee: float = 0.0005) -> Dict[str, float]:
    """简单 RSI 区间策略：RSI 上穿 low 买入，下穿 high 卖出。"""
    result = {"trades": 0, "return": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
    if df is None or df.empty or "close" not in df.columns:
        return result
    close = df["close"].astype(float)
    r = rsi(close, period=period)
    long_sig = (r > low).astype(int)  # 上穿低位后持有
    flat_sig = (r < high).astype(int)
    # 构造持仓：进入后持有，直到跌破 high（可按需改为上下穿交叉检测）
    position = long_sig.copy()
    position[r < low] = 0
    position[r > high] = 0
    position = position.shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    strategy_ret = position * ret - abs(position.diff().fillna(0)) * fee
    equity = (1 + strategy_ret).cumprod()
    result["return"] = equity.iloc[-1] - 1 if len(equity) else 0.0
    result["max_drawdown"] = ((equity.cummax() - equity) / equity.cummax()).max() if len(equity) else 0.0
    trades = (position.diff().abs() == 1).sum() // 2
    result["trades"] = int(trades)
    wins = (strategy_ret[strategy_ret != 0] > 0).sum()
    total = (strategy_ret != 0).sum()
    result["win_rate"] = float(wins / total) if total > 0 else 0.0
    return result