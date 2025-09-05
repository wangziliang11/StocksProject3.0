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