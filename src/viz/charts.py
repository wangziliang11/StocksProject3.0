import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Optional

try:
    # 可选导入，避免循环依赖；若不可用则在本模块内实现等价计算
    from src.logic.indicators import macd as macd_fn
except Exception:
    macd_fn = None


def _calc_macd(series: pd.Series):
    if macd_fn is not None:
        return macd_fn(series)
    ema_fast = series.ewm(span=12, adjust=False, min_periods=1).mean()
    ema_slow = series.ewm(span=26, adjust=False, min_periods=1).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=9, adjust=False, min_periods=1).mean()
    hist = dif - dea
    return {"dif": dif, "dea": dea, "hist": hist}


def kline_with_volume(
    df: pd.DataFrame,
    title: str = "K线图",
    ma_windows: Optional[List[int]] = None,
    show_macd: bool = False,
    period: str = "daily",
) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # 确保日期与数值类型
    x = pd.to_datetime(df["date"])  # type: ignore
    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else pd.Series([None]*len(df))

    # 判断是否日线：仅日线才补充工作日缺失的 rangebreak，非日线只隐藏周末
    is_daily = (period == "daily")

    # 子图：价格、成交量、（可选）MACD
    rows = 3 if show_macd else 2
    row_heights = [0.62, 0.24, 0.14] if show_macd else [0.7, 0.3]
    specs = [[{"type": "xy"}] for _ in range(rows)]
    subplot_titles = [title, "成交量"] + (["MACD"] if show_macd else [])

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=specs,
        subplot_titles=subplot_titles,
    )

    # K线
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=o,
            high=h,
            low=l,
            close=c,
            name="K线",
            increasing_line_color="#e74c3c",  # 红涨
            decreasing_line_color="#2ecc71",  # 绿跌
        ),
        row=1, col=1
    )

    # MA 叠加
    if ma_windows:
        palette = ["#3498db", "#f1c40f", "#9b59b6", "#e67e22", "#1abc9c", "#e84393"]
        for i, w in enumerate(ma_windows):
            ma = c.rolling(window=int(w), min_periods=1).mean()
            fig.add_trace(
                go.Scatter(x=x, y=ma, mode="lines", name=f"MA{w}", line=dict(color=palette[i % len(palette)], width=1.6)),
                row=1, col=1
            )

    # 成交量：按涨跌上色
    colors = ["\n" for _ in range(len(df))]
    for i in range(len(df)):
        colors[i] = "#e74c3c" if (c.iloc[i] >= o.iloc[i]) else "#2ecc71"
    fig.add_trace(
        go.Bar(x=x, y=v, name="成交量", marker_color=colors, opacity=0.8),
        row=2, col=1
    )

    # MACD 子图
    if show_macd:
        macd_vals = _calc_macd(c)
        dif = macd_vals["dif"]
        dea = macd_vals["dea"]
        hist = macd_vals["hist"]
        # 柱状图：红正绿负
        bar_colors = ["#e74c3c" if val >= 0 else "#2ecc71" for val in hist]
        fig.add_trace(go.Bar(x=x, y=hist, name="MACD Hist", marker_color=bar_colors, opacity=0.9), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=dif, name="DIF", line=dict(color="#ff9f43", width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=dea, name="DEA", line=dict(color="#0984e3", width=1.2)), row=3, col=1)

    # 隐藏非交易日：周末 + （仅日线）缺失工作日
    rangebreaks = [dict(bounds=["sat", "mon"])]
    if is_daily:
        try:
            # 仅对工作日缺失的日期做 break（避免非日线加入过多 breaks）
            all_bd = pd.date_range(start=x.min(), end=x.max(), freq="B")
            present = pd.to_datetime(x).normalize().unique()
            missing_bd = sorted(list(set(all_bd) - set(present)))
            if missing_bd:
                rangebreaks.append(dict(values=missing_bd))
        except Exception:
            pass

    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=rangebreaks,
        showspikes=True,
        spikemode="across+toaxis",
        spikesnap="cursor",
        row=1, col=1
    )
    for r in range(2, rows+1):
        fig.update_xaxes(rangeslider_visible=False, rangebreaks=rangebreaks, row=r, col=1)

    # 坐标轴标题与样式
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1, showgrid=False)
    if show_macd:
        fig.update_yaxes(title_text="MACD", row=3, col=1, showgrid=False)

    fig.update_layout(
        title=title,
        showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        bargap=0.1,
    )
    return fig