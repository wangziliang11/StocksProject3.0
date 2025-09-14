import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Optional

try:
    # 可选导入，避免循环依赖；若不可用则在本模块内实现等价计算
    from src.logic.indicators import macd as macd_fn
except Exception:
    macd_fn = None

# 新增：尝试导入 RSI 计算
try:
    from src.logic.indicators import rsi as rsi_fn
except Exception:
    rsi_fn = None


def _calc_macd(series: pd.Series):
    if macd_fn is not None:
        return macd_fn(series)
    ema_fast = series.ewm(span=12, adjust=False, min_periods=1).mean()
    ema_slow = series.ewm(span=26, adjust=False, min_periods=1).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=9, adjust=False, min_periods=1).mean()
    hist = dif - dea
    return {"dif": dif, "dea": dea, "hist": hist}

# 新增：RSI 计算（若外部不可用则本地实现简版）
def _calc_rsi(series: pd.Series, period: int = 14):
    if rsi_fn is not None:
        return rsi_fn(series, period=period)
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period, min_periods=1).mean()
    roll_down = down.rolling(period, min_periods=1).mean().replace(0, 1e-9)
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def kline_with_volume(
    df: pd.DataFrame,
    title: str = "K线图",
    ma_windows: Optional[List[int]] = None,
    show_macd: bool = False,
    period: str = "daily",
    second_rows: Optional[List[str]] = None,
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

    # 新增：计算涨跌幅（相对前一收盘）
    prev_close = c.shift(1)
    pct_chg = (c / prev_close - 1.0) * 100.0

    # 判断是否日线：仅日线才补充工作日缺失的 rangebreak，非日线只隐藏周末
    is_daily = (period == "daily")

    # 新增：根据 second_rows 决定价格下方的子图（最多两行），兼容旧的 show_macd 参数
    allowed = ["成交量", "MACD", "RSI"]
    if second_rows is None:
        chosen = ["成交量"] + (["MACD"] if show_macd else [])
    else:
        chosen = [x for x in second_rows if x in allowed][:2]

    # 子图：价格 + 选中的子图
    rows = 1 + len(chosen)
    if len(chosen) == 0:
        row_heights = [1.0]
    elif len(chosen) == 1:
        row_heights = [0.74, 0.26]
    else:
        row_heights = [0.62, 0.24, 0.14]
    specs = [[{"type": "xy"}] for _ in range(rows)]
    subplot_titles = [title] + chosen

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=specs,
        subplot_titles=subplot_titles,
    )

    # K线（兼容旧版 Plotly：优先使用 hovertemplate，失败则回退到 hovertext）
    hover_tpl = (
        "K线： open: %{open:.2f}"
        "<br>high: %{high:.2f}"
        "<br>low: %{low:.2f}"
        "<br>close: %{close:.2f}"
        "<br>涨跌幅: %{customdata:.2f}%"
        "<extra></extra>"
    )
    try:
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
                customdata=pct_chg,
                hovertemplate=hover_tpl,
            ),
            row=1, col=1
        )
    except Exception:
        # 回退：构造逐点 hover 文本
        hover_text = []
        for oi, hi, lo, ci, pi in zip(o.fillna(0), h.fillna(0), l.fillna(0), c.fillna(0), pct_chg):
            pct_str = "" if pd.isna(pi) else f"{pi:.2f}%"
            hover_text.append(
                f"K线： open: {oi:.2f}<br>high: {hi:.2f}<br>low: {lo:.2f}<br>close: {ci:.2f}<br>涨跌幅: {pct_str}"
            )
        fig.add_trace(
            go.Candlestick(
                x=x,
                open=o,
                high=h,
                low=l,
                close=c,
                name="K线",
                increasing_line_color="#e74c3c",
                decreasing_line_color="#2ecc71",
                hovertext=hover_text,
                hoverinfo="text",
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

    # 行索引映射：价格为 row=1，其余依次 2,3
    row_idx = {name: (i + 2) for i, name in enumerate(chosen)}

    # 成交量：按涨跌上色（仅在选择时绘制）
    if "成交量" in chosen:
        colors = ["\n" for _ in range(len(df))]
        for i in range(len(df)):
            colors[i] = "#e74c3c" if (c.iloc[i] >= o.iloc[i]) else "#2ecc71"
        fig.add_trace(
            go.Bar(x=x, y=v, name="成交量", marker_color=colors, opacity=0.8),
            row=row_idx["成交量"], col=1
        )

    # MACD 子图（仅在选择时绘制）
    if "MACD" in chosen:
        macd_vals = _calc_macd(c)
        dif = macd_vals["dif"]
        dea = macd_vals["dea"]
        hist = macd_vals["hist"]
        # 柱状图：红正绿负
        bar_colors = ["#e74c3c" if val >= 0 else "#2ecc71" for val in hist]
        fig.add_trace(go.Bar(x=x, y=hist, name="MACD Hist", marker_color=bar_colors, opacity=0.9), row=row_idx["MACD"], col=1)
        fig.add_trace(go.Scatter(x=x, y=dif, name="DIF", line=dict(color="#ff9f43", width=1.2)), row=row_idx["MACD"], col=1)
        fig.add_trace(go.Scatter(x=x, y=dea, name="DEA", line=dict(color="#0984e3", width=1.2)), row=row_idx["MACD"], col=1)

    # RSI 子图（仅在选择时绘制）
    if "RSI" in chosen:
        rsi_vals = _calc_rsi(c, period=14)
        fig.add_trace(go.Scatter(x=x, y=rsi_vals, name="RSI", line=dict(color="#6c5ce7", width=1.2)), row=row_idx["RSI"], col=1)
        # 参考线 30/70
        lv30 = pd.Series(30, index=x)
        lv70 = pd.Series(70, index=x)
        fig.add_trace(go.Scatter(x=x, y=lv30, name="RSI-30", line=dict(color="#2ecc71", width=1, dash="dot"), showlegend=False), row=row_idx["RSI"], col=1)
        fig.add_trace(go.Scatter(x=x, y=lv70, name="RSI-70", line=dict(color="#e74c3c", width=1, dash="dot"), showlegend=False), row=row_idx["RSI"], col=1)

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
    if "成交量" in chosen:
        fig.update_yaxes(title_text="成交量", row=row_idx["成交量"], col=1, showgrid=False)
    if "MACD" in chosen:
        fig.update_yaxes(title_text="MACD", row=row_idx["MACD"], col=1, showgrid=False)
    if "RSI" in chosen:
        fig.update_yaxes(title_text="RSI", row=row_idx["RSI"], col=1, showgrid=False, range=[0, 100])

    fig.update_layout(
        title=title,
        showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        bargap=0.1,
    )
    return fig