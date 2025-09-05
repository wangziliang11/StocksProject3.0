# 股票分析系统入口（Streamlit）
import streamlit as st
import pandas as pd
import akshare as ak
import sys, os
from pathlib import Path
from typing import Optional
import importlib
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.ak_client import AKDataClient
from src.logic.indicators import sma, ema, macd, backtest_ma_cross
from src.viz.charts import kline_with_volume
from src.llm.providers.registry import ProviderRegistry, LLMRouter, OpenAICompatClient
from src.llm.tools.schema import get_tools_schema
from src.llm.tools.executor import fetch_stock_info_a, fetch_stock_info_hk


# 缓存获取股票名称（A股/港股）
@st.cache_data(ttl=3600)
def get_stock_name_cached(market: str, symbol: str) -> Optional[str]:
    try:
        if market == "A":
            try:
                df_spot = ak.stock_zh_a_spot_em()
                code_col = "代码" if "代码" in df_spot.columns else None
                name_col = "名称" if "名称" in df_spot.columns else None
                if code_col and name_col:
                    sym_norm = str(symbol).lstrip("0")
                    df_spot["_code_norm"] = df_spot[code_col].astype(str).str.lstrip("0")
                    hit = df_spot[df_spot["_code_norm"] == sym_norm]
                    if not hit.empty:
                        return str(hit[name_col].iloc[0])
            except Exception:
                pass
            # 回退：用个股信息接口提取“股票简称/证券简称/简称”
            try:
                info = ak.stock_individual_info_em(symbol=symbol)
                if info is not None and not info.empty and set(["item","value"]).issubset(info.columns):
                    alias_keys = ["股票简称", "证券简称", "简称", "股票简称(中文)"]
                    sub = info[info["item"].astype(str).isin(alias_keys)]
                    if not sub.empty:
                        return str(sub["value"].iloc[0])
            except Exception:
                pass
        else:
            df_spot = ak.stock_hk_spot_em()
            code_col = None
            for c in ["代码", "symbol", "code", "证券代码"]:
                if c in df_spot.columns:
                    code_col = c
                    break
            name_col = "名称" if "名称" in df_spot.columns else None
            if code_col and name_col:
                sym = str(symbol).upper().replace(".HK", "")
                sym_norm = sym.lstrip("0")
                df_spot["_code_norm"] = (
                    df_spot[code_col]
                    .astype(str)
                    .str.upper()
                    .str.replace(".HK", "", regex=False)
                    .str.lstrip("0")
                )
                hit = df_spot[df_spot["_code_norm"] == sym_norm]
                if not hit.empty:
                    return str(hit[name_col].iloc[0])
    except Exception:
        return None
    return None


def main():
    st.set_page_config(page_title="A/H 股票分析系统", layout="wide")
    st.title("A/H 股票分析系统")
    st.caption("数据源：AKShare | 绘图：Plotly | 前端：Streamlit")

    with st.sidebar:
        st.header("数据与模型配置")
        market = st.selectbox("市场", ["A", "H"], index=0)
        symbol = st.text_input("股票代码", value="600519" if market == "A" else "00700")
        period = st.selectbox("周期", ["daily", "weekly", "monthly", "quarterly", "yearly"], index=0)
        start = st.text_input("开始日期(YYYYMMDD)", value="20180101")
        end = st.text_input("结束日期(YYYYMMDD)", value="20251231")
        adjust = st.selectbox("复权(A股)", [None, "qfq", "hfq"], index=0) if market=="A" else None
        st.subheader("缓存控制")
        use_cache = st.checkbox("使用缓存", value=True, help="读取/写入 data/cache 缓存")
        refresh = st.checkbox("强制刷新", value=False, help="忽略过期策略，重新拉取并覆盖缓存")
        expire_days = st.number_input("过期天数", min_value=0, max_value=365, value=3, step=1, help="0 表示永不过期")
        st.divider()
        st.subheader("指标叠加")
        show_ma = st.checkbox("显示 MA 均线", value=True)
        ma_list_str = st.text_input("MA窗口(逗号分隔)", value="5,10,20,60")
        ma_windows = [int(x) for x in ma_list_str.split(',') if x.strip().isdigit()] if show_ma else []
        show_macd = st.checkbox("显示 MACD", value=False)
        st.divider()
        st.subheader("大模型(可选)")
        st.caption("直连使用 models.local.yaml 中的 API Key；如未配置则回退到路由(models.yaml/routing.yaml)。")
        provider_label = st.selectbox("提供商", ["硅基流动", "豆包", "通义千问"], index=0)
        default_model_map = {
            "硅基流动": "Qwen/Qwen2.5-7B-Instruct",
            "豆包": "ep-xxxxxxxx",  # 推理接入点ID
            "通义千问": "qwen-turbo",
        }
        model_ui = st.text_input("模型(或接入点)", value=default_model_map[provider_label])
        route_name = st.text_input("路由名(route)", value="default")
        enable_tools = st.checkbox("启用联网工具(Function Calling)", value=True)
        user_query = st.text_area("问模型：个股/行业信息、策略建议...", value="")

    # 数据获取
    client = AKDataClient()
    df = client.get_hist(market=market, symbol=symbol, period=period, start=start, end=end, adjust=adjust, use_cache=use_cache, refresh=refresh, expire_days=expire_days)

    # 股票名称（缓存）
    stock_name = get_stock_name_cached(market, symbol)

    # 展示日期区间（仅影响展示与绘图，不触发重新抓取）
    if df is not None and not df.empty:
        dmin = pd.to_datetime(df["date"].min()).date()
        dmax = pd.to_datetime(df["date"].max()).date()
        # 快捷按钮
        c1, c2, c3, c4 = st.columns([1,1,1,4])
        with c1:
            if st.button("近30天"):
                dmin_sel = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=30)).date())
                st.session_state["disp_range_override"] = (dmin_sel, dmax)
        with c2:
            if st.button("近90天"):
                dmin_sel = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=90)).date())
                st.session_state["disp_range_override"] = (dmin_sel, dmax)
        with c3:
            if st.button("近1年"):
                dmin_sel = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=365)).date())
                st.session_state["disp_range_override"] = (dmin_sel, dmax)

        if "disp_range_override" in st.session_state:
            default_range = st.session_state["disp_range_override"]
        else:
            default_range = (dmin, dmax)

        disp_range = st.date_input("展示日期区间", value=default_range)
        if isinstance(disp_range, tuple) and len(disp_range) == 2:
            ds, de = disp_range
        else:
            ds, de = default_range
        df_disp = df[(df["date"] >= pd.to_datetime(ds)) & (df["date"] <= pd.to_datetime(de))]
    else:
        df_disp = df

    # 列名中文映射（仅用于展示）
    cn_map = {
        "date": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "volume": "成交量",
        "amount": "成交额",
        "adj_factor": "复权因子",
        "market": "市场",
        "symbol": "代码",
    }

    st.subheader(f"历史数据概览 - {stock_name}({symbol})" if stock_name else f"历史数据概览 - {symbol}")
    if df_disp is not None and not df_disp.empty:
        st.dataframe(df_disp.sort_values("date", ascending=False).rename(columns=cn_map).head(200))
    else:
        st.info("无数据，请检查代码、周期或日期范围。")

    # 个股/行业信息（A股/港股）
    st.subheader("个股/行业信息")
    if market == "A":
        show_info = st.checkbox("拉取个股基本信息(A股)", value=False)
        if show_info:
            try:
                info_df = ak.stock_individual_info_em(symbol=symbol)
                st.dataframe(info_df)
            except Exception as e:
                st.warning(f"获取个股信息失败: {e}")
    else:
        show_info_hk = st.checkbox("拉取个股实时报价(港股)", value=False)
        if show_info_hk:
            try:
                df_spot = ak.stock_hk_spot_em()
                code_col = None
                for c in ["代码", "symbol", "code", "证券代码"]:
                    if c in df_spot.columns:
                        code_col = c
                        break
                if code_col is None:
                    st.warning("数据格式异常：未找到代码列")
                else:
                    sym = str(symbol).upper().replace(".HK", "")
                    df_spot["_code_norm"] = df_spot[code_col].astype(str).str.upper().str.replace(".HK", "", regex=False).str.lstrip("0")
                    sym_norm = sym.lstrip("0")
                    candidates = df_spot[df_spot["_code_norm"] == sym_norm].drop(columns=["_code_norm"])
                    if candidates.empty:
                        st.info("未在实时报价列表中找到该代码，可能为停牌或代码格式不匹配。请尝试包含或去除前导 0。")
                    else:
                        st.dataframe(candidates)
            except Exception as e:
                st.warning(f"获取港股实时报价失败: {e}")

    # 指标与回测
    st.subheader("指标与回测")
    if df_disp is not None and not df_disp.empty:
        close = df_disp["close"].astype(float)
        df_ind = df_disp.copy()
        df_ind["SMA20"] = sma(close, 20)
        df_ind["SMA60"] = sma(close, 60)
        res_bt = backtest_ma_cross(df_ind, short=20, long=60)
        st.write("回测结果：", res_bt)

        # 绘图
        st.subheader("K线图")
        fig_title = f"{stock_name}({symbol}) K线" if stock_name else f"{symbol} K线"
        # 确保使用 charts 最新定义，避免旧模块签名导致的报错
        try:
            import src.viz.charts as charts_mod
            charts_mod = importlib.reload(charts_mod)
            fig = charts_mod.kline_with_volume(df_disp, title=fig_title, ma_windows=ma_windows, show_macd=show_macd, period=period)
        except Exception:
            # 回退：直接使用已导入的函数
            fig = kline_with_volume(df_disp, title=fig_title, ma_windows=ma_windows, show_macd=show_macd, period=period)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("无数据，请检查代码、周期或日期范围。港股周/月/季/年线为日线重采样。A股周/月线 AKShare 可能会在新股或特殊日期返回空。")

    # 大模型问答（通过配置 + 路由 或 直连）
    st.subheader("大模型问答")
    inject_ctx = st.checkbox("将行情/回测摘要注入模型上下文", value=True)
    if user_query:
        try:
            # 构造消息
            sys_prompt = {"role": "system", "content": "你是资深量化分析师。可以结合已知数据做基本面与技术面分析，并提醒数据来源和不构成投资建议。"}
            messages = [sys_prompt]
            if inject_ctx and df_disp is not None and not df_disp.empty:
                try:
                    latest_close = float(df_disp["close"].iloc[-1])
                    latest_vol = float(df_disp["volume"].iloc[-1]) if pd.notna(df_disp["volume"].iloc[-1]) else 0
                    ctx_lines = [
                        f"市场: {market}, 代码: {symbol}, 区间: {start}-{end}, 周期: {period}",
                        f"最新收盘: {latest_close}, 最新成交量: {latest_vol}",
                    ]
                    if 'df_ind' in locals():
                        try:
                            ctx_lines.append(f"SMA20: {float(df_ind['SMA20'].iloc[-1]):.4f}, SMA60: {float(df_ind['SMA60'].iloc[-1]):.4f}")
                        except Exception:
                            pass
                    if 'res_bt' in locals() and isinstance(res_bt, dict):
                        try:
                            ctx_lines.append(
                                f"回测摘要: 累计收益 {res_bt.get('return',0):.4%}, 最大回撤 {res_bt.get('max_drawdown',0):.4%}, 交易次数 {res_bt.get('trades',0)}"
                            )
                        except Exception:
                            pass
                    context_str = "\n".join(ctx_lines)
                    messages.append({"role": "system", "content": "以下是行情与回测摘要，请结合回答问题：\n" + context_str})
                except Exception:
                    pass
            user_msg = {"role": "user", "content": user_query}
            messages.append(user_msg)

            tools = get_tools_schema() if enable_tools else None

            # 优先尝试：使用所选提供商直连（从 models.local.yaml 读取 api_key）；若无则回退到路由
            registry = ProviderRegistry(public_cfg_path="models.yaml", local_cfg_path="models.local.yaml")
            # 后续 LLM 调用与工具派发逻辑...（略）
        except Exception as e:
            st.error(f"调用模型失败: {e}")


if __name__ == "__main__":
    main()