# 股票分析系统入口（Streamlit）
import streamlit as st
import pandas as pd
import akshare as ak
import sys, os
from pathlib import Path
from typing import Optional, List, Dict, Any
import importlib
import json
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.ak_client import AKDataClient
from src.logic.indicators import sma, ema, macd, backtest_ma_cross
from src.viz.charts import kline_with_volume
from src.llm.providers.registry import ProviderRegistry, LLMRouter, OpenAICompatClient
from src.llm.tools.schema import get_tools_schema
from src.llm.tools.executor import fetch_stock_info_a, fetch_stock_info_hk

# --- 工具分发映射与自动执行循环 ---
TOOL_MAP = {
    "fetch_stock_info_a": fetch_stock_info_a,
    "fetch_stock_info_hk": fetch_stock_info_hk,
}

# 仅展示模型答案的工具函数，避免把原始返回（raw JSON）直接渲染到页面
from typing import Any as _Any

def _extract_text_from_raw(rsp: _Any) -> str:
    try:
        if isinstance(rsp, dict):
            choices = rsp.get("choices") or []
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") or {}
                if isinstance(msg, dict):
                    txt = (msg.get("content") or msg.get("reasoning_content") or "")
                    if txt:
                        return str(txt)
            msg = rsp.get("message") or {}
            if isinstance(msg, dict):
                txt = (msg.get("content") or msg.get("reasoning_content") or "")
                if txt:
                    return str(txt)
        return ""
    except Exception:
        return ""

def _render_llm_answer(result: Dict[str, _Any]):
    final_text = (result or {}).get("final_text") or ""
    if isinstance(final_text, str) and final_text.strip():
        st.markdown(final_text)
        return
    fallback = _extract_text_from_raw((result or {}).get("raw"))
    if fallback.strip():
        st.markdown(fallback)
    else:
        st.warning("模型未返回可读答案，请重试或更换路由/模型。")

def chat_with_tools(router: LLMRouter, messages, tools_schema=None, max_rounds: int = 3):
    """
    通用自动工具执行循环：
    - 使用提供的 router（OpenAI 兼容接口）发送消息
    - 当模型返回 tool_calls 时，在本地调用 TOOL_MAP 对应函数
    - 将工具结果作为 tool 消息回传，再次让模型整合，直到产出最终答案或达到轮数上限
    返回: {"final_text": str, "raw": any}
    """
    conv = list(messages)
    last_raw = None
    for _ in range(max_rounds):
        rsp = router.chat(messages=conv, tools=tools_schema, tool_choice="auto")
        last_raw = rsp
        # 兼容 OpenAI 风格返回
        msg = None
        if isinstance(rsp, dict):
            choices = rsp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
            else:
                msg = rsp.get("message") or {}
        # 无法解析，直接返回原始
        if not isinstance(msg, dict):
            return {"final_text": "", "raw": rsp}

        role = msg.get("role", "assistant")
        content = (msg.get("content") or msg.get("reasoning_content") or "")
        tool_calls = msg.get("tool_calls") or []
        # 追加 assistant 回复（可能同时带工具调用）
        if tool_calls:
            conv.append({"role": role, "content": content, "tool_calls": tool_calls})
        else:
            conv.append({"role": role, "content": content})

        # 若模型触发工具调用，则本地执行
        if tool_calls:
            for call in tool_calls:
                fn_meta = call.get("function", {}) if isinstance(call, dict) else {}
                fn_name = fn_meta.get("name")
                raw_args = fn_meta.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception:
                    args = {}
                tool_fn = TOOL_MAP.get(fn_name)
                if not tool_fn:
                    tool_output = {"error": f"unknown tool: {fn_name}"}
                else:
                    try:
                        tool_output = tool_fn(**args)
                    except Exception as e:
                        tool_output = {"error": str(e)}
                conv.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": fn_name,
                    "content": json.dumps(tool_output, ensure_ascii=False)
                })
            # 继续下一轮，让模型整合工具结果
            continue
        else:
            # 没有工具调用，认为已给出最终回答
            return {"final_text": content, "raw": rsp}

    # 达到回合上限，返回最后一次原始结果
    return {"final_text": content if content else "", "raw": last_raw}


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

# -------- 新增：导航与自选股持久化工具 --------
WATCHLIST_PATH = ROOT / "data" / "watchlist.json"

def _ensure_data_dir():
    (ROOT / "data").mkdir(parents=True, exist_ok=True)

def load_watchlist() -> List[Dict[str, str]]:
    _ensure_data_dir()
    if WATCHLIST_PATH.exists():
        try:
            return json.load(open(WATCHLIST_PATH, "r", encoding="utf-8"))
        except Exception:
            return []
    return []

def save_watchlist(items: List[Dict[str, str]]):
    _ensure_data_dir()
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def go_detail(market: str, symbol: str):
    st.session_state["detail_market"] = market
    st.session_state["detail_symbol"] = symbol
    try:
        st.switch_page("pages/StockDetail.py")
    except Exception:
        st.warning("无法跳转到详情页，请确认已创建 pages/StockDetail.py")

# -------- 详情页所复用的单股页面（原 main）提取为函数 --------
def single_stock_page():
    st.header("单只股票查询")
    # 原 sidebar 中的控件迁移为当前分区内控件
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        market = st.selectbox("市场", ["A", "H"], index=st.session_state.get("_ss_market_idx", 0), key="ss_market")
        st.session_state["_ss_market_idx"] = 0 if market == "A" else 1
        symbol = st.text_input("股票代码", value=st.session_state.get("detail_symbol", "600519" if market == "A" else "00700"))
    with c2:
        period = st.selectbox("周期", ["daily", "weekly", "monthly", "quarterly", "yearly"], index=0, key="ss_period")
        start = st.text_input("开始(YYYYMMDD)", value="20180101")
        end = st.text_input("结束(YYYYMMDD)", value="20251231")
    with c3:
        adjust = st.selectbox("复权(A)", [None, "qfq", "hfq"], index=0, key="ss_adjust") if market == "A" else None
        use_cache = st.checkbox("使用缓存", value=True)
        refresh = st.checkbox("强制刷新", value=False)
        expire_days = st.number_input("过期天数", 0, 365, 3, 1)

    show_ma = st.checkbox("显示 MA 均线", value=True)
    ma_list_str = st.text_input("MA窗口(逗号分隔)", value="5,10,20,60")
    ma_windows = [int(x) for x in ma_list_str.split(',') if x.strip().isdigit()] if show_ma else []
    show_macd = st.checkbox("显示 MACD", value=False)

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
        c1b, c2b, c3b, c4b = st.columns([1,1,1,4])
        with c1b:
            if st.button("近30天"):
                dmin_sel = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=30)).date())
                st.session_state["disp_range_override"] = (dmin_sel, dmax)
        with c2b:
            if st.button("近90天"):
                dmin_sel = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=90)).date())
                st.session_state["disp_range_override"] = (dmin_sel, dmax)
        with c3b:
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
    # 详情页跳转按钮
    if st.button("跳转到详情页", key="go_detail_single"):
        go_detail(market, symbol)
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

        # 新增：一键生成中文总结
        col_a, col_b = st.columns([1, 4])
        with col_a:
            gen_cn = st.button("生成中文总结", use_container_width=True)
        if gen_cn:
            try:
                trades = int(res_bt.get("trades", 0))
                ret = float(res_bt.get("return", 0.0))
                mdd = float(res_bt.get("max_drawdown", 0.0))
                win = float(res_bt.get("win_rate", 0.0))
                period_txt = {"daily":"日线","weekly":"周线","monthly":"月线","quarterly":"季线","yearly":"年线"}.get(period, str(period))
                ma_txt = "MA(20/60) 趋势策略：当短均线上穿长均线买入、下穿卖出；适合趋势行情，震荡时容易虚假信号。"
                summary = (
                    f"回测基于{period_txt}与所选展示区间。共 {trades} 笔交易，区间累计收益约 {ret:.2%}，最大回撤约 {mdd:.2%}，胜率约 {win:.2%}。\n"
                    f"直观解读：收益较{'可观' if ret>0 else '一般'}，回撤{'偏高' if mdd>0.25 else '可控'}，胜率{'略低于' if win<0.5 else '高于'} 50%。策略更依赖趋势段与盈亏比，需控制仓位与止损。\n"
                    f"策略说明：{ma_txt}\n"
                    f"风控建议：可叠加成交量/RSI/布林带过滤、设置固定或ATR止损、使用多周期共振（周线定方向，{period_txt}择时）。以上仅为方法参考，非投资建议。"
                )
                st.success("中文总结已生成：")
                st.write(summary)
            except Exception as e:
                st.warning(f"生成中文总结失败：{e}")

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
    user_query = st.text_area("问模型：个股/行业信息、策略建议...", value="")
    if user_query:
        try:
            sys_prompt = {"role": "system", "content": "你是资深量化分析师。可以结合已知数据做基本面与技术面分析，并提醒数据来源和不构成投资建议。必要时请使用可用的联网工具（function calling）查询个股/行业实时信息，避免臆测。"}
            messages = [sys_prompt]
            if inject_ctx and df_disp is not None and not df_disp.empty:
                try:
                    latest_close = float(df_disp["close"].iloc[-1])
                    latest_vol = float(df_disp["volume"].iloc[-1]) if pd.notna(df_disp["volume"].iloc[-1]) else 0
                    ctx_lines = [
                        f"页面: 单股查询",
                        f"市场: {market}",
                        f"股票代码: {symbol}",
                        f"股票名称: {stock_name or ''}",
                        f"周期: {period}",
                        f"展示区间: {str(ds)} ~ {str(de)}" if 'ds' in locals() and 'de' in locals() else "展示区间: 未知",
                        f"MA窗口: {','.join(map(str, ma_windows)) if (isinstance(ma_windows, list) and len(ma_windows)>0) else '关闭'}",
                        f"MACD: {'开启' if show_macd else '关闭'}",
                        f"最新收盘: {latest_close}",
                        f"最新成交量: {latest_vol}",
                    ]
                    # 尝试注入行业信息（A股）
                    if market == "A":
                        try:
                            info_df = ak.stock_individual_info_em(symbol=symbol)
                            if info_df is not None and not info_df.empty:
                                col0, col1 = info_df.columns[:2]
                                for key in ["所属行业", "行业", "行业分类", "细分行业"]:
                                    _row = info_df[info_df[col0].astype(str).str.contains(key, na=False)]
                                    if not _row.empty:
                                        ind_name = str(_row[col1].iloc[0])
                                        if ind_name:
                                            ctx_lines.append(f"所属行业: {ind_name}")
                                        break
                        except Exception:
                            pass
                    # 指标与回测摘要
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
                    ctx_lines.append("如需查询个股或行业的实时信息，请按市场选择工具：A股用 fetch_stock_info_a，港股用 fetch_stock_info_hk，并传入当前 symbol；热点题材/新闻亦可通过工具检索。")
                    context_str = "\n".join(ctx_lines)
                    messages.append({"role": "system", "content": "以下是当前页面上下文，请结合回答问题：\n" + context_str})
                except Exception:
                    pass
            user_msg = {"role": "user", "content": user_query}
            messages.append(user_msg)

            # 读取路由/模型设置
            route_name = st.session_state.get("route_name", "default")
            tools = get_tools_schema()
            registry = ProviderRegistry(public_cfg_path="models.yaml", local_cfg_path="models.local.yaml")
            router = LLMRouter(registry=registry, route_name=route_name)

            result = chat_with_tools(router, messages, tools_schema=tools, max_rounds=3)
            final_text = result.get("final_text") or ""
            if final_text.strip():
                st.markdown(final_text)
            else:
                _render_llm_answer(result)
        except Exception as e:
            st.error(f"调用模型失败: {e}")
    if False and user_query:
        try:
            sys_prompt = {"role": "system", "content": "你是资深量化分析师。可以结合已知数据做基本面与技术面分析，并提醒数据来源和不构成投资建议。必要时请使用可用的联网工具（function calling）查询个股/行业实时信息，避免臆测。"}
            messages = [sys_prompt]
            if inject_ctx and df_disp is not None and not df_disp.empty:
                try:
                    latest_close = float(df_disp["close"].iloc[-1])
                    latest_vol = float(df_disp["volume"].iloc[-1]) if pd.notna(df_disp["volume"].iloc[-1]) else 0
                    ctx_lines = [
                        f"最新收盘: {latest_close}",
                        f"最新成交量: {latest_vol}",
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

            # 读取路由/模型设置
            route_name = st.session_state.get("route_name", "default")
            tools = get_tools_schema()
            registry = ProviderRegistry(public_cfg_path="models.yaml", local_cfg_path="models.local.yaml")
            router = LLMRouter(registry, route_name=route_name)
            messages = [
                {"role":"system","content":"你是资深行业分析师。给出条理清晰、可执行的行业研判，不构成投资建议。必要时请使用可用的联网工具（function calling）查询个股/行业实时信息。"},
                {"role":"system","content":"当前行业上下文：\n" + "\n".join(ctx_lines)},
                {"role":"user","content": prompt},
            ]
            tools = get_tools_schema()
            result = chat_with_tools(router, messages, tools_schema=tools, max_rounds=3)
            final_text = result.get("final_text") or ""
            if final_text.strip():
                st.markdown(final_text)
            else:
                _render_llm_answer(result)
        except Exception as e:
            st.error(f"行业分析失败：{e}")
    # 重复的行业分析问答块已移除（避免与 industry_page 重复，并消除未定义变量 prompt）

# -------- 自选股页面 --------
def watchlist_page():
    st.header("自选股")
    items = load_watchlist()

    with st.form("add_watch_item"):
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            mkt = st.selectbox("市场", ["A","H"], key="wl_market")
        with c2:
            sym = st.text_input("股票代码", key="wl_symbol")
        submitted = st.form_submit_button("添加到自选")
        if submitted:
            sym = (sym or "").strip()
            if sym:
                if not any((it.get("market"), it.get("symbol")) == (mkt, sym) for it in items):
                    items.append({"market": mkt, "symbol": sym})
                    save_watchlist(items)
                    st.success("已添加到自选")
                else:
                    st.info("自选中已存在该代码")
            else:
                st.warning("请输入代码")

    if not items:
        st.info("暂无自选。可以在上方添加股票代码。")
        return

    # 展示与操作
    for i, it in enumerate(list(items)):
        market = it.get("market")
        symbol = it.get("symbol")
        name = get_stock_name_cached(market, symbol)
        c1, c2, c3, c4 = st.columns([2,2,2,2])
        c1.write(f"{name or ''} ({symbol})")
        c2.write(f"市场：{market}")
        if c3.button("查看", key=f"wl_view_{i}_{market}_{symbol}"):
            go_detail(market, symbol)
        if c4.button("删除", key=f"wl_del_{i}_{market}_{symbol}"):
            items = [x for x in items if not ((x.get('market')==market) and (x.get('symbol')==symbol))]
            save_watchlist(items)
            st.experimental_rerun()


# -------- 行业信息页面 --------
def industry_page():
    st.header("行业信息")
    kw = st.text_input("行业/主题关键词", value=st.session_state.get("industry_keyword", "半导体"))
    st.session_state["industry_keyword"] = kw
    inject_ctx = st.checkbox("注入行业上下文", value=True)

    user_query = st.text_area("问模型：行业逻辑、景气度、龙头比较、估值与风险…", value="")
    if user_query:
        try:
            sys_prompt = {"role":"system","content":"你是资深行业分析师。给出条理清晰、可执行的行业研判，不构成投资建议。必要时请使用可用的联网工具（function calling）查询个股/行业实时信息。"}
            messages = [sys_prompt]
            if inject_ctx:
                ctx_lines = [
                    "页面: 行业信息",
                    f"行业关键词: {kw}",
                    "如需获取成份股或个股数据，可按市场调用工具：A股用 fetch_stock_info_a，港股用 fetch_stock_info_hk。",
                ]
                messages.append({"role":"system","content":"以下是当前页面上下文：\n" + "\n".join(ctx_lines)})
            messages.append({"role":"user","content": user_query})

            route_name = st.session_state.get("route_name","default")
            registry = ProviderRegistry(public_cfg_path="models.yaml", local_cfg_path="models.local.yaml")
            router = LLMRouter(registry=registry, route_name=route_name)
            tools = get_tools_schema()
            result = chat_with_tools(router, messages, tools_schema=tools, max_rounds=3)
            final_text = result.get("final_text") or ""
            if final_text.strip():
                st.markdown(final_text)
            else:
                _render_llm_answer(result)
        except Exception as e:
            st.error(f"行业分析失败：{e}")


# -------- 工具筛选页面（示例） --------
def tools_filter_page():
    st.header("工具筛选（示例）")
    st.info("示例：批量把 A 股代码加入自选。逗号分隔输入即可。")
    codes = st.text_input("A 股代码（逗号分隔）", value="")
    if st.button("批量加入自选", key="bulk_add_watchlist"):
        items = load_watchlist()
        added = 0
        for sym in [c.strip() for c in codes.split(',') if c.strip()]:
            if not any((it.get('market'), it.get('symbol')) == ("A", sym) for it in items):
                items.append({"market":"A","symbol": sym})
                added += 1
        save_watchlist(items)
        st.success(f"已加入 {added} 个代码到自选")

# -------- 新的入口：四大模块 Tabs --------
def main():
    st.set_page_config(page_title="A/H 股票分析系统", layout="wide")
    st.title("A/H 股票分析系统")
    st.caption("数据源：AKShare | 绘图：Plotly | 前端：Streamlit")

    with st.sidebar:
        st.header("大模型配置")
        provider_label = st.selectbox("提供商(路由)", ["default","fast","qwen","fallback","analysis"], index=0, help="使用 routing.yaml 中的路由名", key="provider_route")
        st.session_state["route_name"] = provider_label
        st.session_state["enable_tools"] = st.checkbox("启用联网工具(Function Calling)", value=True)
        st.markdown("—")
        st.header("导航")
        st.info("上次筛选/自选/查询的股票可跳转至详情页。")

    tabs = st.tabs(["单股查询", "自选股", "行业信息", "工具筛选"])
    with tabs[0]:
        single_stock_page()
    with tabs[1]:
        watchlist_page()
    with tabs[2]:
        industry_page()
    with tabs[3]:
        tools_filter_page()

if __name__ == "__main__":
    main()