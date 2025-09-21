# -*- coding: utf-8 -*-
# 股票分析系统入口（Streamlit）
import streamlit as st
# 移除不被当前版本支持的序列化配置项，避免 StreamlitAPIException
# st.set_option("global.dataFrameSerialization", "legacy")
import pandas as pd
import akshare as ak
import sys, os
from pathlib import Path
from typing import Optional, List, Dict, Any
import importlib
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any as _Any
from pandas.api.types import is_bool_dtype, is_object_dtype

# Arrow 兼容工具：将布尔列转为字符串、对象列统一转字符串，避免 pyarrow 转换报错

def ensure_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df is None or getattr(df, "empty", True):
            return df
        df2 = df.copy()
        for col in df2.columns:
            s = df2[col]
            if is_bool_dtype(s):
                df2[col] = s.map(lambda x: "是" if bool(x) else "否")
            elif is_object_dtype(s):
                df2[col] = s.astype(str)
        return df2
    except Exception:
        return df

# 项目根目录路径（供数据/模块导入）
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- DEMO 模式：自动清空缓存 -----------------------------------------------
# 触发条件：
# 1) 环境变量：LOCAL_DEMO / DEMO_MODE / STREAMLIT_DEMO 任意为 1/true/yes/on
# 2) URL 参数：?demo=1 或 true/yes/on
# 仅在每个会话首次触发时清空，避免重复清理影响性能

def _is_truthy(v: Any) -> bool:
    try:
        return str(v).strip().lower() in ("1", "true", "yes", "on", "y")
    except Exception:
        return False


def auto_clear_caches_for_demo():
    """在本地 DEMO 模式下自动清空 Streamlit 缓存（数据与资源）。
    - 不清空 session_state，避免打断用户交互。
    """
    try:
        # 已清理则跳过
        if st.session_state.get("_demo_caches_cleared"):
            return

        # 检测环境变量
        env_flags = ["LOCAL_DEMO", "DEMO_MODE", "STREAMLIT_DEMO"]
        env_on = any(_is_truthy(os.environ.get(k, "")) for k in env_flags)

        # 检测 URL 参数
        demo_qp_on = False
        try:
            # 兼容不同版本的 Streamlit 查询参数 API
            if hasattr(st, "query_params"):
                qps = st.query_params or {}
            else:
                qps = st.experimental_get_query_params() or {}
            dq = qps.get("demo")
            if isinstance(dq, list):
                dq = dq[0] if dq else None
            demo_qp_on = _is_truthy(dq)
        except Exception:
            demo_qp_on = False

        if env_on or demo_qp_on:
            try:
                st.cache_data.clear()
            except Exception:
                pass
            try:
                st.cache_resource.clear()
            except Exception:
                pass
            st.session_state["_demo_caches_cleared"] = True
            try:
                st.toast("已自动清空缓存（DEMO 模式）", icon="🧹")
            except Exception:
                st.info("已自动清空缓存（DEMO 模式）")
    except Exception:
        # 任何错误都不影响主流程
        pass

# 应用启动时尝试进行 DEMO 缓存清理（满足触发条件才会执行）
auto_clear_caches_for_demo()

# 行业自选列表持久化
INDUSTRY_WATCHLIST_PATH = ROOT / "data" / "industry_watchlist.json"
# 自定义行业数据路径
CUSTOM_INDUSTRIES_PATH = ROOT / "data" / "custom_industries.json"

def _ensure_industry_data_dir():
    (ROOT / "data").mkdir(parents=True, exist_ok=True)

# 自定义行业读写
def load_custom_industries() -> Dict[str, List[Dict[str, str]]]:
    _ensure_industry_data_dir()
    try:
        if CUSTOM_INDUSTRIES_PATH.exists():
            with open(CUSTOM_INDUSTRIES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                out: Dict[str, List[Dict[str, str]]] = {}
                for k, v in data.items():
                    if not isinstance(k, str) or not isinstance(v, list):
                        continue
                    items: List[Dict[str, str]] = []
                    for it in v:
                        if isinstance(it, dict) and "symbol" in it:
                            market = str(it.get("market", "A")).strip() or "A"
                            sym = str(it.get("symbol", "")).strip()
                            nm = str(it.get("name", "")).strip()
                            if sym:
                                items.append({"market": market, "symbol": sym, "name": nm})
                    out[k] = items
                return out
    except Exception:
        pass
    return {}

def save_custom_industries(data: Dict[str, List[Dict[str, str]]]):
    _ensure_industry_data_dir()
    try:
        with open(CUSTOM_INDUSTRIES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# 统一行业列表：官方 + 自定义
def get_industry_list_all() -> List[str]:
    base = ak_get_industry_list() or []
    custom = load_custom_industries()
    seen = set()
    out: List[str] = []
    for n in base:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    for n in custom.keys():
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out

# 统一成份股：优先自定义（仅当有效），否则回退官方
def get_industry_cons(industry_name: str) -> pd.DataFrame:
    custom = load_custom_industries()
    if industry_name in custom:
        items = custom[industry_name]
        # 仅当自定义有有效 symbol 才使用，否则回退官方
        if items:
            df = pd.DataFrame(items)
            if "symbol" in df.columns and df["symbol"].notna().any():
                if "name" not in df.columns:
                    df["name"] = ""
                df["symbol"] = df["symbol"].astype(str).str.strip()
                # 规范化为6位数字，兼容 000001.SZ / sz000001 / 空格等
                try:
                    sym6 = df["symbol"].str.extract(r"(\d{6})", expand=False)
                    df["symbol"] = sym6.fillna(df["symbol"]).astype(str).str.strip()
                except Exception:
                    pass
                df["name"] = df["name"].astype(str).str.strip()
                return df[["symbol", "name"]].copy()
    # 回退官方/概念成份
    return ak_get_industry_cons(industry_name)

def load_industry_watchlist() -> List[str]:
    _ensure_industry_data_dir()
    try:
        if INDUSTRY_WATCHLIST_PATH.exists():
            with open(INDUSTRY_WATCHLIST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # 去重并保持顺序
                seen = set()
                out = []
                for x in data:
                    if isinstance(x, str) and x not in seen:
                        seen.add(x)
                        out.append(x)
                return out
    except Exception:
        pass
    return []

def save_industry_watchlist(items: List[str]):
    _ensure_industry_data_dir()
    try:
        with open(INDUSTRY_WATCHLIST_PATH, "w", encoding="utf-8") as f:
            json.dump(items[:50], f, ensure_ascii=False, indent=2)
    except Exception:
        pass

@st.cache_data(ttl=3600)
def get_ths_industry_index_map() -> Dict[str, str]:
    """
    获取同花顺行业板块 名称->代码 映射。
    - 优先使用 ak.stock_board_industry_name_ths（通常包含行业板块名称与代码）
    - 自动兼容不同版本的列名（如：行业名称/板块名称/名称；代码/板块代码/指数代码/code）
    - 返回示例：{"人工智能": "885XXX", ...}
    """
    import akshare as ak
    m: Dict[str, str] = {}
    try:
        if hasattr(ak, "stock_board_industry_name_ths"):
            df = getattr(ak, "stock_board_industry_name_ths")()
            if df is not None and not df.empty:
                name_col = None
                code_col = None
                for c in ["行业名称", "板块名称", "板块简称", "名称", "name"]:
                    if c in df.columns:
                        name_col = c; break
                for c in ["代码", "板块代码", "指数代码", "symbol", "code"]:
                    if c in df.columns:
                        code_col = c; break
                if name_col and code_col:
                    for _, r in df[[name_col, code_col]].dropna().iterrows():
                        nm = str(r[name_col]).strip()
                        cd = str(r[code_col]).strip()
                        if nm and cd:
                            m[nm] = cd
    except Exception:
        pass
    return m

@st.cache_data(ttl=3600)
def ak_get_industry_list() -> List[str]:
    import akshare as ak
    # 统一返回“行业板块 ∪ 概念板块”的去重并集
    out: List[str] = []
    seen = set()
    try:
        # 1) 同花顺 行业板块
        ths_map = get_ths_industry_index_map()
        if ths_map:
            for n in ths_map.keys():
                n = str(n).strip()
                if n and n not in seen:
                    seen.add(n)
                    out.append(n)
        # 2) 东方财富 行业列表
        df = None
        for fn in [getattr(ak, n) for n in [
            "stock_board_industry_name_url",
            "stock_board_industry_name_em",
        ] if hasattr(ak, n)]:
            try:
                df = fn()
                if df is not None and not df.empty:
                    break
            except Exception:
                continue
        if df is not None and not df.empty:
            candidate_cols = ["行业名称", "板块名称", "板块简称", "名称", "name"]
            col = next((c for c in candidate_cols if c in df.columns), df.columns[0])
            for v in df[col].dropna().tolist():
                n = str(v).strip()
                if n and n not in seen:
                    seen.add(n)
                    out.append(n)
        # 3) 东方财富 概念列表
        df2 = None
        for fn2 in [getattr(ak, n) for n in [
            "stock_board_concept_name_em",
            "stock_board_concept_name_url",
        ] if hasattr(ak, n)]:
            try:
                df2 = fn2()
                if df2 is not None and not df2.empty:
                    break
            except Exception:
                continue
        if df2 is not None and not df2.empty:
            candidate_cols2 = ["概念名称", "板块名称", "板块简称", "名称", "name"]
            col2 = next((c for c in candidate_cols2 if c in df2.columns), df2.columns[0])
            for v in df2[col2].dropna().tolist():
                n = str(v).strip()
                if n and n not in seen:
                    seen.add(n)
                    out.append(n)
    except Exception:
        out = []

    # 若依然为空，回退到本地自定义行业；仍为空则提供少量内置常见板块，避免下拉为空
    if not out:
        try:
            local_names = list(load_custom_industries().keys())
        except Exception:
            local_names = []
        if local_names:
            out = []
            seen2 = set()
            for n in local_names:
                n = str(n).strip()
                if n and n not in seen2:
                    seen2.add(n)
                    out.append(n)
        else:
            out = [
                "电池", "半导体", "光伏设备", "储能", "风电设备",
                "人工智能", "机器人", "券商", "白酒", "煤炭", "有色金属",
            ]
    return out

@st.cache_data(ttl=1800)
def ak_get_industry_cons(industry_name: str) -> pd.DataFrame:
    import akshare as ak
    df = pd.DataFrame()
    # 先尝试：同花顺 行业板块成份（名称->代码->成份）
    try:
        ths_map = get_ths_industry_index_map()
        if ths_map and industry_name in ths_map and hasattr(ak, "stock_board_industry_cons_ths"):
            bk_code = ths_map[industry_name]
            try:
                df = getattr(ak, "stock_board_industry_cons_ths")(symbol=bk_code)
            except TypeError:
                try:
                    df = getattr(ak, "stock_board_industry_cons_ths")(industry=bk_code)
                except Exception:
                    df = getattr(ak, "stock_board_industry_cons_ths")(bk_code)
    except Exception:
        df = pd.DataFrame()
    # 次之：东方财富 行业成份
    try:
        if hasattr(ak, "stock_board_industry_cons_em"):
            fn = getattr(ak, "stock_board_industry_cons_em")
            try:
                df = fn(symbol=industry_name)
            except TypeError:
                try:
                    df = fn(industry=industry_name)
                except Exception:
                    try:
                        df = fn(industry_name)
                    except Exception:
                        df = pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    # 若行业为空，进一步尝试：东方财富 概念成份（兼容“电池”等概念板块）
    if df is None or df.empty:
        candidates = [industry_name]
        if not str(industry_name).endswith("概念"):
            candidates.append(f"{industry_name}概念")
        for nm in candidates:
            got = pd.DataFrame()
            try:
                # 1) 东方财富 概念成份（多个参数名/位置参数兼容）
                if hasattr(ak, "stock_board_concept_cons_em"):
                    fn2 = getattr(ak, "stock_board_concept_cons_em")
                    try:
                        got = fn2(symbol=nm)
                    except TypeError:
                        try:
                            got = fn2(concept=nm)
                        except Exception:
                            try:
                                got = fn2(nm)
                            except Exception:
                                got = pd.DataFrame()
                # 2) 同花顺 概念成份（直接用名称尝试）
                if (got is None or got.empty) and hasattr(ak, "stock_board_concept_cons_ths"):
                    fn3 = getattr(ak, "stock_board_concept_cons_ths")
                    try:
                        got = fn3(symbol=nm)
                    except TypeError:
                        try:
                            got = fn3(concept=nm)
                        except Exception:
                            try:
                                got = fn3(nm)
                            except Exception:
                                got = pd.DataFrame()
                # 3) 同花顺 概念成份（名称->代码映射再取成份）
                if (got is None or got.empty) and hasattr(ak, "stock_board_concept_name_ths") and hasattr(ak, "stock_board_concept_cons_ths"):
                    try:
                        name_df = getattr(ak, "stock_board_concept_name_ths")()
                    except Exception:
                        name_df = None
                    if name_df is not None and not name_df.empty:
                        # 兼容列名
                        name_col = None
                        code_col = None
                        for c in ["概念名称", "名称", "板块名称", "name"]:
                            if c in name_df.columns:
                                name_col = c; break
                        for c in ["代码", "板块代码", "指数代码", "symbol", "code"]:
                            if c in name_df.columns:
                                code_col = c; break
                        if name_col and code_col:
                            # 在名称里匹配 nm 或 nm+概念
                            try:
                                row = name_df[(name_df[name_col] == nm) | (name_df[name_col] == f"{nm}概念")].iloc[0]
                                bk_code = str(row[code_col]).strip()
                                try:
                                    got = getattr(ak, "stock_board_concept_cons_ths")(symbol=bk_code)
                                except TypeError:
                                    try:
                                        got = getattr(ak, "stock_board_concept_cons_ths")(concept=bk_code)
                                    except Exception:
                                        got = getattr(ak, "stock_board_concept_cons_ths")(bk_code)
                            except Exception:
                                pass
            except Exception:
                got = pd.DataFrame()
            if got is not None and not got.empty:
                df = got
                break

    # 4) 行业方向再兜底：同花顺行业（名称->代码->成份）
    if (df is None or df.empty) and hasattr(ak, "stock_board_industry_name_ths") and hasattr(ak, "stock_board_industry_cons_ths"):
        try:
            ind_list = getattr(ak, "stock_board_industry_name_ths")()
        except Exception:
            ind_list = None
        if ind_list is not None and not ind_list.empty:
            name_col2 = None
            code_col2 = None
            for c in ["行业名称", "板块名称", "板块简称", "名称", "name"]:
                if c in ind_list.columns:
                    name_col2 = c; break
            for c in ["代码", "板块代码", "指数代码", "symbol", "code"]:
                if c in ind_list.columns:
                    code_col2 = c; break
            if name_col2 and code_col2:
                try:
                    row2 = ind_list[(ind_list[name_col2] == industry_name)].iloc[0]
                    bk_code2 = str(row2[code_col2]).strip()
                    try:
                        df = getattr(ak, "stock_board_industry_cons_ths")(symbol=bk_code2)
                    except TypeError:
                        try:
                            df = getattr(ak, "stock_board_industry_cons_ths")(industry=bk_code2)
                        except Exception:
                            df = getattr(ak, "stock_board_industry_cons_ths")(bk_code2)
                except Exception:
                    pass

    if df is None:
        df = pd.DataFrame()
    # 标准化代码/名称列
    code_col = None
    name_col = None
    for c in ["代码", "股票代码", "证券代码", "code"]:
        if c in df.columns:
            code_col = c
            break
    for c in ["名称", "股票名称", "证券简称", "name"]:
        if c in df.columns:
            name_col = c
            break
    if code_col is None:
        df["代码"] = []
        code_col = "代码"
    if name_col is None:
        df["名称"] = []
        name_col = "名称"
    out = df[[code_col, name_col]].rename(columns={code_col: "symbol", name_col: "name"}).copy()
    # 代码统一为字符串
    out["symbol"] = out["symbol"].astype(str).str.strip()
    # 若包含交易所后缀或其他字符，优先提取6位数字作为A股代码
    try:
        sym6 = out["symbol"].str.extract(r"(\d{6})", expand=False)
        out["symbol"] = sym6.fillna(out["symbol"]).astype(str).str.strip()
    except Exception:
        pass
    return out

@st.cache_data(ttl=1200)
def compute_symbol_volume_metrics(market: str, symbol: str, N: int) -> Dict[str, Any]:
    client = get_client()
    df = client.get_hist(market=market, symbol=symbol, period="daily")
    # 若字段缺失或全为0/NA，尝试强制刷新一次
    if df is None or df.empty or "volume" not in df.columns or pd.to_numeric(df.get("volume", pd.Series([])), errors="coerce").fillna(0).sum() == 0:
        df2 = client.get_hist(market=market, symbol=symbol, period="daily", refresh=True, expire_days=0)
        if df2 is not None and not df2.empty:
            df = df2
    if df is None or df.empty or "volume" not in df.columns:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    x = df.sort_values("date").copy()
    x["volume"] = pd.to_numeric(x["volume"], errors="coerce").fillna(0)
    if len(x) < N + 2:
        N = max(1, min(N, len(x)))
    curr = float(x["volume"].tail(N).sum())
    # 环比：前一N日
    prev = float(x["volume"].tail(N*2).head(N).sum()) if len(x) >= N*2 else None
    # 同比：去年同期N日（按交易日回溯约 252 天）
    try:
        last_date = x["date"].iloc[-1]
        one_year_ago = last_date - pd.Timedelta(days=365)
        ywin = x[(x["date"] <= one_year_ago)].tail(N)
        yoy = float(ywin["volume"].sum()) if not ywin.empty else None
    except Exception:
        yoy = None
    yoy_pct = (curr - yoy) / yoy if (yoy and yoy > 0) else None
    mom_pct = (curr - prev) / prev if (prev and prev > 0) else None
    return {"curr": curr, "yoy": yoy, "prev": prev, "yoy_pct": yoy_pct, "mom_pct": mom_pct}

@st.cache_data(ttl=1200)
def compute_industry_volume_metrics(industry_name: str, N: int) -> Dict[str, Any]:
    cons = get_industry_cons(industry_name)
    if cons is None or cons.empty:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None, "leaders": [], "count": 0}
    curr_sum = 0.0
    prev_sum = 0.0
    yoy_sum = 0.0
    prev_has = False
    yoy_has = False
    leaders: List[Dict[str, Any]] = []
    for _, row in cons.iterrows():
        sym = str(row.get("symbol", "")).strip()
        name = str(row.get("name", "")).strip()
        if not sym:
            continue
        m = compute_symbol_volume_metrics("A", sym, N)
        curr_sum += m.get("curr") or 0.0
        if m.get("prev") is not None:
            prev_sum += m["prev"] or 0.0
            prev_has = True
        if m.get("yoy") is not None:
            yoy_sum += m["yoy"] or 0.0
            yoy_has = True
        leaders.append({"symbol": sym, "name": name, "curr": m.get("curr", 0.0)})
    # 选龙头：按近N日成交量排序取前5
    leaders = sorted(leaders, key=lambda x: x.get("curr", 0.0), reverse=True)[:5]
    yoy_val = yoy_sum if yoy_has else None
    prev_val = prev_sum if prev_has else None
    yoy_pct = (curr_sum - yoy_val) / yoy_val if (yoy_val and yoy_val > 0) else None
    mom_pct = (curr_sum - prev_val) / prev_val if (prev_val and prev_val > 0) else None
    return {"curr": curr_sum, "yoy": yoy_val, "prev": prev_val, "yoy_pct": yoy_pct, "mom_pct": mom_pct, "leaders": leaders, "count": int(len(cons))}

@st.cache_data(ttl=1200)
def _compute_symbol_window_metrics(market: str, symbol: str, N: int, field: str) -> Dict[str, Any]:
    client = get_client()
    df = client.get_hist(market=market, symbol=symbol, period="daily")
    # 若字段缺失或全为0/NA，尝试强制刷新一次
    if df is None or df.empty or field not in df.columns or pd.to_numeric(df.get(field, pd.Series([])), errors="coerce").fillna(0).sum() == 0:
        df2 = client.get_hist(market=market, symbol=symbol, period="daily", refresh=True, expire_days=0)
        if df2 is not None and not df2.empty:
            df = df2
    if df is None or df.empty or field not in df.columns:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    x = df.sort_values("date").copy()
    x[field] = pd.to_numeric(x[field], errors="coerce").fillna(0)
    if len(x) < N + 2:
        N = max(1, min(N, len(x)))
    curr = float(x[field].tail(N).sum())
    prev = float(x[field].tail(N*2).head(N).sum()) if len(x) >= N*2 else None
    try:
        last_date = x["date"].iloc[-1]
        one_year_ago = last_date - pd.Timedelta(days=365)
        ywin = x[(x["date"] <= one_year_ago)].tail(N)
        yoy = float(ywin[field].sum()) if not ywin.empty else None
    except Exception:
        yoy = None
    yoy_pct = (curr - yoy) / yoy if (yoy and yoy > 0) else None
    mom_pct = (curr - prev) / prev if (prev and prev > 0) else None
    return {"curr": curr, "yoy": yoy, "prev": prev, "yoy_pct": yoy_pct, "mom_pct": mom_pct}

# ---- 基于“时间周期”的统计 ----
@st.cache_data(ttl=1200)
def _compute_symbol_period_metrics(market: str, symbol: str, start_date: _Any, end_date: _Any, field: str = "volume") -> Dict[str, Any]:
    client = get_client()
    df = client.get_hist(market=market, symbol=symbol, period="daily")
    # 若字段缺失或全为0/NA，尝试强制刷新一次
    if df is None or df.empty or field not in df.columns or pd.to_numeric(df.get(field, pd.Series([])), errors="coerce").fillna(0).sum() == 0:
        df2 = client.get_hist(market=market, symbol=symbol, period="daily", refresh=True, expire_days=0)
        if df2 is not None and not df2.empty:
            df = df2
    if df is None or df.empty or field not in df.columns:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    x = df.sort_values("date").copy()
    x[field] = pd.to_numeric(x[field], errors="coerce").fillna(0)
    try:
        s = pd.to_datetime(start_date)
        e = pd.to_datetime(end_date)
    except Exception:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    if pd.isna(s) or pd.isna(e):
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    if s > e:
        s, e = e, s
    cur_win = x[(x["date"] >= s) & (x["date"] <= e)]
    if cur_win is None or cur_win.empty:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    L = len(cur_win)
    curr = float(cur_win[field].sum())
    # 环比：紧挨着上一个同长度交易日窗口
    prev_win = x[x["date"] < s].tail(L)
    prev = float(prev_win[field].sum()) if not prev_win.empty else None
    # 同比：去年同一日期区间
    try:
        s_py = s - pd.Timedelta(days=365)
        e_py = e - pd.Timedelta(days=365)
        yoy_win = x[(x["date"] >= s_py) & (x["date"] <= e_py)]
        yoy = float(yoy_win[field].sum()) if not yoy_win.empty else None
    except Exception:
        yoy = None
    yoy_pct = (curr - yoy) / yoy if (yoy and yoy > 0) else None
    mom_pct = (curr - prev) / prev if (prev and prev > 0) else None
    return {"curr": curr, "yoy": yoy, "prev": prev, "yoy_pct": yoy_pct, "mom_pct": mom_pct}

@st.cache_data(ttl=1200)
def compute_symbol_volume_metrics_period(market: str, symbol: str, start_date: _Any, end_date: _Any) -> Dict[str, Any]:
    return _compute_symbol_period_metrics(market, symbol, start_date, end_date, "volume")

@st.cache_data(ttl=1200)
def compute_symbol_amount_metrics_period(market: str, symbol: str, start_date: _Any, end_date: _Any) -> Dict[str, Any]:
    return _compute_symbol_period_metrics(market, symbol, start_date, end_date, "amount")

@st.cache_data(ttl=1200)
def compute_industry_volume_metrics_period(industry_name: str, start_date: _Any, end_date: _Any) -> Dict[str, Any]:
    cons = get_industry_cons(industry_name)
    if cons is None or cons.empty:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None, "leaders": [], "count": 0}
    curr_sum = 0.0
    prev_sum = 0.0
    yoy_sum = 0.0
    prev_has = False
    yoy_has = False
    leaders: List[Dict[str, Any]] = []
    for _, row in cons.iterrows():
        sym = str(row.get("symbol", "")).strip()
        name = str(row.get("name", "")).strip()
        if not sym:
            continue
        m = compute_symbol_volume_metrics_period("A", sym, start_date, end_date)
        curr_sum += m.get("curr") or 0.0
        if m.get("prev") is not None:
            prev_sum += m["prev"] or 0.0
            prev_has = True
        if m.get("yoy") is not None:
            yoy_sum += m["yoy"] or 0.0
            yoy_has = True
        leaders.append({"symbol": sym, "name": name, "curr": m.get("curr", 0.0)})
    leaders = sorted(leaders, key=lambda x: x.get("curr", 0.0), reverse=True)[:5]
    yoy_val = yoy_sum if yoy_has else None
    prev_val = prev_sum if prev_has else None
    yoy_pct = (curr_sum - yoy_val) / yoy_val if (yoy_val and yoy_val > 0) else None
    mom_pct = (curr_sum - prev_val) / prev_val if (prev_val and prev_val > 0) else None
    return {"curr": curr_sum, "yoy": yoy_val, "prev": prev_val, "yoy_pct": yoy_pct, "mom_pct": mom_pct, "leaders": leaders, "count": int(len(cons))}

@st.cache_data(ttl=1200)
def compute_industry_amount_metrics_period(industry_name: str, start_date: _Any, end_date: _Any) -> Dict[str, Any]:
    cons = get_industry_cons(industry_name)
    if cons is None or cons.empty:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    curr_sum = 0.0
    prev_sum = 0.0
    yoy_sum = 0.0
    prev_has = False
    yoy_has = False
    for _, row in cons.iterrows():
        sym = str(row.get("symbol", "")).strip()
        if not sym:
            continue
        m = compute_symbol_amount_metrics_period("A", sym, start_date, end_date)
        curr_sum += m.get("curr") or 0.0
        if m.get("prev") is not None:
            prev_sum += m["prev"] or 0.0
            prev_has = True
        if m.get("yoy") is not None:
            yoy_sum += m["yoy"] or 0.0
            yoy_has = True
    yoy_val = yoy_sum if yoy_has else None
    prev_val = prev_sum if prev_has else None
    yoy_pct = (curr_sum - yoy_val) / yoy_val if (yoy_val and yoy_val > 0) else None
    mom_pct = (curr_sum - prev_val) / prev_val if (prev_val and prev_val > 0) else None
    return {"curr": curr_sum, "yoy": yoy_val, "prev": prev_val, "yoy_pct": yoy_pct, "mom_pct": mom_pct}

@st.cache_data(ttl=1200)
def compute_symbol_amount_metrics(market: str, symbol: str, N: int) -> Dict[str, Any]:
    return _compute_symbol_window_metrics(market, symbol, N, "amount")

@st.cache_data(ttl=1200)
def compute_industry_amount_metrics(industry_name: str, N: int) -> Dict[str, Any]:
    cons = get_industry_cons(industry_name)
    if cons is None or cons.empty:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    curr_sum = 0.0
    prev_sum = 0.0
    yoy_sum = 0.0
    prev_has = False
    yoy_has = False
    for _, row in cons.iterrows():
        sym = str(row.get("symbol", "")).strip()
        if not sym:
            continue
        m = compute_symbol_amount_metrics("A", sym, N)
        curr_sum += m.get("curr") or 0.0
        if m.get("prev") is not None:
            prev_sum += m["prev"] or 0.0
            prev_has = True
        if m.get("yoy") is not None:
            yoy_sum += m["yoy"] or 0.0
            yoy_has = True
    yoy_val = yoy_sum if yoy_has else None
    prev_val = prev_sum if prev_has else None
    yoy_pct = (curr_sum - yoy_val) / yoy_val if (yoy_val and yoy_val > 0) else None
    mom_pct = (curr_sum - prev_val) / prev_val if (prev_val and prev_val > 0) else None
    return {"curr": curr_sum, "yoy": yoy_val, "prev": prev_val, "yoy_pct": yoy_pct, "mom_pct": mom_pct}

@st.cache_data(ttl=1200)
def compute_industry_agg_series(industry_name: str, column: str, days: int = 60, start_date: _Any = None, end_date: _Any = None) -> pd.DataFrame:
    cons = get_industry_cons(industry_name)
    if cons is None or cons.empty:
        return pd.DataFrame(columns=["date", column])
    agg = None
    client = get_client()
    s_dt = pd.to_datetime(start_date) if start_date is not None else None
    e_dt = pd.to_datetime(end_date) if end_date is not None else None
    for _, row in cons.iterrows():
        sym = str(row.get("symbol", "")).strip()
        if not sym:
            continue
        df = client.get_hist(market="A", symbol=sym, period="daily")
        # 若字段缺失或全为0/NA，尝试强制刷新一次
        if df is None or df.empty or column not in df.columns or pd.to_numeric(df.get(column, pd.Series([])), errors="coerce").fillna(0).sum() == 0:
            df2 = client.get_hist(market="A", symbol=sym, period="daily", refresh=True, expire_days=0)
            if df2 is not None and not df2.empty:
                df = df2
        if df is None or df.empty or column not in df.columns:
            continue
        x = df[["date", column]].copy().dropna()
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        x[column] = pd.to_numeric(x[column], errors="coerce").fillna(0)
        if s_dt is not None and e_dt is not None:
            x = x[(x["date"] >= s_dt) & (x["date"] <= e_dt)]
        else:
            x = x.sort_values("date").tail(max(days*2, days))
        x = x.set_index("date").rename(columns={column: sym})
        agg = x if agg is None else agg.join(x, how="outer")
    if agg is None or agg.empty:
        return pd.DataFrame(columns=["date", column])
    s = agg.fillna(0).sum(axis=1).reset_index()
    s.columns = ["date", column]
    s = s.sort_values("date")
    if s_dt is None or e_dt is None:
        s = s.tail(days)
    return s

from src.data.ak_client import AKDataClient

@st.cache_resource
def get_client() -> AKDataClient:
    return AKDataClient()

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

def format_cn_amount(v: float) -> str:
    """金额中文单位格式化，>1亿显示"亿"，>1万显示"万"。"""
    try:
        x = float(v or 0)
    except Exception:
        return "-"
    if abs(x) >= 1e8:
        return f"{x/1e8:,.2f}亿"
    if abs(x) >= 1e4:
        return f"{x/1e4:,.2f}万"
    return f"{x:,.0f}"

def format_cn_volume(v: float) -> str:
    """成交量单位保持股，>1亿股显示"亿股"、>1万股显示"万股"。"""
    try:
        x = float(v or 0)
    except Exception:
        return "-"
    if abs(x) >= 1e8:
        return f"{x/1e8:,.2f}亿股"
    if abs(x) >= 1e4:
        return f"{x/1e4:,.2f}万股"
    return f"{x:,.0f}股"

def format_messages_for_download(messages: List[Dict[str, Any]]) -> str:
    """将 messages 列表格式化为可读文本，便于下载保存。
    输出形如：
    #1 [system]\n...\n\n#2 [system]\n...\n\n#3 [user]\n...
    """
    parts: List[str] = []
    for i, m in enumerate(messages or [], 1):
        role = str(m.get("role", ""))
        content = m.get("content")
        try:
            content_str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
        except Exception:
            content_str = str(content)
        parts.append(f"#{i} [{role}]\n{content_str}")
    return "\n\n".join(parts)


def chat_with_tools(router: LLMRouter, messages, tools_schema=None, max_rounds: int = 3):
    """
    通用自动工具执行循环
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
        # 无法解析，直接返回原始结果
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
            # 没有工具调用，认为已给出最终回复
            return {"final_text": content, "raw": rsp}

    # 达到回合上限，返回最后一次原始结果
    return {"final_text": content if content else "", "raw": last_raw}


# 缓存获取股票名称（A/港股）
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

@st.cache_data(ttl=3600)
def get_a_stock_list_cached() -> pd.DataFrame:
    """
    优先从 akshare 获取 A 股代码/名称；若接口异常或返回空，则回退到本地缓存目录 data/cache/A 中已存在的标的目录作为兜底。
    """
    try:
        df = pd.DataFrame()
        # 1) 优先尝试多个 akshare 接口（不同版本字段/函数名不一致）
        for fn_name, kwargs in [
            ("stock_zh_a_spot_em", {}),
            ("stock_info_a_code_name", {}),
            ("stock_zh_a_spot", {}),
            ("stock_zh_a_spot_deal", {}),
        ]:
            try:
                fn = getattr(ak, fn_name, None)
                if fn is None:
                    continue
                tmp = fn(**kwargs)
                if tmp is not None and not tmp.empty:
                    df = tmp
                    break
            except Exception:
                continue
        # 2) 若仍为空，尝试从本地缓存兜底
        if df is None or df.empty:
            try:
                a_dir = ROOT / "data" / "cache" / "A"
                codes: list[str] = []
                if a_dir.exists():
                    for p in a_dir.iterdir():
                        if p.is_dir():
                            s = "".join(ch for ch in p.name if ch.isdigit())
                            if len(s) >= 6:
                                codes.append(s[-6:])
                codes = sorted(set(codes))
                if codes:
                    out = pd.DataFrame({"代码": [c for c in codes], "名称": [""] * len(codes)})
                    return ensure_arrow_compatible(out)
            except Exception:
                pass
            return pd.DataFrame(columns=["代码","名称"])  
        # 3) 规范列名并标准化代码
        code_col = None
        name_col = None
        for c in ["代码", "symbol", "code", "证券代码", "股票代码"]:
            if c in df.columns:
                code_col = c
                break
        for c in ["名称", "name", "证券名称", "股票简称", "简称"]:
            if c in df.columns:
                name_col = c
                break
        if code_col is None:
            return pd.DataFrame(columns=["代码","名称"])  
        out = pd.DataFrame()
        out["代码"] = (
            df[code_col]
            .astype(str)
            .str.upper()
            .str.replace(".SH", "", regex=False)
            .str.replace(".SZ", "", regex=False)
            .str.replace("SH", "", regex=False)
            .str.replace("SZ", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.zfill(6)
        )
        out["名称"] = df[name_col].astype(str) if name_col else ""
        return ensure_arrow_compatible(out)
    except Exception:
        # 兜底：硬返回空结构，避免 UI 崩溃
        return pd.DataFrame(columns=["代码","名称"])  

@st.cache_data(ttl=3600)
def get_hk_ggt_list_cached() -> pd.DataFrame:
    """
    优先从 akshare 获取港股通成分；若失败则回退到本地缓存目录 data/cache/H 中已有的标的目录。
    修复：部分环境中 ak 接口默认只返回单个方向，需分别拉取“港股通（沪）/（深）”，并统一标准化为 5 位代码。
    """
    try:
        # 1) akshare 官方港股通列表：分别尝试“港股通（沪）/（深）”，兼容半角括号
        df_list: list[pd.DataFrame] = []
        dbg: list[Dict[str, Any]] = []
        for fn_name, kwargs in [
            ("stock_hk_ggt_components_em", {"symbol": "港股通（沪）"}),
            ("stock_hk_ggt_components_em", {"symbol": "港股通（深）"}),
            ("stock_hk_ggt_components_em", {"symbol": "港股通(沪)"}),
            ("stock_hk_ggt_components_em", {"symbol": "港股通(深)"}),
            # 兼容旧版：无参调用可能直接返回全部或默认一个方向
            ("stock_hk_ggt_components_em", {}),
        ]:
            try:
                fn = getattr(ak, fn_name, None)
                if fn is None:
                    continue
                tmp = fn(**kwargs)
                if tmp is not None and not tmp.empty:
                    df_list.append(tmp)
                    try:
                        dbg.append({"source": f"{fn_name}({kwargs})", "len": int(len(tmp)), "columns": list(tmp.columns)})
                    except Exception:
                        pass
                else:
                    try:
                        dbg.append({"source": f"{fn_name}({kwargs})", "len": 0, "columns": list(tmp.columns) if tmp is not None else []})
                    except Exception:
                        pass
            except Exception as e:
                try:
                    dbg.append({"source": f"{fn_name}({kwargs})", "error": str(e)})
                except Exception:
                    pass
                continue
        df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

        # 2) 若仍为空，回退到本地缓存目录，但不早退：继续尝试“全量行情”兜底合并
        if df is None or df.empty:
            out = pd.DataFrame(columns=["代码","名称"])  # 初始为空
            try:
                h_dir = ROOT / "data" / "cache" / "H"
                codes: list[str] = []
                if h_dir.exists():
                    for p in h_dir.iterdir():
                        if p.is_dir():
                            s = "".join(ch for ch in p.name if ch.isdigit())
                            if s:
                                if len(s) >= 5:
                                    s = s[-5:]
                                codes.append(s.zfill(5))
                codes = sorted(set(codes))
                if codes:
                    out = pd.DataFrame({"代码": codes, "名称": [""] * len(codes)})
            except Exception:
                pass
            # 记录调试信息（即使只有本地缓存）
            try:
                st.session_state["_dbg_hk_ggt_sources"] = (dbg or []) + [{"source": "local_cache_H", "len": int(len(out))}]
                st.session_state["_dbg_hk_ggt_final"] = {"len": int(len(out)), "columns": list(out.columns)}
            except Exception:
                pass
            # 若仍偏少，继续尝试用“港股全量行情”列表合并兜底
            try:
                if len(out) < 50:
                    for spot_name in [
                        "stock_hk_main_board_spot_em",
                        "stock_hk_spot_em",
                        "stock_hk_spot",
                    ]:
                        fn_spot = getattr(ak, spot_name, None)
                        if fn_spot is None:
                            continue
                        try:
                            hk = fn_spot()
                            if hk is None or hk.empty:
                                continue
                            code_col2 = next((c for c in ["代码","symbol","code","证券代码"] if c in hk.columns), None)
                            name_col2 = next((c for c in ["名称","name","证券名称","股票简称","简称"] if c in hk.columns), None)
                            if code_col2 is None:
                                continue
                            tmp = pd.DataFrame()
                            tmp["代码"] = (
                                hk[code_col2]
                                .astype(str)
                                .str.upper()
                                .str.replace(".HK", "", regex=False)
                                .str.extract(r"(\d+)")[0]
                                .fillna("")
                                .apply(lambda s: s[-5:].zfill(5) if s else "")
                            )
                            tmp["名称"] = hk[name_col2].astype(str) if name_col2 else ""
                            tmp = tmp[tmp["代码"] != ""].drop_duplicates(subset=["代码"])
                            if not tmp.empty:
                                codes_merged = sorted(set(list(out["代码"])) | set(list(tmp["代码"]))) if not out.empty else sorted(set(list(tmp["代码"])))
                                out = pd.DataFrame({"代码": codes_merged, "名称": [""] * len(codes_merged)})
                                try:
                                    dbg.append({"source": f"fallback: {spot_name}", "len": int(len(tmp)), "columns": list(hk.columns)})
                                    st.session_state["_dbg_hk_ggt_sources"] = dbg
                                    st.session_state["_dbg_hk_ggt_final"] = {"len": int(len(out)), "columns": list(out.columns), "fallback_spot": spot_name}
                                except Exception:
                                    pass
                                break
                        except Exception as e:
                            try:
                                dbg.append({"source": f"{spot_name}()", "error": str(e)})
                                st.session_state["_dbg_hk_ggt_sources"] = dbg
                            except Exception:
                                pass
                            continue
            except Exception:
                pass
            return ensure_arrow_compatible(out)

        # 3) 规范列名并标准化代码
        code_col = None
        name_col = None
        for c in ["代码", "symbol", "code", "证券代码", "股票代码"]:
            if c in df.columns:
                code_col = c
                break
        for c in ["名称", "name", "证券名称", "股票简称", "简称"]:
            if c in df.columns:
                name_col = c
                break
        if code_col is None:
            return pd.DataFrame(columns=["代码","名称"])  

        out = pd.DataFrame()
        # 提取数字并标准化为 5 位港股代码
        codes_series = df[code_col].astype(str).str.extract(r"(\d+)")[0].fillna("")
        out["代码"] = codes_series.apply(lambda s: s[-5:].zfill(5) if s else "")
        out["名称"] = df[name_col].astype(str) if name_col else ""

        # 去重与排序
        out = out[out["代码"] != ""].drop_duplicates(subset=["代码"]).sort_values("代码").reset_index(drop=True)

        # 记录调试信息
        try:
            st.session_state["_dbg_hk_ggt_sources"] = dbg
            st.session_state["_dbg_hk_ggt_final"] = {"len": int(len(out)), "columns": list(out.columns), "head": out.head(10).to_dict(orient="records")}
        except Exception:
            pass

        # 若数量异常偏少，尝试与本地缓存目录合并兜底
        try:
            if len(out) < 50:
                h_dir = ROOT / "data" / "cache" / "H"
                codes_local: list[str] = []
                if h_dir.exists():
                    for p in h_dir.iterdir():
                        if p.is_dir():
                            s = "".join(ch for ch in p.name if ch.isdigit())
                            if s:
                                if len(s) >= 5:
                                    s = s[-5:]
                                codes_local.append(s.zfill(5))
                if codes_local:
                    codes_merged = sorted(set(list(out["代码"]) + codes_local))
                    out = pd.DataFrame({"代码": codes_merged, "名称": [""] * len(codes_merged)})
                    try:
                        st.session_state["_dbg_hk_ggt_final"] = {"len": int(len(out)), "columns": list(out.columns), "merged_local": True}
                    except Exception:
                        pass
        except Exception:
            pass

        # 4) 数量仍偏少：临时大兜底——合并港股全量行情列表
        try:
            if len(out) < 50:
                for spot_name in [
                    "stock_hk_main_board_spot_em",
                    "stock_hk_spot_em",
                    "stock_hk_spot",
                ]:
                    fn_spot = getattr(ak, spot_name, None)
                    if fn_spot is None:
                        continue
                    try:
                        hk = fn_spot()
                        if hk is None or hk.empty:
                            continue
                        code_col2 = next((c for c in ["代码","symbol","code","证券代码"] if c in hk.columns), None)
                        name_col2 = next((c for c in ["名称","name","证券名称","股票简称","简称"] if c in hk.columns), None)
                        if code_col2 is None:
                            continue
                        tmp = pd.DataFrame()
                        tmp["代码"] = (
                            hk[code_col2]
                            .astype(str)
                            .str.upper()
                            .str.replace(".HK", "", regex=False)
                            .str.extract(r"(\d+)")[0]
                            .fillna("")
                            .apply(lambda s: s[-5:].zfill(5) if s else "")
                        )
                        tmp["名称"] = hk[name_col2].astype(str) if name_col2 else ""
                        tmp = tmp[tmp["代码"] != ""].drop_duplicates(subset=["代码"])
                        if not tmp.empty:
                            codes_merged = sorted(set(list(out["代码"])) | set(list(tmp["代码"])) ) if not out.empty else sorted(set(list(tmp["代码"])) )
                            out = pd.DataFrame({"代码": codes_merged, "名称": [""] * len(codes_merged)})
                            try:
                                dbg.append({"source": f"fallback: {spot_name}", "len": int(len(tmp)), "columns": list(hk.columns)})
                                st.session_state["_dbg_hk_ggt_sources"] = dbg
                                st.session_state["_dbg_hk_ggt_final"] = {"len": int(len(out)), "columns": list(out.columns), "fallback_spot": spot_name}
                            except Exception:
                                pass
                            break
                    except Exception as e:
                        try:
                            dbg.append({"source": f"{spot_name}()", "error": str(e)})
                        except Exception:
                            pass
                        continue
        except Exception:
            pass

        return ensure_arrow_compatible(out)
    except Exception:
        return pd.DataFrame(columns=["代码","名称"])  
    except Exception:
        return pd.DataFrame(columns=["代码","名称"])  

# -------- 新增：导航与自选股持久化工具 --------
WATCHLIST_PATH = ROOT / "data" / "watchlist.json"
# 新增：数据更新历史日志文件（JSONL，每行一条记录）
UPDATE_HISTORY_PATH = ROOT / "data" / "update_history.jsonl"

# 追加写入一条更新历史记录
def _append_update_history(entry: Dict[str, Any]):
    try:
        UPDATE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = dict(entry)
        rec["ts"] = datetime.now().isoformat()
        with open(UPDATE_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # 忽略写入异常，不影响主流程
        pass

@st.cache_data(ttl=60)
def load_update_history_cached(ver: int = 0) -> pd.DataFrame:
    """读取历史记录（带缓存）。ver 作为缓存破坏参数用于刷新。返回按时间倒序的 DataFrame。"""
    try:
        if not UPDATE_HISTORY_PATH.exists():
            return pd.DataFrame()
        rows: list[dict] = []
        with open(UPDATE_HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        df = pd.DataFrame(rows)
        if not df.empty:
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            df = df.sort_values("ts", ascending=False)
        return df
    except Exception:
        return pd.DataFrame()

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

# -------- 详情页所复用的单股页面（从 main）提取为函数 --------
def single_stock_page():
    st.header("单只股票查询")
    # 将 sidebar 中的控件迁移为当前分区内控件
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        market = st.selectbox("市场", ["A", "H"], index=st.session_state.get("_ss_market_idx", 0), key="ss_market")
        st.session_state["_ss_market_idx"] = 0 if market == "A" else 1
        symbol = st.text_input("股票代码", value=st.session_state.get("detail_symbol", "600519" if market == "A" else "00700"))
    with c2:
        period = st.selectbox("周期", ["daily", "weekly", "monthly", "quarterly", "yearly"], index=0, key="ss_period")
        start = st.text_input("开始日期(YYYYMMDD)", value="20180101")
        end = st.text_input("结束(YYYYMMDD)", value="20251231")
    with c3:
        adjust = st.selectbox("复权(A)", [None, "qfq", "hfq"], index=0, key="ss_adjust") if market == "A" else None
        use_cache = st.checkbox("使用缓存", value=True)
        refresh = st.checkbox("强制刷新", value=False)
        expire_days = st.number_input("过期天数", 0, 365, 3, 1)

    # 数据获取前：指标显示控制（放到 K 线图旁边的布局中使用）
    # 先临时保存配置，稍后绘图区域再布局到 K 线图旁边
    show_ma_default = st.session_state.get("_show_ma", True)
    ma_list_default = st.session_state.get("_ma_list_str", "5,10,20,60")
    show_macd_default = st.session_state.get("_show_macd", False)
    second_rows_default = st.session_state.get("_second_rows", ["成交量"])  # 次级子图最多两行

    # 懒加载：仅在点击“开始查询”或勾选“自动查询”时才获取数据
    col_ctrl1, col_ctrl2 = st.columns([1,1])
    with col_ctrl1:
        btn_query = st.button("开始查询", key="_single_do_query")
    with col_ctrl2:
        auto_query = st.checkbox("自动查询", value=False, key="_single_auto_query", help="默认不在首次渲染时发起网络请求")
    _do_query = (str(symbol or "").strip() != "") and (btn_query or auto_query)

    if not _do_query:
        st.info("为提升首屏速度，默认不自动查询。请设置参数后点击“开始查询”或勾选“自动查询”。")
        return

    # 数据获取
    client = get_client()
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

    # 已按需求：隐藏历史详情数据模块，改为弹窗方式展示
    @st.dialog("历史详情数据")
    def _show_history_dialog():
        if df_disp is not None and not df_disp.empty:
            st.dataframe(
                ensure_arrow_compatible(
                    df_disp.sort_values("date", ascending=False).rename(columns=cn_map).head(500)
                )
            )
        else:
            st.info("无数据，请检查代码、周期或日期范围。")

    # 隐藏历史数据概览直接展示，改用弹窗触发
    cols_hist = st.columns([1, 1, 6])
    with cols_hist[0]:
        if st.button("跳转到详情页", key="go_detail_single"):
            go_detail(market, symbol)
    with cols_hist[1]:
        if st.button("查看历史详情数据", key="btn_show_hist_dialog_top"):
            _show_history_dialog()

    # 先绘制 K 线图，并把“查看历史详情数据”按钮放在旁边（顶部已有同名按钮，保留两个入口）
    st.subheader("K线图")
    fig_title = f"{stock_name}({symbol}) K线" if stock_name else f"{symbol} K线"

    # 左右并列：左侧图，右侧控制
    lc, rc = st.columns([5, 2])
    with rc:
        show_ma = st.checkbox("显示 MA 均线", value=show_ma_default, key="_show_ma")
        ma_list_str = st.text_input("MA窗口(逗号分隔)", value=ma_list_default, key="_ma_list_str")
        ma_windows = [int(x) for x in ma_list_str.split(',') if x.strip().isdigit()] if show_ma else []
        # 次级子图选择（最多两行）
        second_opts = ["成交量", "MACD", "RSI"]
        second_rows = st.multiselect("次级子图（最多选2项）", options=second_opts, default=second_rows_default, max_selections=2, key="_second_rows")
        show_macd = ("MACD" in second_rows)
    with lc:
        try:
            import src.viz.charts as charts_mod
            charts_mod = importlib.reload(charts_mod)
            fig = charts_mod.kline_with_volume(df_disp, title=fig_title, ma_windows=ma_windows, show_macd=show_macd, period=period, second_rows=second_rows)
        except Exception:
            fig = kline_with_volume(df_disp, title=fig_title, ma_windows=ma_windows, show_macd=show_macd, period=period, second_rows=second_rows)
        st.plotly_chart(fig, use_container_width=True)

    # 新增：成交量统计（近N日、去年同比、环比）
    with st.container():
        st.subheader("成交量统计")
        cols_n = st.columns([1.2,1.2,1.6,1.6,1.6,1.6])
        with cols_n[0]:
            mode = st.selectbox("统计模式", options=["近N日", "时间周期"], index=0, key="_single_stat_mode")
            N = st.number_input("N(日)", min_value=5, max_value=250, value=20, step=5, key="_vol_N_single") if mode == "近N日" else None
        with cols_n[1]:
            if mode == "时间周期":
                default_end = pd.to_datetime("today").normalize()
                default_start = default_end - pd.Timedelta(days=30)
                s = st.date_input("开始", value=default_start.date(), key="_single_period_start")
                e = st.date_input("结束", value=default_end.date(), key="_single_period_end")
            else:
                s = e = None
        # 计算并展示
        try:
            if mode == "近N日":
                metrics = compute_symbol_volume_metrics(market, symbol, int(N))
            else:
                metrics = compute_symbol_volume_metrics_period(market, symbol, s, e)
            curr = metrics.get("curr") or 0.0
            yoy = metrics.get("yoy")
            prev = metrics.get("prev")
            yoy_pct = metrics.get("yoy_pct")
            mom_pct = metrics.get("mom_pct")
            with cols_n[2]:
                title_v_curr = "近N日成交量(股)" if mode == "近N日" else "周期内成交量(股)"
                st.metric(title_v_curr, f"{curr:,.0f}")
            with cols_n[3]:
                title_v_y = "去年同期N日(股)" if mode == "近N日" else "去年同期(股)"
                st.metric(title_v_y, "-" if yoy is None else f"{yoy:,.0f}", delta=None)
            with cols_n[4]:
                st.metric("同比", "-" if yoy_pct is None else f"{yoy_pct:.2%}")
            with cols_n[5]:
                st.metric("环比", "-" if mom_pct is None else f"{mom_pct:.2%}")
        except Exception as e:
            st.info(f"成交量统计暂不可用：{e}")

    # 移到 K 线图下方的“指标与回测”
    st.subheader("指标与回测")
    if df_disp is not None and not df_disp.empty:
        close = df_disp["close"].astype(float)
        df_ind = df_disp.copy()
        # 策略选择
        st.write("")
        col_sel1, col_sel2, col_sel3 = st.columns([1.2, 1, 1])
        with col_sel1:
            strat = st.selectbox("策略", ["MA 金叉", "MACD 金叉", "RSI 区间"], index=0, key="_bt_strat")
        with col_sel2:
            ma_params = st.text_input("MA参数(短,长)", value="20,60", key="_bt_ma_params")
        with col_sel3:
            rsi_params = st.text_input("RSI参数(周期,低位,高位)", value="14,30,70", key="_bt_rsi_params")

        # 解析参数
        short, long = 20, 60
        try:
            parts = [int(x) for x in ma_params.split(',') if x.strip().isdigit()]
            if len(parts) >= 2:
                short, long = parts[0], parts[1]
        except Exception:
            pass
        rsi_period, rsi_low, rsi_high = 14, 30, 70
        try:
            ps = [int(x) for x in rsi_params.split(',') if x.strip().isdigit()]
            if len(ps) >= 3:
                rsi_period, rsi_low, rsi_high = ps[0], ps[1], ps[2]
        except Exception:
            pass

        # 计算指标以便复用（不强制显示）
        df_ind["SMA20"] = sma(close, 20)
        df_ind["SMA60"] = sma(close, 60)

        # 回测执行
        if strat == "MA 金叉":
            from src.logic.indicators import backtest_ma_cross
            res_bt = backtest_ma_cross(df_ind, short=short, long=long)
            strat_name = f"MA({short}/{long}) 趋势策略"
        elif strat == "MACD 金叉":
            from src.logic.indicators import backtest_macd_cross
            res_bt = backtest_macd_cross(df_ind)
            strat_name = "MACD 金叉/死叉"
        else:
            from src.logic.indicators import backtest_rsi
            res_bt = backtest_rsi(df_ind, period=rsi_period, low=rsi_low, high=rsi_high)
            strat_name = f"RSI 区间({rsi_period},{rsi_low},{rsi_high})"

        # 展示回测信息（策略、周期、区间与核心指标）
        trades = int((res_bt or {}).get("trades", 0))
        ret = float((res_bt or {}).get("return", 0.0))
        mdd = float((res_bt or {}).get("max_drawdown", 0.0))
        win = float((res_bt or {}).get("win_rate", 0.0))
        period_txt = {"daily":"日线","weekly":"周线","monthly":"月线","quarterly":"季线","yearly":"年线"}.get(period, str(period))
        try:
            start_date = pd.to_datetime(df_disp["date"].min()).date() if "date" in df_disp.columns else None
            end_date = pd.to_datetime(df_disp["date"].max()).date() if "date" in df_disp.columns else None
        except Exception:
            start_date, end_date = None, None
        st.caption(f"策略：{strat_name} | 周期：{period_txt}" + (f" | 区间：{start_date} ~ {end_date}" if start_date and end_date else ""))
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("交易笔数", f"{trades}")
        with c2: st.metric("累计收益", f"{ret:.2%}")
        with c3: st.metric("最大回撤", f"{mdd:.2%}")
        with c4: st.metric("胜率", f"{win:.2%}")
        if trades == 0:
            st.info("无交易产生，可能因为区间内无信号或区间过短。")

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
                ma_txt = "MA(20/60) 趋势策略：当短均线上穿长均线买入、下穿卖出；适合趋势行情，震荡时容易出现虚假信号。"
                summary = (
                    f"回测基于{period_txt}与所选展示区间。共 {trades} 笔交易，区间累计收益约 {ret:.2%}，最大回撤约 {mdd:.2%}，胜率约 {win:.2%}。\n"
                    f"直观解读：收益较{'可观' if ret>0 else '一般'}，回撤{'偏高' if mdd>0.25 else '可控'}，胜率{'略低于' if win<0.5 else '高于'} 50%。策略更依赖趋势段与盈亏比，需控制仓位与止损。\n"
                    f"策略说明：{ma_txt}\n"
                    f"风控建议：可叠加成交量/RSI/布林带过滤、设置固定或 ATR 止损、使用多周期共振（周线定方向，{period_txt}择时）。以上仅为方法参考，非投资建议。"
                )
                st.success("中文总结已生成：")
                st.write(summary)
            except Exception as e:
                st.warning(f"生成中文总结失败：{e}")

    # 删除原先的“个股/行业信息（A/港股）”模块
    # === 新增：个股详情四项信息 ===
    st.subheader("个股详情")
    tab_base, tab_fin, tab_ind, tab_risk = st.tabs(["基本面信息", "最新财报信息", "所属板块信息", "未来三个月风险提示"])

        # 1) 基本面信息
    with tab_base:
            try:
                if market == "A":
                    base_df = ak.stock_individual_info_em(symbol=symbol)
                    if base_df is not None and not base_df.empty:
                        st.dataframe(ensure_arrow_compatible(base_df))
                    else:
                        st.info("未获取到 A 股基本面信息。")
                else:
                    # 尝试展示港股基本信息：优先实时报价中的基本字段
                    try:
                        hk_spot = ak.stock_hk_spot_em()
                        code_col = None
                        for c in ["代码", "symbol", "code", "证券代码"]:
                            if c in hk_spot.columns:
                                code_col = c
                                break
                        if code_col:
                            sym = str(symbol).upper().replace(".HK", "")
                            hk_spot["_code_norm"] = hk_spot[code_col].astype(str).str.upper().str.replace(".HK", "", regex=False).str.lstrip("0")
                            row = hk_spot[hk_spot["_code_norm"] == sym.lstrip("0")].drop(columns=["_code_norm"])  # type: ignore
                            if not row.empty:
                                st.dataframe(ensure_arrow_compatible(row))
                            else:
                                st.info("未在港股实时报价中找到该代码。")
                        else:
                            st.info("港股基本信息暂未适配该数据结构。")
                    except Exception as e:
                        st.info(f"无法获取港股基本信息：{e}")
            except Exception as e:
                st.warning(f"获取基本面信息失败：{e}")

        # 2) 最新财报信息
    with tab_fin:
            ok = False
            fin_df_used = None
            if market == "A":
                # 依次尝试多种接口，兼容不同 akshare 版本
                for fn in [
                    getattr(ak, "stock_financial_analysis_indicator", None),
                    getattr(ak, "stock_financial_report_sina", None),
                    getattr(ak, "stock_financial_abstract_ths", None),
                ]:
                    try:
                        if fn is None:
                            continue
                        fin_df = fn(symbol=symbol)  # type: ignore
                        if fin_df is not None and not fin_df.empty:
                            # 新增：按报告期倒排
                            try:
                                cand_cols = [
                                    "报告期", "报表日期", "公告日期", "日期", "报告日期", "endDate", "REPORT_DATE", "REPORTDATE", "截止日期", "period"
                                ]
                                col = next((c for c in cand_cols if c in fin_df.columns), None)
                                if col:
                                    _tmp = fin_df.copy()
                                    _tmp["__dt__"] = pd.to_datetime(_tmp[col].astype(str).str.replace("年", "-", regex=False).str.replace("月", "-", regex=False).str.replace("日", "", regex=False), errors="coerce")
                                    _tmp = _tmp.sort_values("__dt__", ascending=False, na_position="last").drop(columns=["__dt__"])
                                    fin_df_used = _tmp
                                else:
                                    fin_df_used = fin_df
                            except Exception:
                                fin_df_used = fin_df
                            st.dataframe(ensure_arrow_compatible(fin_df_used))
                            ok = True
                            break
                    except Exception:
                        continue
                if not ok:
                    st.info("未获取到 A 股最新财报信息（可能接口变动或版本不兼容）。")
            else:
                # 港股：尝试常见接口，若失败则提示
                tried = False
                for name in [
                    "stock_hk_financial_analysis_indicator_em",
                    "stock_hk_finance_analysis_em",
                ]:
                    fn = getattr(ak, name, None)
                    if fn is None:
                        continue
                    tried = True
                    try:
                        fin_df = fn(symbol=symbol)  # type: ignore
                        if fin_df is not None and not fin_df.empty:
                            # 新增：按报告期倒排
                            try:
                                cand_cols = [
                                    "报告期", "报表日期", "公告日期", "日期", "报告日期", "endDate", "REPORT_DATE", "REPORTDATE", "截止日期", "period"
                                ]
                                col = next((c for c in cand_cols if c in fin_df.columns), None)
                                if col:
                                    _tmp = fin_df.copy()
                                    _tmp["__dt__"] = pd.to_datetime(_tmp[col].astype(str).str.replace("年", "-", regex=False).str.replace("月", "-", regex=False).str.replace("日", "", regex=False), errors="coerce")
                                    _tmp = _tmp.sort_values("__dt__", ascending=False, na_position="last").drop(columns=["__dt__"])
                                    fin_df_used = _tmp
                                else:
                                    fin_df_used = fin_df
                            except Exception:
                                fin_df_used = fin_df
                            st.dataframe(ensure_arrow_compatible(fin_df_used))
                            ok = True
                            break
                    except Exception:
                        continue
                if not ok:
                    if tried:
                        st.info("未获取到港股最新财报信息（接口返回为空或筛选不到该代码）。")
                    else:
                        st.info("当前 akshare 版本暂未提供港股财报适配接口。")

            # 新增（3） 在数据表格后追加财报文字总结（由大模型生成）
            if ok and fin_df_used is not None and not fin_df_used.empty:
                with st.expander("生成财报文字总结", expanded=False):
                    topn = st.slider("纳入总结的最近期数量", min_value=1, max_value=min(8, len(fin_df_used)), value=min(4, len(fin_df_used)))
                    df_for_sum = fin_df_used.head(topn)
                    st.dataframe(ensure_arrow_compatible(df_for_sum))
                    gen_fin_sum = st.button("生成总结", key=f"btn_fin_sum_{market}_{symbol}")
                    if gen_fin_sum:
                        try:
                            # 将表格压缩为 JSON 文本供模型理解
                            jtxt = df_for_sum.to_json(orient="records", force_ascii=False)
                            sys_prompt = {"role":"system","content":"你是资深卖方分析师。请基于最近几个财报期的关键指标，给出中文要点总结：收入与利润同比、毛利率与净利率趋势、费用趋势、现金流与资产负债变化、分红与指引（如有）、核心风险与看点。避免夸大，不构成投资建议。"}
                            messages = [sys_prompt, {"role":"system","content":f"股票: {stock_name or ''}({symbol}) | 市场: {market}"}, {"role":"user","content":"以下是最近期财报数据（JSON）：\n" + jtxt}]
                            route_name = st.session_state.get("route_name", "default")
                            enable_tools = st.session_state.get("enable_tools", True)
                            registry = ProviderRegistry(public_cfg_path="models.yaml", local_cfg_path="models.local.yaml")
                            router = LLMRouter(registry=registry, route_name=route_name)
                            tools = get_tools_schema() if enable_tools else None
                            # 下载Prompt（本次请求的完整messages）
                            try:
                                _fname = f"fin_{market}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            except Exception:
                                _fname = f"fin_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            _prompt_text = format_messages_for_download(messages)
                            st.download_button(
                                label="下载Prompt",
                                data=_prompt_text.encode('utf-8'),
                                file_name=_fname,
                                mime="text/plain",
                                key=f"dl_prompt_fin_{market}_{symbol}"
                            )

                            res = chat_with_tools(router, messages, tools_schema=tools, max_rounds=2)
                            txt = (res or {}).get("final_text") or ""
                            if txt.strip():
                                st.markdown(txt)
                            else:
                                _render_llm_answer(res)
                        except Exception as e:
                            st.warning(f"生成财报总结失败：{e}")

        # 3) 所属行业信息
    with tab_ind:
            try:
                ind_name = None
                if market == "A":
                    try:
                        info_df = ak.stock_individual_info_em(symbol=symbol)
                        if info_df is not None and not info_df.empty:
                            col0, col1 = info_df.columns[:2]
                            for key in ["所属行业", "行业", "行业分类", "细分行业"]:
                                _row = info_df[info_df[col0].astype(str).str.contains(key, na=False)]
                                if not _row.empty:
                                    ind_name = str(_row[col1].iloc[0])
                                    break
                    except Exception:
                        pass
                else:
                    # 简要行业信息：港股尝试从常见列中读取
                    try:
                        hk_spot = ak.stock_hk_spot_em()
                        code_col = None
                        for c in ["代码", "symbol", "code", "证券代码"]:
                            if c in hk_spot.columns:
                                code_col = c
                                break
                        if code_col:
                            sym = str(symbol).upper().replace(".HK", "")
                            hk_spot["_code_norm"] = hk_spot[code_col].astype(str).str.upper().str.replace(".HK", "", regex=False).str.lstrip("0")
                            row = hk_spot[hk_spot["_code_norm"] == sym.lstrip("0")]
                            # 在行情表里尝试常见行业列
                            for candi in ["行业", "所属行业", "板块", "行业分类"]:
                                if candi in row.columns and not row.empty:
                                    val = str(row.iloc[0][candi])
                                    if val and val != "nan":
                                        ind_name = val
                                        break
                    except Exception:
                        pass
                if ind_name:
                    st.success(f"所属板块：{ind_name}")
                    # 新增（2） 调用大模型生成行业上中下游与龙头总结
                    with st.expander("生成板块产业链与龙头总结", expanded=False):
                        extra = st.text_input("可选：补充关键子板块/区域（提高针对性）", value="")
                        gen_ind = st.button("生成板块总结", key=f"btn_ind_summary_{market}_{symbol}")
                        if gen_ind:
                            try:
                                ctx_lines = [
                                    f"市场: {market}",
                                    f"代码: {symbol}",
                                    f"名称: {stock_name or ''}",
                                    f"所属板块: {ind_name}",
                                ]
                                if extra:
                                    ctx_lines.append(f"补充: {extra}")
                                sys_prompt = {"role":"system","content":"你是资深板块分析师。请围绕所给板块，概述上游-中游-下游的关键环节、各环节A/H常见龙头公司（如能给出）、驱动因素、景气度指标与风险点。中文分点输出，条理清晰，不构成投资建议。"}
                                messages = [sys_prompt, {"role":"system","content":"上下文：\n" + "\n".join(ctx_lines)}]
                                route_name = st.session_state.get("route_name", "default")
                                enable_tools = st.session_state.get("enable_tools", True)
                                registry = ProviderRegistry(public_cfg_path="models.yaml", local_cfg_path="models.local.yaml")
                                router = LLMRouter(registry=registry, route_name=route_name)
                                tools = get_tools_schema() if enable_tools else None
                                # 下载Prompt（本次请求的完整messages）
                                try:
                                    _fname = f"ind_{market}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                except Exception:
                                    _fname = f"ind_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                _prompt_text = format_messages_for_download(messages)
                                st.download_button(
                                    label="下载Prompt",
                                    data=_prompt_text.encode('utf-8'),
                                    file_name=_fname,
                                    mime="text/plain",
                                    key=f"dl_prompt_ind_{market}_{symbol}"
                                )
                                res = chat_with_tools(router, messages, tools_schema=tools, max_rounds=2)
                                txt = (res or {}).get("final_text") or ""
                                if txt.strip():
                                    st.markdown(txt)
                                else:
                                    _render_llm_answer(res)
                            except Exception as e:
                                st.warning(f"生成板块总结失败：{e}")
                else:
                    st.info("暂未识别到所属板块信息。")
            except Exception as e:
                st.warning(f"板块信息处理失败：{e}")

        # 4) 未来三个月内可能存在的风险提示（调用大模型生成）
    with tab_risk:
            try:
                # 构造上下文（价格走势、回测概况、行业信息、基础面摘要）
                ctx_lines = [
                    f"市场: {market}",
                    f"代码: {symbol}",
                    f"名称: {stock_name or ''}",
                    f"周期: {period}",
                ]
                # 行业
                try:
                    if 'ind_name' in locals() and ind_name:
                        ctx_lines.append(f"板块: {ind_name}")
                except Exception:
                    pass
                # 简要走势统计（近60交易日）
                try:
                    last_n = df_disp.tail(60).copy()
                    last_n["ret"] = last_n["close"].pct_change()
                    ret_1m = (1 + last_n.tail(22)["ret"].fillna(0)).prod() - 1
                    ret_3m = (1 + last_n["ret"].fillna(0)).prod() - 1
                    vol_mean = float(last_n["volume"].tail(22).mean()) if "volume" in last_n.columns else 0
                    ctx_lines.append(f"近1月涨跌幅: {ret_1m:.2%}")
                    ctx_lines.append(f"近3月涨跌幅: {ret_3m:.2%}")
                    ctx_lines.append(f"近1月平均成交量: {vol_mean:.0f}")
                except Exception:
                    pass

                sys_prompt = {"role":"system","content":"你是资深卖方分析师和风控专家。请基于提供的上下文，在不臆测、不过度承诺的前提下，给出未来三个月内的主要风险点与监控要点，中文输出，列出 5-8 条要点，每条尽量具体且可操作。"}
                messages = [sys_prompt, {"role":"system","content":"以下是上下文：\n" + "\n".join(ctx_lines)}]
                # 允许用户补充提问或说明
                user_tip = st.text_area("可选：补充你关注的风险点（将影响生成结果）", value="")
                if user_tip:
                    messages.append({"role":"user","content": user_tip})

                route_name = st.session_state.get("route_name", "default")
                enable_tools = st.session_state.get("enable_tools", True)
                registry = ProviderRegistry(public_cfg_path="models.yaml", local_cfg_path="models.local.yaml")
                router = LLMRouter(registry=registry, route_name=route_name)
                tools = get_tools_schema() if enable_tools else None
                # 下载Prompt（本次请求的完整messages）
                try:
                    _fname = f"risk_{market}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                except Exception:
                    _fname = f"risk_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                _prompt_text = format_messages_for_download(messages)
                st.download_button(
                    label="下载Prompt",
                    data=_prompt_text.encode('utf-8'),
                    file_name=_fname,
                    mime="text/plain",
                    key=f"dl_prompt_risk_{market}_{symbol}"
                )
                res = chat_with_tools(router, messages, tools_schema=tools, max_rounds=2)
                txt = (res or {}).get("final_text") or ""
                if txt.strip():
                    st.markdown(txt)
                else:
                    _render_llm_answer(res)

                # 新增（4） 可选风险：抓取文本并做进一步分析
                with st.expander("可选：抓取相关新闻/公告文本并做进一步分析", expanded=False):
                    st.caption("尝试多数据源获取最近若干条与该股相关的新闻/公告（因 akshare 版本差异，若接口不可用会自动跳过）。")
                    max_items = st.slider("抓取条数", 3, 20, 8)
                    btn_fetch = st.button("抓取并生成分析", key=f"btn_risk_news_{market}_{symbol}")
                    if btn_fetch:
                        texts = []
                        try:
                            # 候选函数名与可能的参数结构（自动探测）
                            candidates = [
                                ("stock_news_em", {"symbol": symbol}),
                                ("stock_company_announcement_em", {"symbol": symbol}),
                                ("stock_notice", {"symbol": symbol}),
                                ("stock_zh_a_notice", {"symbol": symbol}),
                                ("stock_zh_a_news_em", {"symbol": symbol}),
                                ("stock_hk_announcement_em", {"symbol": symbol}),
                            ]
                            for name, params in candidates:
                                fn = getattr(ak, name, None)
                                if fn is None:
                                    continue
                                try:
                                    df_txt = fn(**params)  # type: ignore
                                except TypeError:
                                    # 有的函数使用不同参数名，尝试通配
                                    try:
                                        df_txt = fn(code=symbol)  # type: ignore
                                    except Exception:
                                        continue
                                except Exception:
                                    continue
                                if df_txt is None or df_txt.empty:
                                    continue
                                # 提取常见文本列拼
                                cols_pref = [
                                    "标题", "摘要", "内容", "公告标题", "公告内容", "新闻标题", "新闻内容", "文章标题", "简介", "题材"
                                ]
                                cols = [c for c in cols_pref if c in df_txt.columns]
                                if not cols:
                                    # 退化为选取第一列做标题
                                    cols = [df_txt.columns[0]] if len(df_txt.columns) > 0 else []
                                for _, r in df_txt.head(max_items).iterrows():
                                    parts = []
                                    for c in cols:
                                        try:
                                            val = str(r[c])
                                            if val and val != "nan":
                                                parts.append(val)
                                        except Exception:
                                            pass
                                    if parts:
                                        texts.append(" | ".join(parts))
                                if len(texts) >= max_items:
                                    break
                        except Exception:
                            pass
                        if texts:
                            st.success(f"已获取到 {len(texts)} 条文本。")
                            st.write("\n\n".join([f"- {t}" for t in texts[:max_items]]))
                            try:
                                sys_prompt2 = {"role":"system","content":"你是资深研究员。请基于给定的新闻/公告要点，结合该股的基本面与走势上下文，给出对未来三个月风险与催化的分析（中文、分点、可操作、不过度自信）。"}
                                messages2 = [sys_prompt2, {"role":"system","content":"基础上下文：\n" + "\n".join(ctx_lines)}, {"role":"user","content":"以下为抓取的文本要点：\n" + "\n".join(texts[:max_items])}]
                                # 下载Prompt（本次请求的完整messages2）
                                try:
                                    _fname = f"risk_news_{market}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                except Exception:
                                    _fname = f"risk_news_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                _prompt_text = format_messages_for_download(messages2)
                                st.download_button(
                                    label="下载Prompt",
                                    data=_prompt_text.encode('utf-8'),
                                    file_name=_fname,
                                    mime="text/plain",
                                    key=f"dl_prompt_risk_news_{market}_{symbol}"
                                )
                                res2 = chat_with_tools(router, messages2, tools_schema=tools, max_rounds=2)
                                txt2 = (res2 or {}).get("final_text") or ""
                                if txt2.strip():
                                    st.markdown(txt2)
                                else:
                                    _render_llm_answer(res2)
                            except Exception as e:
                                st.warning(f"基于文本的风险分析失败：{e}")
                        else:
                            st.info("未能抓取到可用文本（可能接口不可用或返回为空）")
            except Exception as e:
                st.error(f"生成风险提示失败：{e}")

    # 大模型问答（通过配置 + 路由/直连）
    st.subheader("大模型问答")
    # 展示当前模型（来源：合并后的路由 + 提供商配置）
    try:
        route_name = st.session_state.get("route_name", "default")
        registry_preview = ProviderRegistry(public_cfg_path="models.yaml", local_cfg_path="models.local.yaml")
        r = registry_preview.get_route(route_name)
        p = registry_preview.get_provider(r.provider)
        st.caption(f"当前模型：{p.name} · {r.model}（路由：{r.name}）")
    except Exception as _e:
        st.caption("当前模型：配置未就绪")

    inject_ctx = st.checkbox("将行情/回测摘要注入模型上下文", value=True)
    user_query = st.text_area("问模型：个股/板块信息、策略建议..", value="")
    if user_query:
        try:
            sys_prompt = {"role": "system", "content": "你是资深量化分析师。可以结合已知数据做基本面与技术面分析，并提醒数据来源且不构成投资建议。必要时请使用可用的联网工具（function calling）查询个股/板块实时信息，避免臆测。"}
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
                                            ctx_lines.append(f"所属板块: {ind_name}")
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
                    ctx_lines.append("如需查询个股或板块的实时信息，请按市场选择工具：A股用 fetch_stock_info_a，港股用 fetch_stock_info_hk，并传入当前 symbol；热点题材/新闻亦可通过工具检索。")
                    context_str = "\n".join(ctx_lines)
                    messages.append({"role": "system", "content": "以下是当前页面上下文，请结合回答问题：\n" + context_str})
                except Exception:
                    pass
            user_msg = {"role": "user", "content": user_query}
            messages.append(user_msg)

            # 下载Prompt（本次请求的完整messages）
            try:
                _fname = f"single_{market}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            except Exception:
                _fname = f"single_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            _prompt_text = format_messages_for_download(messages)
            st.download_button(
                label="下载Prompt",
                data=_prompt_text.encode('utf-8'),
                file_name=_fname,
                mime="text/plain",
                key=f"dl_prompt_single_{market}_{symbol}"
            )

            # 读取路由/模型设置
            route_name = st.session_state.get("route_name", "default")
            enable_tools = st.session_state.get("enable_tools", True)
            tools = get_tools_schema() if enable_tools else None
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
    # 旧版行业问答块（已禁用）已移除，避免与 industry_page 重复并消除潜在语法问题
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
            st.rerun()


# -------- 新增：异步片段（Fragment）定义，用于行业页的局部刷新 --------
@st.fragment
def _frag_industry_stats(ind: str):
    """异步片段：板块统计与成份股列表（仅刷新此区块）"""
    if not ind:
        st.info("请选择板块后使用异步统计")
        return
    # 局部控件（带前缀避免与页面控件冲突）
    col_sel2, col_sel3 = st.columns([1.4, 2.6])
    with col_sel2:
        mode = st.selectbox("统计模式", ["近N日", "时间周期"], index=0, key=f"frag_ind_mode_{ind}")
        N = st.number_input("N(日)", 5, 250, 20, 5, key=f"frag_ind_N_{ind}") if mode == "近N日" else None
    with col_sel3:
        if mode == "时间周期":
            default_end = pd.to_datetime("today").normalize()
            default_start = default_end - pd.Timedelta(days=30)
            s = st.date_input("开始", value=default_start.date(), key=f"frag_ind_start_{ind}")
            e = st.date_input("结束", value=default_end.date(), key=f"frag_ind_end_{ind}")
        else:
            s = e = None
        show_cons = st.checkbox("显示成份股列表", value=False, key=f"frag_ind_show_cons_{ind}")

    _c1, _c2 = st.columns([1, 1])
    with _c1:
        trig = st.button("计算统计", key=f"frag_ind_calc_btn_{ind}")
    with _c2:
        auto = st.checkbox("自动计算", value=False, key=f"frag_ind_auto_calc_{ind}")

    if not (trig or auto):
        st.caption("提示：点击“计算统计”或勾选“自动计算”开始，操作仅刷新本区块。")
        return

    # 片段内独立的自选添加工具，避免引用页面内局部函数
    def _wl_add(market: str, symbol: str):
        items = load_watchlist()
        if not any((it.get("market"), it.get("symbol")) == (market, symbol) for it in items):
            items.append({"market": market, "symbol": symbol})
            save_watchlist(items)
            try:
                st.toast(f"已加入自选：{symbol}")
            except Exception:
                st.success(f"已加入自选：{symbol}")
        else:
            try:
                st.toast("自选中已存在")
            except Exception:
                st.info("自选中已存在")

    try:
        # 成交量统计
        if mode == "近N日":
            metrics = compute_industry_volume_metrics(ind, int(N))
        else:
            metrics = compute_industry_volume_metrics_period(ind, s, e)
        curr = metrics.get("curr") or 0.0
        yoy = metrics.get("yoy")
        yoy_pct = metrics.get("yoy_pct")
        mom_pct = metrics.get("mom_pct")
        leaders = metrics.get("leaders") or []
        count = metrics.get("count")

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            title_v_curr = (f"近{int(N)}日板块成交量" if mode == "近N日" else "周期内板块成交量")
            st.metric(title_v_curr, format_cn_volume(curr))
        with m2:
            title_v_y = (f"去年同期{int(N)}日" if mode == "近N日" else "去年同期")
            st.metric(title_v_y, "-" if yoy is None else format_cn_volume(yoy))
        with m3:
            st.metric("同比", "-" if yoy_pct is None else f"{yoy_pct:.2%}")
        with m4:
            st.metric("环比", "-" if mom_pct is None else f"{mom_pct:.2%}")
        with m5:
            st.metric("成份股数", "-" if count is None else f"{int(count)}")

        # 新增：当无成份股或数据为空时，提前返回，避免空图与空列表渲染
        if not count or int(count) == 0:
            st.info("当前板块暂无成份股或数据为空，已跳过趋势与龙头展示。")
            return

        # 成交额统计
        if mode == "近N日":
            am = compute_industry_amount_metrics(ind, int(N))
        else:
            am = compute_industry_amount_metrics_period(ind, s, e)
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            title_a_curr = (f"近{int(N)}日板块成交额" if mode == "近N日" else "周期内板块成交额")
            st.metric(title_a_curr, "-" if am.get("curr") is None else format_cn_amount(am.get('curr',0)))
        with a2:
            title_a_y = (f"去年同期{int(N)}日" if mode == "近N日" else "去年同期")
            st.metric(title_a_y, "-" if am.get("yoy") is None else format_cn_amount(am.get('yoy',0)))
        with a3:
            st.metric("同比(额)", "-" if am.get("yoy_pct") is None else f"{am.get('yoy_pct'):.2%}")
        with a4:
            st.metric("环比(额)", "-" if am.get("mom_pct") is None else f"{am.get('mom_pct'):.2%}")

        # 趋势图
        tab1, tab2 = st.tabs(["成交量趋势", "成交额趋势"])
        with tab1:
            ser_v = compute_industry_agg_series(ind, "volume", days=int(N)) if mode == "近N日" else compute_industry_agg_series(ind, "volume", start_date=s, end_date=e)
            if not ser_v.empty:
                st.line_chart(ser_v.set_index("date")[ ["volume"] ], use_container_width=True)
            else:
                st.info("暂无趋势数据")
        with tab2:
            ser_a = compute_industry_agg_series(ind, "amount", days=int(N)) if mode == "近N日" else compute_industry_agg_series(ind, "amount", start_date=s, end_date=e)
            if not ser_a.empty:
                st.line_chart(ser_a.set_index("date")[ ["amount"] ], use_container_width=True)
            else:
                st.info("暂无趋势数据")

        # 龙头与成份
        st.subheader(f"板块龙头（按近{int(N)}日成交量，TOP5）" if mode == "近N日" else "板块龙头（按周期内成交量，TOP5）")
        if leaders:
            for i, row in enumerate(leaders):
                code = row.get("symbol"); name = row.get("name"); val = row.get("curr")
                c1, c2, c3, c4 = st.columns([3,2,1,1])
                c1.write(f"{name} ({code})")
                c2.write(f"近{int(N)}日量：{format_cn_volume(val)}" if mode == "近N日" else f"周期内量：{format_cn_volume(val)}")
                if c3.button("加自选", key=f"frag_lead_add_{ind}_{code}_{i}"):
                    _wl_add("A", code)
                if c4.button("详情", key=f"frag_lead_view_{ind}_{code}_{i}"):
                    go_detail("A", code)
        else:
            st.info("暂无可识别的龙头数据")

        if show_cons:
            st.subheader("成份股列表")
            cons = get_industry_cons(ind)
            if cons is not None and not cons.empty:
                st.dataframe(
                    ensure_arrow_compatible(cons.rename(columns={"symbol": "代码", "name": "名称"})),
                    use_container_width=True, hide_index=True, height=420
                )
            else:
                st.info("未获取到成份股数据")
    except Exception as e:
        st.warning(f"板块统计暂不可用：{e}")

@st.fragment
def _frag_industry_llm(ind: str):
    """异步片段：板块 LLM 问答（仅刷新此区块）。新增：对话历史按板块命名空间持久化与展示，避免刷新后内容丢失。"""
    # 为不同板块使用独立的历史键，避免跨板块/跨片段状态互相覆盖
    _ind_key = (ind or "all").strip() or "all"
    history_key = f"_frag_llm_history_{_ind_key}"
    kw_key = f"frag_llm_kw_{_ind_key}"
    inject_key = f"frag_llm_inject_{_ind_key}"
    query_key = f"frag_llm_query_{_ind_key}"
    send_key = f"frag_llm_send_{_ind_key}"
    clear_key = f"frag_llm_clear_{_ind_key}"

    # 兼容旧版本：如果存在旧的通用历史键且新键尚未初始化，则迁移一次
    if history_key not in st.session_state and "_frag_llm_history" in st.session_state:
        try:
            st.session_state[history_key] = list(st.session_state.get("_frag_llm_history", []))
        except Exception:
            st.session_state[history_key] = []

    # 初始化会话历史
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    kw_default = (ind or st.session_state.get("industry_keyword") or "半导体")
    kw = st.text_input("板块/主题关键词", value=kw_default, key=kw_key)
    st.session_state["industry_keyword"] = kw
    inject_ctx = st.checkbox("注入板块上下文", value=True, key=inject_key)

    # 历史对话展示与清空
    if st.session_state[history_key]:
        with st.expander("历史对话", expanded=True):
            for msg in st.session_state[history_key]:
                role = "用户" if msg.get("role") == "user" else "助手"
                st.markdown(f"**{role}**: {msg.get('content','')}")
        col_h1, col_h2 = st.columns([1,1])
        with col_h1:
            if st.button("清空历史", key=clear_key):
                st.session_state[history_key] = []
                st.rerun()
        with col_h2:
            st.caption("提示：发送新问题将追加到历史中")

    user_query = st.text_area("问模型：板块逻辑、景气度、龙头比较、估值与风险点", value="", key=query_key)

    if st.button("发送", key=send_key) and user_query:
        try:
            sys_prompt = {"role":"system","content":"你是资深板块分析师。给出条理清晰、可执行的板块研判，不构成投资建议。必要时请使用可用的联网工具（function calling）查询个股/板块实时信息。"}
            messages = [sys_prompt]
            if inject_ctx:
                ctx_lines = [
                    "页面: 板块信息(异步)",
                    f"板块关键词: {kw}",
                    f"已选板块: {ind or ''}",
                ]
                messages.append({"role":"system","content":"以下是当前页面上下文：\n" + "\n".join(ctx_lines)})
            messages.append({"role":"user","content": user_query})

            # 下载Prompt（本次请求的完整messages）
            try:
                _fname = f"industry_{_ind_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            except Exception:
                _fname = f"industry_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            _prompt_text = format_messages_for_download(messages)
            st.download_button(
                label="下载Prompt",
                data=_prompt_text.encode('utf-8'),
                file_name=_fname,
                mime="text/plain",
                key=f"dl_prompt_industry_{_ind_key}"
            )

            route_name = st.session_state.get("route_name","default")
            enable_tools = st.session_state.get("enable_tools", True)
            registry = ProviderRegistry(public_cfg_path="models.yaml", local_cfg_path="models.local.yaml")
            router = LLMRouter(registry=registry, route_name=route_name)
            tools = get_tools_schema() if enable_tools else None
            result = chat_with_tools(router, messages, tools_schema=tools, max_rounds=3)
            final_text = result.get("final_text") or ""

            # 保存历史对话：问题+回答
            st.session_state[history_key].append({"role": "user", "content": user_query})
            if final_text.strip():
                st.session_state[history_key].append({"role": "assistant", "content": final_text})
                st.markdown(final_text)
            else:
                _render_llm_answer(result)
                parsed_text = _extract_text_from_raw(result)
                st.session_state[history_key].append({"role": "assistant", "content": parsed_text or "(见上方结构化回答)"})
        except Exception as e:
            st.error(f"板块分析失败：{e}")

# -------- 行业信息页面 --------
def industry_page():
    st.header("板块信息")

    # 顶部展示 5 个自选行业（可点击切换）
    wl = load_industry_watchlist()
    chips = st.columns(5)
    for i in range(5):
        with chips[i]:
            if i < len(wl):
                nm = wl[i]
                if st.button(nm, key=f"ind_chip_{i}"):
                    st.session_state["_ind_selected"] = nm
                    st.rerun()
            else:
                st.write("")

    with st.expander("管理自选板块", expanded=False):
        all_inds = get_industry_list_all()
        add_name = st.selectbox("添加板块", options=[""] + all_inds, index=0, key="_ind_add_sel")
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("添加", key="btn_ind_add"):
                if add_name and add_name.strip():
                    if add_name not in wl:
                        wl.append(add_name)
                        save_industry_watchlist(wl)
                        st.success("已添加")
                        st.rerun()
                    else:
                        st.info("该板块已在自选中")
        with c2:
            if st.button("清空", key="btn_ind_clear"):
                save_industry_watchlist([])
                st.rerun()
        if wl:
            st.write("当前自选：")
            for idx, nm in enumerate(list(wl)):
                cc1, cc2, cc3, cc4 = st.columns([6,1,1,1])
                cc1.write(nm)
                with cc2:
                    if st.button("上移", key=f"btn_ind_up_{idx}") and idx > 0:
                        wl[idx-1], wl[idx] = wl[idx], wl[idx-1]
                        save_industry_watchlist(wl)
                        st.rerun()
                with cc3:
                    if st.button("下移", key=f"btn_ind_down_{idx}") and idx < len(wl)-1:
                        wl[idx+1], wl[idx] = wl[idx], wl[idx+1]
                        save_industry_watchlist(wl)
                        st.rerun()
                with cc4:
                    if st.button("删除", key=f"btn_ind_del_{idx}"):
                        wl2 = [x for x in wl if x != nm]
                        save_industry_watchlist(wl2)
                        st.rerun()

    # 新增：创建/编辑自定义行业
    with st.expander("创建/编辑自定义板块", expanded=False):
        custom_map = load_custom_industries()
        exist_custom_names = sorted(list(custom_map.keys())) if isinstance(custom_map, dict) else []
        sel = st.selectbox("选择已有或新建", options=["<新建>"] + exist_custom_names, index=0, key="_cid_sel")
        # 工作区的 session key
        sess_key = "_cid_items" if sel == "<新建>" else f"_cid_items_{sel}"
        # 行业名称输入
        default_name = "" if sel == "<新建>" else sel
        ind_name = st.text_input("板块名称", value=default_name, key=f"_cid_ind_name_{sel}")

        # 初始化工作列表
        if sess_key not in st.session_state:
            init_items = [] if sel == "<新建>" else custom_map.get(sel, [])
            # 基本清洗
            clean = []
            for it in init_items:
                if isinstance(it, dict):
                    sym = str(it.get("symbol", "")).strip()
                    nm = str(it.get("name", "")).strip()
                    if sym:
                        clean.append({"market": str(it.get("market", "A")) or "A", "symbol": sym, "name": nm})
            st.session_state[sess_key] = clean
        items = st.session_state.get(sess_key, [])

        st.caption("提示：A股代码建议不带交易所前缀，示例 000001、600519；保存后统计将自动抓取日线数据。")
        c1, c2, c3 = st.columns([1.4, 1.6, 1])
        with c1:
            add_sym = st.text_input("代码", key=f"_cid_add_sym_{sel}")
        with c2:
            add_nm = st.text_input("名称", key=f"_cid_add_nm_{sel}")
        with c3:
            st.write("")
            if st.button("添加成分", key=f"_cid_btn_add_{sel}"):
                s = (add_sym or "").strip()
                n = (add_nm or "").strip()
                if s:
                    # 去重按 symbol
                    exists = {it.get("symbol") for it in items}
                    if s not in exists:
                        items.append({"market": "A", "symbol": s, "name": n})
                        st.session_state[sess_key] = items
                    else:
                        st.info("该代码已在列表中")
                else:
                    st.warning("请填写代码")

        # 成分股条目
        if items:
            st.write("当前成分：")
            for i, it in enumerate(list(items)):
                d1, d2, d3 = st.columns([2.5, 3, 1])
                d1.write(str(it.get("symbol", "")))
                d2.write(str(it.get("name", "")))
                with d3:
                    if st.button("删除", key=f"_cid_del_{sel}_{i}"):
                        items.pop(i)
                        st.session_state[sess_key] = items
                        st.rerun()
        else:
            st.info("暂无成分，请先添加")

        b1, b2, b3 = st.columns([1,1,1])
        with b1:
            if st.button("保存/更新", key=f"_cid_save_{sel}"):
                nm = (ind_name or "").strip()
                if not nm:
                    st.warning("请填写板块名称")
                else:
                    # 写回并清理无效项
                    uniq = []
                    seen = set()
                    for it in items:
                        sym = str(it.get("symbol", "")).strip()
                        if not sym or sym in seen:
                            continue
                        seen.add(sym)
                        uniq.append({"market": "A", "symbol": sym, "name": str(it.get("name", "")).strip()})
                    custom_map[nm] = uniq
                    save_custom_industries(custom_map)
                    # 清理缓存，避免统计老数据
                    try:
                        compute_industry_volume_metrics.clear()
                        compute_industry_amount_metrics.clear()
                        compute_industry_agg_series.clear()
                        compute_industry_volume_metrics_period.clear()
                        compute_industry_amount_metrics_period.clear()
                    except Exception:
                        pass
                    st.success("已保存")
                    st.session_state["_ind_selected"] = nm
                    st.rerun()
        with b2:
            if sel != "<新建>" and st.button("删除板块", key=f"_cid_remove_{sel}"):
                if sel in custom_map:
                    custom_map.pop(sel, None)
                    save_custom_industries(custom_map)
                    try:
                        compute_industry_volume_metrics.clear()
                        compute_industry_amount_metrics.clear()
                        compute_industry_agg_series.clear()
                        compute_industry_volume_metrics_period.clear()
                        compute_industry_amount_metrics_period.clear()
                    except Exception:
                        pass
                    st.success("已删除")
                    st.session_state.pop(sess_key, None)
                    st.rerun()
        with b3:
            if st.button("清空成分股", key=f"_cid_clear_{sel}"):
                st.session_state[sess_key] = []
                st.rerun()

    st.markdown("---")

    # 行业选择 + 统计（合并“自选行业 ∪ 官方行业”，保证点击自选按钮后选项必然可选）
    all_inds_db = get_industry_list_all()
    # 合并去重，优先展示自选顺序
    merged_opts: List[str] = []
    seen = set()
    for x in (wl + all_inds_db):
        if x and x not in seen:
            seen.add(x)
            merged_opts.append(x)
    # 若 session 中已有选择但不在列表，追加保证可选
    curr_sel = st.session_state.get("_ind_selected")
    if curr_sel and curr_sel not in seen:
        merged_opts = [curr_sel] + merged_opts
    default_ind = curr_sel or (wl[0] if wl else (merged_opts[0] if merged_opts else ""))
    idx_default = merged_opts.index(default_ind) if (default_ind and default_ind in merged_opts) else 0

    col_sel1, col_sp2, col_sp3, col_sp4 = st.columns([2,1.4,1.8,2])
    with col_sel1:
        if merged_opts:
            ind = st.selectbox("选择板块", options=merged_opts, index=idx_default, key="_ind_selected")
        else:
            st.info("暂无板块列表，请在上方“管理自选板块”中添加，或稍后再试")
            ind = ""
    with col_sp2:
        st.caption("统计控制与成份列表已移至下方异步片段")
    with col_sp3:
        st.write("")
    with col_sp4:
        st.write("")

    # 统计控制已由异步片段覆盖（保留占位，避免重复）
    _ctrl1, _ctrl2 = st.columns([1, 1])
    with _ctrl1:
        st.caption("统计已由下方异步模块驱动")
    with _ctrl2:
        pass
    _do_calc = False

    # ---- 行业页异步片段（局部刷新）----
    st.markdown("---")
    frag_col1, frag_col2 = st.columns([2.2, 1.8])
    with frag_col1:
        _frag_industry_stats(ind)
    with frag_col2:
        _frag_industry_llm(ind)
    # ---- End fragment zone ----

    def _add_to_watchlist_if_absent(market: str, symbol: str):
        items = load_watchlist()
        if not any((it.get("market"), it.get("symbol")) == (market, symbol) for it in items):
            items.append({"market": market, "symbol": symbol})
            save_watchlist(items)
            try:
                st.toast(f"已加入自选：{symbol}")
            except Exception:
                st.success(f"已加入自选：{symbol}")
        else:
            try:
                st.toast("自选中已存在")
            except Exception:
                st.info("自选中已存在")

    if _do_calc:
        try:
            if mode == "近N日":
                metrics = compute_industry_volume_metrics(ind, int(N))
            else:
                metrics = compute_industry_volume_metrics_period(ind, s, e)
            curr = metrics.get("curr") or 0.0
            yoy = metrics.get("yoy")
            prev = metrics.get("prev")
            yoy_pct = metrics.get("yoy_pct")
            mom_pct = metrics.get("mom_pct")
            leaders = metrics.get("leaders") or []
            count = metrics.get("count")

            # 成交量汇总
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                title_v_curr = "近N日板块成交量(股)" if mode == "近N日" else "周期内板块成交量(股)"
                st.metric(title_v_curr, f"{curr:,.0f}")
            with m2:
                title_v_y = "去年同期N日(股)" if mode == "近N日" else "去年同期(股)"
                st.metric(title_v_y, "-" if yoy is None else f"{yoy:,.0f}")
            with m3:
                st.metric("同比", "-" if yoy_pct is None else f"{yoy_pct:.2%}")
            with m4:
                st.metric("环比", "-" if mom_pct is None else f"{mom_pct:.2%}")
            with m5:
                st.metric("成份股数", "-" if count is None else f"{int(count)}")

            # 成交额统计
            if mode == "近N日":
                am = compute_industry_amount_metrics(ind, int(N))
            else:
                am = compute_industry_amount_metrics_period(ind, s, e)
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                title_a_curr = "近N日板块成交额(元)" if mode == "近N日" else "周期内板块成交额(元)"
                st.metric(title_a_curr, "-" if am.get("curr") is None else f"{am.get('curr',0):,.0f}")
            with a2:
                title_a_y = "去年同期N日(元)" if mode == "近N日" else "去年同期(元)"
                st.metric(title_a_y, "-" if am.get("yoy") is None else f"{am.get('yoy',0):,.0f}")
            with a3:
                st.metric("同比(额)", "-" if am.get("yoy_pct") is None else f"{am.get('yoy_pct'):.2%}")
            with a4:
                st.metric("环比(额)", "-" if am.get("mom_pct") is None else f"{am.get('mom_pct'):.2%}")

            # 小型趋势图
            tab1, tab2 = st.tabs(["成交量趋势", "成交额趋势"])
            with tab1:
                if mode == "近N日":
                    ser_v = compute_industry_agg_series(ind, "volume", days=int(N))
                else:
                    ser_v = compute_industry_agg_series(ind, "volume", start_date=s, end_date=e)
                if not ser_v.empty:
                    st.line_chart(ser_v.set_index("date")[ ["volume"] ], use_container_width=True)
                else:
                    st.info("暂无趋势数据")
            with tab2:
                if mode == "近N日":
                    ser_a = compute_industry_agg_series(ind, "amount", days=int(N))
                else:
                    ser_a = compute_industry_agg_series(ind, "amount", start_date=s, end_date=e)
                if not ser_a.empty:
                    st.line_chart(ser_a.set_index("date")[ ["amount"] ], use_container_width=True)
                else:
                    st.info("暂无趋势数据")

            st.subheader("板块龙头（按近N日成交量，TOP5）")
            if leaders:
                for i, row in enumerate(leaders):
                    code = row.get("symbol")
                    name = row.get("name")
                    val = row.get("curr")
                    c1, c2, c3, c4 = st.columns([3,2,1,1])
                    c1.write(f"{name} ({code})")
                    c2.write(f"近N日量：{val:,.0f}")
                    if c3.button("加自选", key=f"lead_add_{code}_{i}"):
                        _add_to_watchlist_if_absent("A", code)
                    if c4.button("详情", key=f"lead_view_{code}_{i}"):
                        go_detail("A", code)
            else:
                st.info("暂无可识别的龙头数据")

            if show_cons:
                st.subheader("成份股列表")
                cons = get_industry_cons(ind)
                if cons is not None and not cons.empty:
                    st.dataframe(
                        ensure_arrow_compatible(cons.rename(columns={"symbol": "代码", "name": "名称"})),
                        use_container_width=True, hide_index=True, height=420
                    )
                else:
                    st.info("未获取到成份股数据")
        except Exception as e:
            st.warning(f"板块统计暂不可用：{e}")
    elif ind:
        st.info("为提升首页首屏速度，板块统计默认不自动执行。请点击上方“计算统计”或勾选“自动计算”后查看结果。")

    st.markdown("---")

    # 同类行业对比
    with st.expander("同类板块对比", expanded=False):
        # 复用上面合并后的行业列表 merged_opts
        opts = [x for x in merged_opts if x]
        picks = st.multiselect("选择待对比板块(<=3)", options=opts, default=[], key="_ind_compare")
        if len(picks) > 3:
            st.warning("最多选择 3 个板块进行对比，已自动截取前 3 个")
            picks = picks[:3]
        if picks:
            cols = st.columns(len(picks))
            for i, nm in enumerate(picks):
                with cols[i]:
                    st.markdown(f"#### {nm}")
                    try:
                        m_v = compute_industry_volume_metrics(nm, int(N))
                        m_a = compute_industry_amount_metrics(nm, int(N))
                        st.metric("量·近N日", f"{(m_v.get('curr') or 0):,.0f}")
                        st.metric("量·同比", "-" if m_v.get("yoy_pct") is None else f"{m_v.get('yoy_pct'):.2%}")
                        st.metric("量·环比", "-" if m_v.get("mom_pct") is None else f"{m_v.get('mom_pct'):.2%}")
                        st.metric("额·近N日", f"{(m_a.get('curr') or 0):,.0f}")
                        st.metric("额·同比", "-" if m_a.get("yoy_pct") is None else f"{m_a.get('yoy_pct'):.2%}")
                        st.metric("额·环比", "-" if m_a.get("mom_pct") is None else f"{m_a.get('mom_pct'):.2%}")
                        if mode == "近N日":
                            ser = compute_industry_agg_series(nm, "volume", days=int(N))
                        else:
                            ser = compute_industry_agg_series(nm, "volume", start_date=s, end_date=e)
                        if not ser.empty:
                            st.line_chart(ser.set_index("date")[ ["volume"] ], use_container_width=True)
                    except Exception as e:
                        st.info(f"{nm} 统计失败：{e}")

    # 旧版同步 LLM 问答区：已由异步片段覆盖，保留代码但不渲染
    if False:
        kw_default = (ind or st.session_state.get("industry_keyword") or "半导体")
        kw = st.text_input("板块/主题关键词", value=kw_default)
        st.session_state["industry_keyword"] = kw
        inject_ctx = st.checkbox("注入板块上下文", value=True)

        user_query = st.text_area("问模型：板块逻辑、景气度、龙头比较、估值与风险点", value="")
        if user_query:
            try:
                sys_prompt = {"role":"system","content":"你是资深板块分析师。给出条理清晰、可执行的板块研判，不构成投资建议。必要时请使用可用的联网工具（function calling）查询个股/板块实时信息。"}
                messages = [sys_prompt]
                if inject_ctx:
                    ctx_lines = [
                        "页面: 板块信息",
                        f"板块关键词: {kw}",
                        f"已选板块: {ind or ''}",
                        f"N日窗口: {int(N) if ind else ''}",
                        "如需获取成份股或个股数据，可按市场调用工具：A股用 fetch_stock_info_a，港股用 fetch_stock_info_hk。",
                    ]
                    messages.append({"role":"system","content":"以下是当前页面上下文：\n" + "\n".join(ctx_lines)})
                messages.append({"role":"user","content": user_query})

                route_name = st.session_state.get("route_name","default")
                enable_tools = st.session_state.get("enable_tools", True)
                registry = ProviderRegistry(public_cfg_path="models.yaml", local_cfg_path="models.local.yaml")
                router = LLMRouter(registry=registry, route_name=route_name)
                tools = get_tools_schema() if enable_tools else None
                result = chat_with_tools(router, messages, tools_schema=tools, max_rounds=3)
                final_text = result.get("final_text") or ""
                if final_text.strip():
                    st.markdown(final_text)
                else:
                    _render_llm_answer(result)
            except Exception as e:
                st.error(f"板块分析失败：{e}")


# -------- 数据初始化页面 --------

def data_init_page():
    st.header("数据初始化 / 历史行情缓存")
    st.caption("批量下载并缓存 A 股与港股通历史日线。支持复权方式、并发下载、失败重试、仅下载未缓存日期，以及速度/剩余时间估计。")

    # 新增：数据更新历史记录查看/筛选/导出/清空
    with st.expander("数据更新历史记录", expanded=False):
        colhx1, colhx2, colhx3 = st.columns([1,1,3])
        with colhx1:
            if st.button("刷新记录", key="btn_hist_refresh"):
                st.session_state["_hist_ver"] = st.session_state.get("_hist_ver", 0) + 1
        with colhx2:
            if st.button("清空历史记录", key="btn_hist_clear"):
                try:
                    if UPDATE_HISTORY_PATH.exists():
                        UPDATE_HISTORY_PATH.unlink()
                    st.success("已清空历史记录")
                except Exception as e:
                    st.warning(f"清空失败：{e}")
                finally:
                    st.session_state["_hist_ver"] = st.session_state.get("_hist_ver", 0) + 1
        ver = st.session_state.get("_hist_ver", 0)
        dfh = load_update_history_cached(ver)
        if dfh is None or dfh.empty:
            st.info("暂无记录")
        else:
            f1, f2, f3 = st.columns([1,1,2])
            with f1:
                market_filter = st.selectbox("市场", ["全部", "A", "H"], index=0)
            with f2:
                days = st.slider("时间范围(天)", min_value=1, max_value=90, value=7)
            with f3:
                q = st.text_input("代码/名称包含", value="")
            dfv = dfh.copy()
            if market_filter != "全部" and "market" in dfv.columns:
                dfv = dfv[dfv["market"] == market_filter]
            cutoff = datetime.now() - timedelta(days=days)
            if "ts" in dfv.columns:
                dfv = dfv[dfv["ts"] >= cutoff]
            if q:
                dfv = dfv[(dfv.get("symbol", pd.Series(dtype=str)).astype(str).str.contains(q, case=False, na=False)) |
                          (dfv.get("name", pd.Series(dtype=str)).astype(str).str.contains(q, case=False, na=False))]
            total_added = int(dfv.get("added_rows", pd.Series(dtype=float)).fillna(0).sum()) if "added_rows" in dfv.columns else 0
            st.caption(f"记录数：{len(dfv)}，新增总行：{total_added}")
            show_cols = [c for c in ["ts","market","symbol","name","start","end","added_rows","duration_s","skipped","error","adjust","source"] if c in dfv.columns]
            st.dataframe(ensure_arrow_compatible(dfv[show_cols].head(500)), use_container_width=True)
            try:
                csv = dfv.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("导出CSV(筛选后)", data=csv, file_name="update_history.csv", mime="text/csv", key="btn_hist_dl")
            except Exception:
                pass

    # 懒加载控制：进入页面不自动加载标的列表与不自动开始下载
    col_lazy1, col_lazy2 = st.columns([1,1])
    with col_lazy1:
        auto_load_lists = st.checkbox("自动加载标的列表", value=False, key="auto_load_lists", help="为提升首屏速度，默认不自动加载。")
    with col_lazy2:
        auto_start_download = st.checkbox("自动开始下载", value=False, key="auto_start_download", help="不建议默认自动下载，避免长任务误触发。")

    # 一次性自动触发开关（防止每次重绘重复执行）
    auto_load_lists_trig = False
    if auto_load_lists and not st.session_state.get("_auto_load_lists_done"):
        auto_load_lists_trig = True
        st.session_state["_auto_load_lists_done"] = True
    auto_start_download_trig = False
    if auto_start_download and not st.session_state.get("_auto_start_download_done"):
        auto_start_download_trig = True
        st.session_state["_auto_start_download_done"] = True

    # 全局区间
    today = datetime.now().date()
    default_start = (today - timedelta(days=3650))
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("开始日期", value=default_start, key="init_start")
    with c2:
        end_date = st.date_input("结束日期", value=today, key="init_end")

    # 控制项
    c3, c4, c5, c6 = st.columns(4)
    with c3:
        adj_label = st.selectbox("A股复权方式", ["不复权", "前复权", "后复权"], index=0, help="仅 A 股生效")
        adj_map = {"不复权": None, "前复权": "qfq", "后复权": "hfq"}
        adjust = adj_map.get(adj_label)
    with c4:
        max_workers = st.slider("并发任务数", min_value=1, max_value=16, value=8, step=1)
    with c5:
        max_retries = st.slider("失败重试次数", min_value=0, max_value=5, value=2, step=1)
    with c6:
        skip_cached = st.checkbox("仅下载未缓存日期", value=True, help="若缓存已覆盖到结束日期则跳过；否则仅从最后缓存日期的次日开始下载并合并保存。")

    show_detail = st.checkbox("显示详细日志", value=False)

    st.markdown("---")
    colA, colH = st.columns(2)

    def _merge_and_save(client: AKDataClient, market: str, symbol: str, adjust: str | None, df_new: pd.DataFrame,
                        start_date_v: datetime.date, end_date_v: datetime.date):
        if df_new is None or df_new.empty:
            return 0
        # 过滤区间
        try:
            s = pd.to_datetime(start_date_v)
            e = pd.to_datetime(end_date_v)
            df_new = df_new[(df_new["date"] >= s) & (df_new["date"] <= e)]
        except Exception:
            pass
        try:
            df_old = client._load_cached_daily(market, symbol, adjust)  # type: ignore
        except Exception:
            df_old = pd.DataFrame()
        if df_old is not None and not df_old.empty:
            try:
                df_old["date"] = pd.to_datetime(df_old["date"])
            except Exception:
                pass
            merged = pd.concat([df_old, df_new], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            added = len(merged) - len(df_old)
        else:
            merged = df_new.sort_values("date")
            added = len(merged)
        try:
            client._save_cached_daily(merged, market, symbol, adjust)  # type: ignore
        except Exception:
            pass
        return max(0, int(added))

    def _compute_fetch_window(client: AKDataClient, market: str, symbol: str, adjust: str | None,
                              start_date_v: datetime.date, end_date_v: datetime.date):
        # 仅用于确定起始日期，避免完全重复抓取
        if not skip_cached:
            return start_date_v, end_date_v
        try:
            df_old = client._load_cached_daily(market, symbol, adjust)  # type: ignore
        except Exception:
            df_old = pd.DataFrame()
        if df_old is None or df_old.empty or "date" not in df_old.columns:
            return start_date_v, end_date_v
        try:
            last_dt = pd.to_datetime(df_old["date"]).max().date()
        except Exception:
            return start_date_v, end_date_v
        fetch_start = max(start_date_v, last_dt + timedelta(days=1))
        if fetch_start > end_date_v:
            return None, None
        return fetch_start, end_date_v

    # 通用并发执行器
    def _run_concurrent(list_df: pd.DataFrame, market: str, adjust_for_market: str | None):
        if list_df is None or list_df.empty:
            st.warning("标的列表为空")
            return
        total = len(list_df)
        bar = st.progress(0)
        status = st.empty()
        metrics = st.empty()
        detail = st.empty() if show_detail else None
        start_ts = time.time()
        completed = 0
        total_added = 0
        errors = 0

        client = get_client()

        def task(symbol: str, name: str):
            nonlocal total_added
            t0 = time.time()
            tries = 0
            last_err = None
            while tries <= max_retries:
                try:
                    # 计算起止，仅下载未缓存区间
                    fw = _compute_fetch_window(client, market, symbol, adjust_for_market, start_date, end_date)
                    if fw == (None, None):
                        return 0, 0.0, f"已覆盖，跳过"
                    s_fetch, e_fetch = fw
                    # 使用统一接口，确保标准化一致
                    df_new = client.get_hist(market=market, symbol=symbol, period="daily",
                                             start=s_fetch.isoformat(), end=e_fetch.isoformat(),
                                             adjust=adjust_for_market, use_cache=True, refresh=False)
                    added = _merge_and_save(client, market, symbol, adjust_for_market, df_new, start_date, end_date)
                    dt = time.time() - t0
                    return added, dt, None
                except Exception as e:
                    last_err = e
                    tries += 1
                    time.sleep(min(1.5 * tries, 5))
            return 0, time.time() - t0, str(last_err)

        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for _, row in list_df.iterrows():
                code = str(row.get("代码", "")).strip()
                name = str(row.get("名称", "")).strip()
                if not code:
                    continue
                futures[exe.submit(task, code, name)] = (code, name)

            for fut in as_completed(futures):
                code, name = futures[fut]
                added, dt, err = fut.result()
                completed += 1
                if err:
                    errors += 1
                    msg = f"失败 {name}（{code}）：{err}"
                    if detail is not None:
                        detail.write(msg)
                else:
                    total_added += added
                    if show_detail and detail is not None:
                        detail.write(f"完成 {name}（{code}）：+{added} 行，用时 {dt:.1f}s")

                # 新增：写入更新历史（仅记录有新增数据的）
                try:
                    if int(added or 0) > 0 and not err:
                        _append_update_history({
                            "source": "data_init",
                            "market": market,
                            "symbol": code,
                            "name": name,
                            "adjust": adjust_for_market,
                            "start": None,  # 当前版本 task 未返回精确起止，保留字段
                            "end": None,
                            "added_rows": int(added or 0),
                            "duration_s": round(float(dt or 0.0), 3),
                            "error": None,
                            "skipped": False,
                        })
                except Exception:
                    pass

                # 进度与速度/ETA
                elapsed = time.time() - start_ts
                speed_task = completed / elapsed if elapsed > 0 else 0.0
                eta = (total - completed) / speed_task if speed_task > 0 else 0.0
                bar.progress(int(completed / max(1, total) * 100))
                status.info(f"进度：{completed}/{total}，新增行：{total_added}，错误：{errors}")
                metrics.caption(f"速度：{speed_task:.2f} 个标的/秒，预计剩余：{eta:.1f} 秒")

        status.success(f"完成：共处理 {total} 个标的，新增 {total_added} 行，错误 {errors}")

    with colA:
        st.subheader("A股 初始化")
        # 按需加载 + 会话缓存，避免初次渲染即耗时拉取
        if st.button("加载/刷新 A股列表", key="load_list_a") or auto_load_lists_trig:
            with st.spinner("正在加载 A股列表..."):
                _list = get_a_stock_list_cached()
                st.session_state["_list_a"] = _list
            st.success("A股列表已加载")
        list_a = st.session_state.get("_list_a")
        tot_a = (0 if list_a is None or list_a.empty else len(list_a)) if list_a is not None else None
        st.write(f"标的数量：{tot_a if tot_a is not None else '未加载'}")
        if st.button("开始下载 A股", key="btn_init_a") or auto_start_download_trig:
            if list_a is None:
                with st.spinner("首次使用：正在加载 A股列表..."):
                    list_a = get_a_stock_list_cached()
                    st.session_state["_list_a"] = list_a
            _run_concurrent(list_a, market="A", adjust_for_market=adjust)

    with colH:
        st.subheader("港股通 初始化")
        # 新增：强制刷新缓存选项
        force_refresh_h = st.checkbox("强制刷新缓存(港股通)", value=False, key="force_refresh_h", help="清空缓存后重新拉取港股通列表")
        # 按需加载 + 会话缓存，避免初次渲染即耗时拉取
        if st.button("加载/刷新 港股通列表", key="load_list_h") or auto_load_lists_trig:
            if st.session_state.get("force_refresh_h"):
                try:
                    st.cache_data.clear()
                    st.info("已清空缓存，开始重新拉取……")
                except Exception:
                    pass
            with st.spinner("正在加载 港股通列表..."):
                _list = get_hk_ggt_list_cached()
                if _list is not None and not _list.empty and "代码" in _list.columns:
                    _list["代码"] = (
                        _list["代码"].astype(str).str.upper().str.replace(".HK", "", regex=False).str.lstrip("0").apply(lambda s: s.zfill(5))
                    )
                st.session_state["_list_h"] = _list
            st.success("港股通列表已加载")
            # 调试信息展示
            try:
                dbg_sources = st.session_state.get("_dbg_hk_ggt_sources")
                dbg_final = st.session_state.get("_dbg_hk_ggt_final")
                with st.expander("调试：港股通数据来源与清洗", expanded=False):
                    if dbg_sources:
                        st.write("来源抓取尝试：", len(dbg_sources))
                        st.json(dbg_sources)
                    _list_preview = st.session_state.get("_list_h")
                    if _list_preview is not None and not _list_preview.empty:
                        st.write("最终列表预览 (前20)：")
                        st.dataframe(ensure_arrow_compatible(_list_preview.head(20)))
                    if dbg_final:
                        st.write("最终结构：")
                        st.json(dbg_final)
                    if not dbg_sources:
                        st.caption("提示：若未显示来源信息，可能是缓存未刷新或上游接口无数据。可勾选‘强制刷新缓存’后重试。")
            except Exception:
                pass
        list_h = st.session_state.get("_list_h")
        tot_h = (0 if list_h is None or list_h.empty else len(list_h)) if list_h is not None else None
        st.write(f"标的数量：{tot_h if tot_h is not None else '未加载'}")
        if st.button("开始下载 港股通", key="btn_init_h") or auto_start_download_trig:
            if list_h is None:
                with st.spinner("首次使用：正在加载 港股通列表..."):
                    list_h = get_hk_ggt_list_cached()
                    if list_h is not None and not list_h.empty and "代码" in list_h.columns:
                        list_h["代码"] = (
                            list_h["代码"].astype(str).str.upper().str.replace(".HK", "", regex=False).str.lstrip("0").apply(lambda s: s.zfill(5))
                        )
                    st.session_state["_list_h"] = list_h
            _run_concurrent(list_h, market="H", adjust_for_market=None)

# -------- 工具筛选页面（示例）--------

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
        provider_label = st.selectbox("提供商（路由）", ["default","fast","qwen","fallback","analysis"], index=0, help="使用 routing.yaml 中的路由", key="provider_route")
        st.session_state["route_name"] = provider_label
        st.session_state["enable_tools"] = st.checkbox("启用联网工具(Function Calling)", value=True)
        st.markdown("---")
        st.header("导航")
        st.info("上次筛选/自选/查询的股票可跳转至详情页。")

    tabs = st.tabs(["单股查询", "自选股", "板块信息", "数据初始化", "工具筛选"])
    with tabs[0]:
        single_stock_page()
    with tabs[1]:
        watchlist_page()
    with tabs[2]:
        industry_page()
    with tabs[3]:
        data_init_page()
    with tabs[4]:
        tools_filter_page()

if __name__ == "__main__":
    main()

