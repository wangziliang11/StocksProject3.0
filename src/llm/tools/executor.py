from typing import Dict, Any
import akshare as ak
import pandas as pd


def fetch_stock_info_a(symbol: str) -> Dict[str, Any]:
    symbol = (symbol or "").strip()
    if not symbol:
        return {"ok": False, "error": "symbol is required"}
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        data = df.to_dict(orient="records")
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def fetch_stock_info_hk(symbol: str) -> Dict[str, Any]:
    """
    港股个股实时/基础信息：
    - 现阶段优先返回实时报价行（包含名称、最新价、涨跌幅、成交额等）
    - 行业字段若来源不稳定，则暂不强制
    """
    symbol = (symbol or "").strip()
    if not symbol:
        return {"ok": False, "error": "symbol is required"}
    try:
        df_spot = ak.stock_hk_spot_em()
        if df_spot is None or df_spot.empty:
            return {"ok": False, "error": "no hk spot data"}
        # 兼容不同列名
        code_col = None
        for c in ["代码", "symbol", "code", "证券代码"]:
            if c in df_spot.columns:
                code_col = c
                break
        if code_col is None:
            return {"ok": False, "error": "unknown code column in hk spot"}
        sym = str(symbol).upper().replace(".HK", "")
        df_spot["_code_norm"] = df_spot[code_col].astype(str).str.upper().str.replace(".HK", "", regex=False).str.lstrip("0")
        sym_norm = sym.lstrip("0")
        row = df_spot[df_spot["_code_norm"] == sym_norm].drop(columns=["_code_norm"])  # type: ignore
        if row.empty:
            return {"ok": False, "error": "symbol not found in hk spot"}
        data = row.to_dict(orient="records")
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}