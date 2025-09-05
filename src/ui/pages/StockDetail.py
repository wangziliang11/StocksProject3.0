import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any

# 保证可以从 src 包导入（将项目根目录加入 sys.path）
ROOT = Path(__file__).resolve().parents[3]  # 指向 <project> 根目录
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.app import single_stock_page, load_watchlist, save_watchlist, get_stock_name_cached  # noqa: E402


def _read_query_params() -> Dict[str, Any]:
    try:
        # 新版本 Streamlit
        params = dict(st.query_params)
    except Exception:
        # 旧版本兼容
        try:
            params = st.experimental_get_query_params()
        except Exception:
            params = {}
    return params or {}


def main():
    st.set_page_config(page_title="股票详情", layout="wide")
    params = _read_query_params()

    # 从会话或 URL 获取股票
    market = st.session_state.get("detail_market", None)
    symbol = st.session_state.get("detail_symbol", None)
    if not market and "market" in params:
        market = params.get("market")
        if isinstance(market, list):
            market = market[0]
    if not symbol and "symbol" in params:
        symbol = params.get("symbol")
        if isinstance(symbol, list):
            symbol = symbol[0]

    # 设置默认值
    if not market:
        market = "A"
    if not symbol:
        symbol = "600519" if market == "A" else "00700"

    st.session_state["detail_market"] = market
    st.session_state["detail_symbol"] = symbol

    name = get_stock_name_cached(market, symbol)
    st.title(f"股票详情 - {name}({symbol})" if name else f"股票详情 - {symbol}")

    # 导航
    try:
        st.page_link("app.py", label="返回主页")
    except Exception:
        st.caption("提示：可使用浏览器后退返回主页。")

    # 自选操作
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("加入自选"):
            wl = load_watchlist()
            if not any(x.get("market") == market and x.get("symbol") == symbol for x in wl):
                wl.append({"market": market, "symbol": symbol})
                save_watchlist(wl)
                st.success("已加入自选")
            else:
                st.info("该股票已在自选中")

    # 直接复用主页面的单股查询模块（含历史数据、K线、指标与回测、LLM）
    single_stock_page()


if __name__ == "__main__":
    main()