"""
优化的板块统计计算模块
使用成分股缓存来提高计算效率和可靠性
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def get_cached_sector_components(sector_name: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    获取板块成分股（优先使用缓存）
    
    Args:
        sector_name: 板块名称
        force_refresh: 是否强制刷新缓存
        
    Returns:
        成分股DataFrame，包含symbol和name列
    """
    try:
        from src.data.sector_component_cache import sector_component_cache
        components_df = sector_component_cache.get_sector_components(sector_name, force_refresh)
        
        if components_df is not None and not components_df.empty:
            logger.info(f"从缓存获取板块 {sector_name} 成分股，共 {len(components_df)} 只")
            return components_df
        else:
            logger.warning(f"缓存中未找到板块 {sector_name} 的成分股数据")
            return None
            
    except Exception as e:
        logger.error(f"从缓存获取板块 {sector_name} 成分股失败: {e}")
        return None

def compute_industry_volume_metrics_optimized(industry_name: str, N: int) -> Dict[str, Any]:
    """
    优化的板块成交量统计计算（使用缓存）
    
    Args:
        industry_name: 板块名称
        N: 统计天数
        
    Returns:
        统计结果字典
    """
    # 优先从缓存获取成分股
    cons = get_cached_sector_components(industry_name)
    
    # 如果缓存获取失败，回退到原有方式
    if cons is None or cons.empty:
        logger.warning(f"缓存获取失败，回退到原有方式获取板块 {industry_name} 成分股")
        try:
            from src.ui.app import get_industry_cons
            cons = get_industry_cons(industry_name)
        except Exception as e:
            logger.error(f"回退方式也失败: {e}")
            return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None, "leaders": [], "count": 0}
    
    if cons is None or cons.empty:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None, "leaders": [], "count": 0}
    
    # 计算统计指标
    curr_sum = 0.0
    prev_sum = 0.0
    yoy_sum = 0.0
    prev_has = False
    yoy_has = False
    leaders: List[Dict[str, Any]] = []
    
    # 批量处理成分股数据
    valid_count = 0
    for _, row in cons.iterrows():
        sym = str(row.get("symbol", "")).strip()
        name = str(row.get("name", "")).strip()
        if not sym:
            continue
            
        try:
            # 使用原有的个股计算逻辑
            from src.ui.app import compute_symbol_volume_metrics
            m = compute_symbol_volume_metrics("A", sym, N)
            
            curr_sum += m.get("curr") or 0.0
            if m.get("prev") is not None:
                prev_sum += m["prev"] or 0.0
                prev_has = True
            if m.get("yoy") is not None:
                yoy_sum += m["yoy"] or 0.0
                yoy_has = True
            
            leaders.append({"symbol": sym, "name": name, "curr": m.get("curr", 0.0)})
            valid_count += 1
            
        except Exception as e:
            logger.warning(f"计算个股 {sym} 指标失败: {e}")
            continue
    
    # 选龙头：按近N日成交量排序取前5
    leaders = sorted(leaders, key=lambda x: x.get("curr", 0.0), reverse=True)[:5]
    
    # 计算同比和环比
    yoy_val = yoy_sum if yoy_has else None
    prev_val = prev_sum if prev_has else None
    yoy_pct = (curr_sum - yoy_val) / yoy_val if (yoy_val and yoy_val > 0) else None
    mom_pct = (curr_sum - prev_val) / prev_val if (prev_val and prev_val > 0) else None
    
    logger.info(f"板块 {industry_name} 统计完成，有效成分股: {valid_count}/{len(cons)}")
    
    return {
        "curr": curr_sum, 
        "yoy": yoy_val, 
        "prev": prev_val, 
        "yoy_pct": yoy_pct, 
        "mom_pct": mom_pct, 
        "leaders": leaders, 
        "count": valid_count
    }

def compute_industry_volume_metrics_period_optimized(industry_name: str, start_date: Any, end_date: Any) -> Dict[str, Any]:
    """
    优化的板块成交量周期统计计算（使用缓存）
    
    Args:
        industry_name: 板块名称
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        统计结果字典
    """
    # 优先从缓存获取成分股
    cons = get_cached_sector_components(industry_name)
    
    # 如果缓存获取失败，回退到原有方式
    if cons is None or cons.empty:
        logger.warning(f"缓存获取失败，回退到原有方式获取板块 {industry_name} 成分股")
        try:
            from src.ui.app import get_industry_cons
            cons = get_industry_cons(industry_name)
        except Exception as e:
            logger.error(f"回退方式也失败: {e}")
            return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None, "leaders": [], "count": 0}
    
    if cons is None or cons.empty:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None, "leaders": [], "count": 0}
    
    # 计算统计指标
    curr_sum = 0.0
    prev_sum = 0.0
    yoy_sum = 0.0
    prev_has = False
    yoy_has = False
    leaders: List[Dict[str, Any]] = []
    
    # 批量处理成分股数据
    valid_count = 0
    for _, row in cons.iterrows():
        sym = str(row.get("symbol", "")).strip()
        name = str(row.get("name", "")).strip()
        if not sym:
            continue
            
        try:
            # 使用原有的个股计算逻辑
            from src.ui.app import compute_symbol_volume_metrics_period
            m = compute_symbol_volume_metrics_period("A", sym, start_date, end_date)
            
            curr_sum += m.get("curr") or 0.0
            if m.get("prev") is not None:
                prev_sum += m["prev"] or 0.0
                prev_has = True
            if m.get("yoy") is not None:
                yoy_sum += m["yoy"] or 0.0
                yoy_has = True
            
            leaders.append({"symbol": sym, "name": name, "curr": m.get("curr", 0.0)})
            valid_count += 1
            
        except Exception as e:
            logger.warning(f"计算个股 {sym} 周期指标失败: {e}")
            continue
    
    # 选龙头：按周期内成交量排序取前5
    leaders = sorted(leaders, key=lambda x: x.get("curr", 0.0), reverse=True)[:5]
    
    # 计算同比和环比
    yoy_val = yoy_sum if yoy_has else None
    prev_val = prev_sum if prev_has else None
    yoy_pct = (curr_sum - yoy_val) / yoy_val if (yoy_val and yoy_val > 0) else None
    mom_pct = (curr_sum - prev_val) / prev_val if (prev_val and prev_val > 0) else None
    
    logger.info(f"板块 {industry_name} 周期统计完成，有效成分股: {valid_count}/{len(cons)}")
    
    return {
        "curr": curr_sum, 
        "yoy": yoy_val, 
        "prev": prev_val, 
        "yoy_pct": yoy_pct, 
        "mom_pct": mom_pct, 
        "leaders": leaders, 
        "count": valid_count
    }

def compute_industry_amount_metrics_optimized(industry_name: str, N: int) -> Dict[str, Any]:
    """
    优化的板块成交额统计计算（使用缓存）
    
    Args:
        industry_name: 板块名称
        N: 统计天数
        
    Returns:
        统计结果字典
    """
    # 优先从缓存获取成分股
    cons = get_cached_sector_components(industry_name)
    
    # 如果缓存获取失败，回退到原有方式
    if cons is None or cons.empty:
        logger.warning(f"缓存获取失败，回退到原有方式获取板块 {industry_name} 成分股")
        try:
            from src.ui.app import get_industry_cons
            cons = get_industry_cons(industry_name)
        except Exception as e:
            logger.error(f"回退方式也失败: {e}")
            return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    
    if cons is None or cons.empty:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    
    # 计算统计指标
    curr_sum = 0.0
    prev_sum = 0.0
    yoy_sum = 0.0
    prev_has = False
    yoy_has = False
    
    # 批量处理成分股数据
    valid_count = 0
    for _, row in cons.iterrows():
        sym = str(row.get("symbol", "")).strip()
        name = str(row.get("name", "")).strip()
        if not sym:
            continue
            
        try:
            # 使用原有的个股计算逻辑
            from src.ui.app import compute_symbol_amount_metrics
            m = compute_symbol_amount_metrics("A", sym, N)
            
            curr_sum += m.get("curr") or 0.0
            if m.get("prev") is not None:
                prev_sum += m["prev"] or 0.0
                prev_has = True
            if m.get("yoy") is not None:
                yoy_sum += m["yoy"] or 0.0
                yoy_has = True
            
            valid_count += 1
            
        except Exception as e:
            logger.warning(f"计算个股 {sym} 成交额指标失败: {e}")
            continue
    
    # 计算同比和环比
    yoy_val = yoy_sum if yoy_has else None
    prev_val = prev_sum if prev_has else None
    yoy_pct = (curr_sum - yoy_val) / yoy_val if (yoy_val and yoy_val > 0) else None
    mom_pct = (curr_sum - prev_val) / prev_val if (prev_val and prev_val > 0) else None
    
    logger.info(f"板块 {industry_name} 成交额统计完成，有效成分股: {valid_count}/{len(cons)}")
    
    return {
        "curr": curr_sum, 
        "yoy": yoy_val, 
        "prev": prev_val, 
        "yoy_pct": yoy_pct, 
        "mom_pct": mom_pct
    }

def compute_industry_amount_metrics_period_optimized(industry_name: str, start_date: Any, end_date: Any) -> Dict[str, Any]:
    """
    优化的板块成交额周期统计计算（使用缓存）
    
    Args:
        industry_name: 板块名称
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        统计结果字典
    """
    # 优先从缓存获取成分股
    cons = get_cached_sector_components(industry_name)
    
    # 如果缓存获取失败，回退到原有方式
    if cons is None or cons.empty:
        logger.warning(f"缓存获取失败，回退到原有方式获取板块 {industry_name} 成分股")
        try:
            from src.ui.app import get_industry_cons
            cons = get_industry_cons(industry_name)
        except Exception as e:
            logger.error(f"回退方式也失败: {e}")
            return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    
    if cons is None or cons.empty:
        return {"curr": 0.0, "yoy": None, "mom": None, "prev": None, "yoy_pct": None, "mom_pct": None}
    
    # 计算统计指标
    curr_sum = 0.0
    prev_sum = 0.0
    yoy_sum = 0.0
    prev_has = False
    yoy_has = False
    
    # 批量处理成分股数据
    valid_count = 0
    for _, row in cons.iterrows():
        sym = str(row.get("symbol", "")).strip()
        name = str(row.get("name", "")).strip()
        if not sym:
            continue
            
        try:
            # 使用原有的个股计算逻辑
            from src.ui.app import compute_symbol_amount_metrics_period
            m = compute_symbol_amount_metrics_period("A", sym, start_date, end_date)
            
            curr_sum += m.get("curr") or 0.0
            if m.get("prev") is not None:
                prev_sum += m["prev"] or 0.0
                prev_has = True
            if m.get("yoy") is not None:
                yoy_sum += m["yoy"] or 0.0
                yoy_has = True
            
            valid_count += 1
            
        except Exception as e:
            logger.warning(f"计算个股 {sym} 成交额周期指标失败: {e}")
            continue
    
    # 计算同比和环比
    yoy_val = yoy_sum if yoy_has else None
    prev_val = prev_sum if prev_has else None
    yoy_pct = (curr_sum - yoy_val) / yoy_val if (yoy_val and yoy_val > 0) else None
    mom_pct = (curr_sum - prev_val) / prev_val if (prev_val and prev_val > 0) else None
    
    logger.info(f"板块 {industry_name} 成交额周期统计完成，有效成分股: {valid_count}/{len(cons)}")
    
    return {
        "curr": curr_sum, 
        "yoy": yoy_val, 
        "prev": prev_val, 
        "yoy_pct": yoy_pct, 
        "mom_pct": mom_pct
    }

def compute_industry_agg_series_optimized(industry_name: str, column: str, days: int = 60, start_date: Any = None, end_date: Any = None) -> pd.DataFrame:
    """
    优化版本的板块聚合序列计算
    优先使用缓存的成分股数据
    """
    logger.info(f"开始计算板块 {industry_name} 的 {column} 聚合序列")
    
    # 获取成分股数据（优先使用缓存）
    cons = get_cached_sector_components(industry_name)
    
    if cons is None or cons.empty:
        return pd.DataFrame(columns=["date", column])
    
    try:
        # 直接实现聚合序列计算逻辑，避免循环导入
        from src.data.ak_client import AKDataClient
        import streamlit as st
        
        @st.cache_resource
        def get_client() -> AKDataClient:
            return AKDataClient()
        
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
        
        logger.info(f"板块 {industry_name} 聚合序列计算完成，数据点: {len(s)}")
        return s
        
    except Exception as e:
        logger.error(f"计算板块 {industry_name} 聚合序列失败: {e}")
        return pd.DataFrame(columns=["date", column])

# 缓存装饰器版本的优化函数
@st.cache_data(ttl=1200)
def compute_industry_volume_metrics_cached(industry_name: str, N: int) -> Dict[str, Any]:
    """带缓存的优化板块成交量统计"""
    return compute_industry_volume_metrics_optimized(industry_name, N)

@st.cache_data(ttl=1200)
def compute_industry_volume_metrics_period_cached(industry_name: str, start_date: Any, end_date: Any) -> Dict[str, Any]:
    """带缓存的优化板块成交量周期统计"""
    return compute_industry_volume_metrics_period_optimized(industry_name, start_date, end_date)

@st.cache_data(ttl=1200)
def compute_industry_amount_metrics_cached(industry_name: str, N: int) -> Dict[str, Any]:
    """带缓存的优化板块成交额统计"""
    return compute_industry_amount_metrics_optimized(industry_name, N)

@st.cache_data(ttl=1200)
def compute_industry_amount_metrics_period_cached(industry_name: str, start_date: Any, end_date: Any) -> Dict[str, Any]:
    """带缓存的优化板块成交额周期统计"""
    return compute_industry_amount_metrics_period_optimized(industry_name, start_date, end_date)

@st.cache_data(ttl=1200)
def compute_industry_agg_series_cached(industry_name: str, column: str, days: int = 60, start_date: Any = None, end_date: Any = None) -> pd.DataFrame:
    """带缓存的优化板块聚合序列计算"""
    return compute_industry_agg_series_optimized(industry_name, column, days, start_date, end_date)