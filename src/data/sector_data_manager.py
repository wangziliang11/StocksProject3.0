"""
板块数据管理器
提供板块成分股数据的获取、缓存和错误处理功能
"""

import pandas as pd
import akshare as ak
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from .ak_client import ak_client
from .backup_sector_data import backup_sector_data

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SectorDataManager:
    """板块数据管理器，提供板块成分股数据的统一接口"""
    
    def __init__(self, cache_dir: str = "data/sector_cache"):
        """
        初始化板块数据管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存文件路径
        self.industry_cache_file = self.cache_dir / "industry_constituents.json"
        self.concept_cache_file = self.cache_dir / "concept_constituents.json"
        self.sector_list_cache = self.cache_dir / "sector_lists.json"
        
        # 缓存过期时间（天）
        self.cache_expire_days = 7
        
    def _is_cache_expired(self, file_path: Path) -> bool:
        """检查缓存是否过期"""
        if not file_path.exists():
            return True
        
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            return (datetime.now() - mtime) > timedelta(days=self.cache_expire_days)
        except Exception:
            return True
    
    def _load_cache(self, file_path: Path) -> Optional[Dict]:
        """加载缓存数据"""
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载缓存失败 {file_path}: {e}")
            return None
    
    def _save_cache(self, data: Dict, file_path: Path) -> None:
        """保存缓存数据"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"缓存已保存: {file_path}")
        except Exception as e:
            logger.error(f"保存缓存失败 {file_path}: {e}")
    
    def _fetch_industry_list_with_retry(self, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        获取行业列表，带重试机制
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            行业列表DataFrame或None
        """
        functions_to_try = [
            ("stock_board_industry_name_ths", "同花顺行业列表"),
            ("stock_board_industry_name_em", "东方财富行业列表"),
        ]
        
        for func_name, desc in functions_to_try:
            for attempt in range(max_retries):
                try:
                    if hasattr(ak, func_name):
                        func = getattr(ak, func_name)
                        df = func()
                        if df is not None and not df.empty:
                            logger.info(f"成功获取{desc}，共{len(df)}条记录")
                            return df
                    else:
                        logger.warning(f"函数 {func_name} 不存在")
                        break
                        
                except Exception as e:
                    logger.warning(f"获取{desc}失败 (尝试{attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2 ** attempt)  # 指数退避
        
        return None
    
    def _fetch_concept_list_with_retry(self, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        获取概念列表，带重试机制
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            概念列表DataFrame或None
        """
        functions_to_try = [
            ("stock_board_concept_name_ths", "同花顺概念列表"),
            ("stock_board_concept_name_em", "东方财富概念列表"),
        ]
        
        for func_name, desc in functions_to_try:
            for attempt in range(max_retries):
                try:
                    if hasattr(ak, func_name):
                        func = getattr(ak, func_name)
                        df = func()
                        if df is not None and not df.empty:
                            logger.info(f"成功获取{desc}，共{len(df)}条记录")
                            return df
                    else:
                        logger.warning(f"函数 {func_name} 不存在")
                        break
                        
                except Exception as e:
                    logger.warning(f"获取{desc}失败 (尝试{attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2 ** attempt)  # 指数退避
        
        return None
    
    def _fetch_sector_constituents_with_retry(self, sector_name: str, sector_type: str = "industry", max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        获取板块成分股，带重试机制
        
        Args:
            sector_name: 板块名称
            sector_type: 板块类型 ("industry" 或 "concept")
            max_retries: 最大重试次数
            
        Returns:
            成分股DataFrame或None
        """
        if sector_type == "industry":
            functions_to_try = [
                ("stock_board_industry_cons_em", "东方财富行业成分股"),
            ]
        else:  # concept
            functions_to_try = [
                ("stock_board_concept_cons_em", "东方财富概念成分股"),
            ]
        
        for func_name, desc in functions_to_try:
            for attempt in range(max_retries):
                try:
                    if hasattr(ak, func_name):
                        func = getattr(ak, func_name)
                        df = func(symbol=sector_name)
                        if df is not None and not df.empty:
                            logger.info(f"成功获取{desc} '{sector_name}'，共{len(df)}条记录")
                            return self._standardize_constituents_df(df)
                        else:
                            logger.warning(f"{desc} '{sector_name}' 返回空数据")
                    else:
                        logger.warning(f"函数 {func_name} 不存在")
                        break
                        
                except Exception as e:
                    error_msg = str(e)
                    # 特殊处理网络连接错误
                    if "ProxyError" in error_msg or "ConnectionError" in error_msg or "HTTPSConnectionPool" in error_msg:
                        logger.warning(f"网络连接问题，获取{desc} '{sector_name}' 失败 (尝试{attempt+1}/{max_retries}): 网络连接超时或代理问题")
                    else:
                        logger.warning(f"获取{desc} '{sector_name}' 失败 (尝试{attempt+1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        import time
                        # 网络错误时使用更长的等待时间
                        wait_time = 5 if "ProxyError" in error_msg or "ConnectionError" in error_msg else 2 ** attempt
                        time.sleep(wait_time)
        
        return None
    
    def _standardize_constituents_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化成分股DataFrame格式
        
        Args:
            df: 原始成分股DataFrame
            
        Returns:
            标准化后的DataFrame，包含 'code' 和 'name' 列
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=['code', 'name'])
        
        # 尝试识别代码和名称列
        code_cols = ['代码', 'code', '股票代码', 'symbol']
        name_cols = ['名称', 'name', '股票名称', '股票简称']
        
        code_col = None
        name_col = None
        
        for col in code_cols:
            if col in df.columns:
                code_col = col
                break
        
        for col in name_cols:
            if col in df.columns:
                name_col = col
                break
        
        if code_col is None or name_col is None:
            logger.warning(f"无法识别代码或名称列，可用列: {list(df.columns)}")
            return pd.DataFrame(columns=['code', 'name'])
        
        # 创建标准化DataFrame
        result = pd.DataFrame()
        result['code'] = df[code_col].astype(str)
        result['name'] = df[name_col].astype(str)
        
        # 清理代码格式，提取6位数字
        result['code'] = result['code'].str.extract(r'(\d{6})')[0]
        result = result.dropna(subset=['code'])
        
        return result
    
    def get_sector_lists(self, force_refresh: bool = False) -> Dict[str, List[str]]:
        """
        获取板块列表（行业和概念）
        
        Args:
            force_refresh: 是否强制刷新缓存
            
        Returns:
            包含 'industries' 和 'concepts' 键的字典
        """
        # 检查缓存
        if not force_refresh and not self._is_cache_expired(self.sector_list_cache):
            cached_data = self._load_cache(self.sector_list_cache)
            if cached_data:
                logger.info("使用缓存的板块列表")
                return cached_data
        
        logger.info("获取最新板块列表...")
        
        # 获取行业列表
        industries = []
        industry_df = self._fetch_industry_list_with_retry()
        if industry_df is not None:
            # 尝试找到名称列
            name_cols = ['名称', 'name', '行业名称']
            for col in name_cols:
                if col in industry_df.columns:
                    industries = industry_df[col].dropna().unique().tolist()
                    break
        
        # 获取概念列表
        concepts = []
        concept_df = self._fetch_concept_list_with_retry()
        if concept_df is not None:
            # 尝试找到名称列
            name_cols = ['名称', 'name', '概念名称']
            for col in name_cols:
                if col in concept_df.columns:
                    concepts = concept_df[col].dropna().unique().tolist()
                    break
        
        result = {
            'industries': industries,
            'concepts': concepts,
            'last_updated': datetime.now().isoformat()
        }
        
        # 保存缓存
        self._save_cache(result, self.sector_list_cache)
        
        logger.info(f"获取到 {len(industries)} 个行业，{len(concepts)} 个概念")
        return result
    
    def get_sector_constituents(self, sector_name: str, sector_type: str = "auto", force_refresh: bool = False) -> pd.DataFrame:
        """
        获取板块成分股
        
        Args:
            sector_name: 板块名称
            sector_type: 板块类型 ("industry", "concept", "auto")
            force_refresh: 是否强制刷新缓存
            
        Returns:
            成分股DataFrame，包含 'code' 和 'name' 列
        """
        # 选择缓存文件
        if sector_type == "industry":
            cache_file = self.industry_cache_file
        elif sector_type == "concept":
            cache_file = self.concept_cache_file
        else:  # auto
            # 先尝试行业，再尝试概念
            result = self.get_sector_constituents(sector_name, "industry", force_refresh)
            if not result.empty:
                return result
            return self.get_sector_constituents(sector_name, "concept", force_refresh)
        
        # 检查缓存
        cache_key = f"{sector_type}_{sector_name}"
        if not force_refresh and not self._is_cache_expired(cache_file):
            cached_data = self._load_cache(cache_file)
            if cached_data and cache_key in cached_data:
                logger.info(f"使用缓存的{sector_type}成分股: {sector_name}")
                df_data = cached_data[cache_key]
                return pd.DataFrame(df_data)
        
        logger.info(f"获取{sector_type}成分股: {sector_name}")
        
        # 获取成分股数据
        df = self._fetch_sector_constituents_with_retry(sector_name, sector_type)
        
        if df is None or df.empty:
            logger.warning(f"无法获取{sector_type}成分股: {sector_name}")
            # 尝试从备用数据源获取
            logger.info(f"尝试从备用数据源获取 {sector_name} 成分股")
            fallback_df = self.get_fallback_constituents(sector_name)
            if not fallback_df.empty:
                logger.info(f"成功从备用数据源获取到 {sector_name} 成分股，共 {len(fallback_df)} 只股票")
                # 更新缓存
                cached_data = self._load_cache(cache_file) or {}
                cached_data[cache_key] = fallback_df.to_dict('records')
                self._save_cache(cached_data, cache_file)
                return fallback_df
            else:
                logger.warning(f"备用数据源中也没有找到 {sector_name} 的成分股")
                return pd.DataFrame(columns=['code', 'name'])
        
        # 更新缓存
        cached_data = self._load_cache(cache_file) or {}
        cached_data[cache_key] = df.to_dict('records')
        cached_data['last_updated'] = datetime.now().isoformat()
        self._save_cache(cached_data, cache_file)
        
        return df
    
    def get_fallback_constituents(self, sector_name: str) -> pd.DataFrame:
        """
        获取备用成分股数据（从预定义列表）
        
        Args:
            sector_name: 板块名称
            
        Returns:
            成分股DataFrame
        """
        logger.info(f"尝试从备用数据源获取: {sector_name}")
        
        # 首先尝试从专用备用数据源获取
        industry_df = backup_sector_data.get_industry_constituents(sector_name)
        if industry_df is not None and not industry_df.empty:
            # 标准化列名
            result_df = industry_df.copy()
            if 'symbol' in result_df.columns:
                result_df['code'] = result_df['symbol']
            if 'name' in result_df.columns and 'code' not in result_df.columns:
                result_df['code'] = result_df['代码'] if '代码' in result_df.columns else ''
            return result_df[['code', 'name']] if 'name' in result_df.columns else result_df
        
        concept_df = backup_sector_data.get_concept_constituents(sector_name)
        if concept_df is not None and not concept_df.empty:
            # 标准化列名
            result_df = concept_df.copy()
            if 'symbol' in result_df.columns:
                result_df['code'] = result_df['symbol']
            if 'name' in result_df.columns and 'code' not in result_df.columns:
                result_df['code'] = result_df['代码'] if '代码' in result_df.columns else ''
            return result_df[['code', 'name']] if 'name' in result_df.columns else result_df
        
        # 如果专用备用数据源没有，使用原有的fallback数据
        fallback_data = {
            "新能源汽车": [
                {"code": "002594", "name": "比亚迪"},
                {"code": "300750", "name": "宁德时代"},
                {"code": "002460", "name": "赣锋锂业"},
                {"code": "300014", "name": "亿纬锂能"},
                {"code": "002812", "name": "恩捷股份"},
            ],
            "人工智能": [
                {"code": "000063", "name": "中兴通讯"},
                {"code": "002415", "name": "海康威视"},
                {"code": "300059", "name": "东方财富"},
                {"code": "002230", "name": "科大讯飞"},
                {"code": "000725", "name": "京东方A"},
            ],
            "半导体": [
                {"code": "000725", "name": "京东方A"},
                {"code": "002415", "name": "海康威视"},
                {"code": "300782", "name": "卓胜微"},
                {"code": "688981", "name": "中芯国际"},
                {"code": "002049", "name": "紫光国微"},
            ]
        }
        
        if sector_name in fallback_data:
            logger.info(f"使用备用数据: {sector_name}")
            return pd.DataFrame(fallback_data[sector_name])
        
        return pd.DataFrame(columns=['code', 'name'])
    
    def clear_cache(self) -> None:
        """清空所有缓存"""
        cache_files = [
            self.industry_cache_file,
            self.concept_cache_file,
            self.sector_list_cache
        ]
        
        for cache_file in cache_files:
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    logger.info(f"已清空缓存: {cache_file}")
                except Exception as e:
                    logger.error(f"清空缓存失败 {cache_file}: {e}")