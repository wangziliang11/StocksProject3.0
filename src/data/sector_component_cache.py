"""
板块成分股缓存管理器
提供高效的板块成分股信息获取、存储和更新功能
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

from .sector_data_manager import SectorDataManager

logger = logging.getLogger(__name__)

class SectorComponentCache:
    """板块成分股缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache/sector_components"):
        """
        初始化成分股缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存文件路径
        self.cache_file = self.cache_dir / "sector_components.json"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # 初始化sector_data_manager
        self.sector_manager = SectorDataManager()
        
        # 缓存有效期（小时）
        self.cache_validity_hours = 24
        
        # 加载现有缓存
        self._load_cache()
    
    def _load_cache(self):
        """加载现有缓存数据"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache_data = json.load(f)
            else:
                self.cache_data = {}
                
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
                
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            self.cache_data = {}
            self.metadata = {}
    
    def _save_cache(self):
        """保存缓存数据到文件"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
                
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def _is_cache_valid(self, sector_name: str) -> bool:
        """
        检查指定板块的缓存是否有效
        
        Args:
            sector_name: 板块名称
            
        Returns:
            bool: 缓存是否有效
        """
        if sector_name not in self.metadata:
            return False
            
        last_update = self.metadata[sector_name].get('last_update')
        if not last_update:
            return False
            
        try:
            last_update_time = datetime.fromisoformat(last_update)
            current_time = datetime.now()
            
            # 检查是否超过有效期
            if current_time - last_update_time > timedelta(hours=self.cache_validity_hours):
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"检查缓存有效性失败: {e}")
            return False
    
    def get_sector_components(self, sector_name: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        获取板块成分股信息
        
        Args:
            sector_name: 板块名称
            force_refresh: 是否强制刷新缓存
            
        Returns:
            pd.DataFrame: 成分股信息，包含symbol和name列
        """
        # 检查缓存是否有效
        if not force_refresh and self._is_cache_valid(sector_name):
            try:
                components_data = self.cache_data.get(sector_name, [])
                if components_data:
                    df = pd.DataFrame(components_data)
                    logger.info(f"从缓存获取板块 {sector_name} 成分股，共 {len(df)} 只")
                    return df
            except Exception as e:
                logger.warning(f"从缓存读取板块 {sector_name} 成分股失败: {e}")
        
        # 从数据源获取最新数据
        try:
            logger.info(f"正在获取板块 {sector_name} 的最新成分股数据...")
            components_df = self.sector_manager.get_sector_constituents(sector_name, sector_type="auto")
            
            if components_df is not None and not components_df.empty:
                # 标准化列名
                if 'symbol' not in components_df.columns and 'code' in components_df.columns:
                    components_df = components_df.rename(columns={'code': 'symbol'})
                
                # 确保必要的列存在
                required_columns = ['symbol', 'name']
                for col in required_columns:
                    if col not in components_df.columns:
                        components_df[col] = ''
                
                # 只保留必要的列
                components_df = components_df[required_columns].copy()
                
                # 数据清洗
                components_df = components_df.dropna(subset=['symbol'])
                components_df['symbol'] = components_df['symbol'].astype(str).str.strip()
                components_df['name'] = components_df['name'].astype(str).str.strip()
                components_df = components_df[components_df['symbol'] != '']
                
                # 去重
                components_df = components_df.drop_duplicates(subset=['symbol'])
                
                # 更新缓存
                self._update_cache(sector_name, components_df)
                
                logger.info(f"成功获取板块 {sector_name} 成分股，共 {len(components_df)} 只")
                return components_df
            else:
                logger.warning(f"未获取到板块 {sector_name} 的成分股数据")
                return None
                
        except Exception as e:
            logger.error(f"获取板块 {sector_name} 成分股失败: {e}")
            
            # 尝试返回过期的缓存数据
            if sector_name in self.cache_data:
                try:
                    components_data = self.cache_data[sector_name]
                    if components_data:
                        df = pd.DataFrame(components_data)
                        logger.info(f"返回过期缓存数据，板块 {sector_name} 成分股，共 {len(df)} 只")
                        return df
                except Exception:
                    pass
            
            return None
    
    def _update_cache(self, sector_name: str, components_df: pd.DataFrame):
        """
        更新指定板块的缓存数据
        
        Args:
            sector_name: 板块名称
            components_df: 成分股数据
        """
        try:
            # 转换为可序列化的格式
            components_data = components_df.to_dict('records')
            
            # 更新缓存数据
            self.cache_data[sector_name] = components_data
            
            # 更新元数据
            self.metadata[sector_name] = {
                'last_update': datetime.now().isoformat(),
                'component_count': len(components_data),
                'update_source': 'sector_data_manager'
            }
            
            # 保存到文件
            self._save_cache()
            
            logger.info(f"已更新板块 {sector_name} 的缓存，成分股数量: {len(components_data)}")
            
        except Exception as e:
            logger.error(f"更新板块 {sector_name} 缓存失败: {e}")
    
    def update_sector_components(self, sector_name: str) -> bool:
        """
        手动更新指定板块的成分股数据
        
        Args:
            sector_name: 板块名称
            
        Returns:
            bool: 更新是否成功
        """
        try:
            components_df = self.get_sector_components(sector_name, force_refresh=True)
            return components_df is not None and not components_df.empty
        except Exception as e:
            logger.error(f"更新板块 {sector_name} 成分股失败: {e}")
            return False
    
    def update_all_cached_sectors(self) -> Dict[str, bool]:
        """
        更新所有已缓存板块的成分股数据
        
        Returns:
            Dict[str, bool]: 各板块更新结果
        """
        results = {}
        
        for sector_name in list(self.cache_data.keys()):
            try:
                success = self.update_sector_components(sector_name)
                results[sector_name] = success
                logger.info(f"板块 {sector_name} 更新{'成功' if success else '失败'}")
            except Exception as e:
                logger.error(f"更新板块 {sector_name} 时出错: {e}")
                results[sector_name] = False
        
        return results
    
    def get_cached_sectors(self) -> List[str]:
        """
        获取所有已缓存的板块名称
        
        Returns:
            List[str]: 板块名称列表
        """
        return list(self.cache_data.keys())
    
    def get_cache_info(self, sector_name: str) -> Optional[Dict]:
        """
        获取指定板块的缓存信息
        
        Args:
            sector_name: 板块名称
            
        Returns:
            Dict: 缓存信息
        """
        if sector_name not in self.metadata:
            return None
            
        info = self.metadata[sector_name].copy()
        info['is_valid'] = self._is_cache_valid(sector_name)
        info['sector_name'] = sector_name
        
        return info
    
    def clear_cache(self, sector_name: Optional[str] = None):
        """
        清空缓存
        
        Args:
            sector_name: 指定板块名称，如果为None则清空所有缓存
        """
        try:
            if sector_name:
                # 清空指定板块的缓存
                if sector_name in self.cache_data:
                    del self.cache_data[sector_name]
                if sector_name in self.metadata:
                    del self.metadata[sector_name]
                logger.info(f"已清空板块 {sector_name} 的缓存")
            else:
                # 清空所有缓存
                self.cache_data = {}
                self.metadata = {}
                logger.info("已清空所有板块缓存")
            
            self._save_cache()
            
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
    
    def get_cache_statistics(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 缓存统计信息
        """
        total_sectors = len(self.cache_data)
        valid_sectors = sum(1 for sector in self.cache_data.keys() if self._is_cache_valid(sector))
        total_components = sum(len(components) for components in self.cache_data.values())
        
        return {
            'total_sectors': total_sectors,
            'valid_sectors': valid_sectors,
            'expired_sectors': total_sectors - valid_sectors,
            'total_components': total_components,
            'cache_validity_hours': self.cache_validity_hours,
            'cache_dir': str(self.cache_dir)
        }


# 创建全局实例
sector_component_cache = SectorComponentCache()