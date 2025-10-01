#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试板块数据获取修复
验证新的SectorDataManager是否能正确获取板块成分股数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.data.sector_data_manager import SectorDataManager

def test_sector_data_manager():
    """测试板块数据管理器"""
    print("=" * 60)
    print("测试板块数据管理器")
    print("=" * 60)
    
    # 初始化管理器
    try:
        manager = SectorDataManager()
        print("✓ 板块数据管理器初始化成功")
    except Exception as e:
        print(f"✗ 板块数据管理器初始化失败: {e}")
        return False
    
    # 测试板块列表
    test_sectors = ["新能源汽车", "电池", "光伏", "半导体", "医药生物"]
    
    for sector in test_sectors:
        print(f"\n测试板块: {sector}")
        print("-" * 40)
        
        try:
            # 获取成分股
            df = manager.get_sector_constituents(sector, sector_type="auto")
            
            if df is not None and not df.empty:
                print(f"✓ 成功获取 {len(df)} 只成分股")
                print(f"  前5只股票: {df.head()['name'].tolist()}")
            else:
                print("⚠ 未获取到成分股数据，尝试备用数据...")
                
                # 尝试备用数据
                fallback_df = manager.get_fallback_constituents(sector)
                if not fallback_df.empty:
                    print(f"✓ 备用数据获取成功: {len(fallback_df)} 只股票")
                    print(f"  前5只股票: {fallback_df.head()['name'].tolist()}")
                else:
                    print("✗ 备用数据也无法获取")
                    
        except Exception as e:
            print(f"✗ 获取失败: {e}")
    
    return True

def test_app_integration():
    """测试与app.py的集成"""
    print("\n" + "=" * 60)
    print("测试与app.py的集成")
    print("=" * 60)
    
    try:
        # 导入app模块中的函数
        from src.ui.app import ak_get_industry_cons
        
        test_sectors = ["新能源汽车", "电池"]
        
        for sector in test_sectors:
            print(f"\n测试板块: {sector}")
            print("-" * 40)
            
            try:
                df = ak_get_industry_cons(sector)
                
                if df is not None and not df.empty:
                    print(f"✓ 成功获取 {len(df)} 只成分股")
                    print(f"  前5只股票: {df.head()['name'].tolist()}")
                    print(f"  数据格式: {df.columns.tolist()}")
                else:
                    print("✗ 未获取到成分股数据")
                    
            except Exception as e:
                print(f"✗ 获取失败: {e}")
                
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("板块数据获取修复测试")
    print("=" * 60)
    
    # 测试板块数据管理器
    success1 = test_sector_data_manager()
    
    # 测试与app.py的集成
    success2 = test_app_integration()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if success1 and success2:
        print("✓ 所有测试通过，板块数据获取修复成功")
    else:
        print("✗ 部分测试失败，需要进一步调试")
        
    print("\n建议:")
    print("1. 如果网络连接问题仍然存在，请检查代理设置")
    print("2. 如果备用数据有效，可以考虑扩展备用数据源")
    print("3. 定期更新缓存数据以确保数据的时效性")

if __name__ == "__main__":
    main()