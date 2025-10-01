#!/usr/bin/env python3
"""
测试修复后的液冷服务器板块成分股数据获取
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.sector_data_manager import SectorDataManager
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_liquid_cooling_sector():
    """测试液冷服务器板块成分股数据获取"""
    print("=" * 60)
    print("测试修复后的液冷服务器板块成分股数据获取")
    print("=" * 60)
    
    # 创建SectorDataManager实例
    manager = SectorDataManager()
    
    # 测试获取液冷服务器概念板块成分股
    sector_name = "液冷服务器"
    print(f"\n正在测试获取 '{sector_name}' 概念板块成分股...")
    
    try:
        # 强制刷新，不使用缓存
        constituents = manager.get_sector_constituents(
            sector_name=sector_name, 
            sector_type="concept", 
            force_refresh=True
        )
        
        if constituents is not None and not constituents.empty:
            print(f"✅ 成功获取 '{sector_name}' 成分股数据！")
            print(f"📊 共获取到 {len(constituents)} 只成分股")
            print("\n前10只成分股：")
            print(constituents.head(10).to_string(index=False))
        else:
            print(f"❌ 未能获取到 '{sector_name}' 成分股数据")
            
    except Exception as e:
        print(f"❌ 获取 '{sector_name}' 成分股时发生错误: {e}")
    
    # 测试其他概念板块
    test_sectors = ["人工智能", "新能源汽车", "芯片概念"]
    
    for test_sector in test_sectors:
        print(f"\n正在测试获取 '{test_sector}' 概念板块成分股...")
        try:
            constituents = manager.get_sector_constituents(
                sector_name=test_sector, 
                sector_type="concept", 
                force_refresh=False  # 使用缓存
            )
            
            if constituents is not None and not constituents.empty:
                print(f"✅ 成功获取 '{test_sector}' 成分股数据，共 {len(constituents)} 只")
            else:
                print(f"❌ 未能获取到 '{test_sector}' 成分股数据")
                
        except Exception as e:
            print(f"❌ 获取 '{test_sector}' 成分股时发生错误: {e}")

if __name__ == "__main__":
    test_liquid_cooling_sector()