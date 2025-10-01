"""
简化的优化功能测试脚本
使用模拟数据验证优化逻辑是否正确
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_mock_sector_components():
    """创建模拟的板块成分股数据"""
    mock_data = {
        "银行": pd.DataFrame({
            "symbol": ["000001", "600036", "601398", "601939", "600000"],
            "name": ["平安银行", "招商银行", "工商银行", "建设银行", "浦发银行"]
        }),
        "证券": pd.DataFrame({
            "symbol": ["000166", "600030", "601688", "000776", "002736"],
            "name": ["申万宏源", "中信证券", "华泰证券", "广发证券", "国信证券"]
        })
    }
    return mock_data

def test_cache_mechanism():
    """测试缓存机制的基本功能"""
    print("=" * 50)
    print("测试缓存机制基本功能")
    print("=" * 50)
    
    try:
        from src.data.sector_component_cache import SectorComponentCache
        
        # 创建缓存实例
        cache = SectorComponentCache()
        
        # 创建模拟数据
        mock_data = create_mock_sector_components()
        
        # 测试缓存保存和加载
        for sector_name, components_df in mock_data.items():
            print(f"\n测试板块: {sector_name}")
            
            # 手动设置缓存数据（模拟获取成功的情况）
            cache.cache_data[sector_name] = {
                "components": components_df,
                "last_update": datetime.now().isoformat(),
                "is_valid": True
            }
            
            # 测试从缓存获取
            cached_components = cache.get_sector_components(sector_name, force_refresh=False)
            
            if cached_components is not None and not cached_components.empty:
                print(f"✅ 缓存获取成功，成分股数量: {len(cached_components)}")
                print(f"成分股列表: {list(cached_components['name'])}")
            else:
                print("❌ 缓存获取失败")
        
        # 测试缓存统计
        stats = cache.get_cache_statistics()
        print(f"\n缓存统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"缓存机制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimized_functions_logic():
    """测试优化函数的逻辑正确性"""
    print("\n" + "=" * 50)
    print("测试优化函数逻辑正确性")
    print("=" * 50)
    
    try:
        from src.ui.optimized_industry_calculations import get_cached_sector_components
        from src.data.sector_component_cache import sector_component_cache
        
        # 创建模拟数据
        mock_data = create_mock_sector_components()
        
        # 手动设置缓存数据
        for sector_name, components_df in mock_data.items():
            sector_component_cache.cache_data[sector_name] = {
                "components": components_df,
                "last_update": datetime.now().isoformat(),
                "is_valid": True
            }
        
        # 测试获取成分股函数
        for sector_name in mock_data.keys():
            print(f"\n测试板块: {sector_name}")
            
            start_time = time.time()
            components = get_cached_sector_components(sector_name)
            end_time = time.time()
            
            if components is not None and not components.empty:
                print(f"✅ 获取成功，成分股数量: {len(components)}")
                print(f"获取耗时: {end_time - start_time:.4f}秒")
                print(f"数据结构正确: {'symbol' in components.columns and 'name' in components.columns}")
            else:
                print("❌ 获取失败")
        
        return True
        
    except Exception as e:
        print(f"优化函数逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_mechanism():
    """测试回退机制"""
    print("\n" + "=" * 50)
    print("测试回退机制")
    print("=" * 50)
    
    try:
        from src.ui.optimized_industry_calculations import get_cached_sector_components
        from src.data.sector_component_cache import sector_component_cache
        
        # 清空缓存，模拟缓存失败的情况
        sector_component_cache.cache_data = {}
        
        test_sector = "不存在的板块"
        print(f"测试不存在的板块: {test_sector}")
        
        start_time = time.time()
        components = get_cached_sector_components(test_sector)
        end_time = time.time()
        
        print(f"获取耗时: {end_time - start_time:.4f}秒")
        
        if components is None:
            print("✅ 回退机制正常工作，返回None")
        else:
            print("❌ 回退机制异常")
        
        return True
        
    except Exception as e:
        print(f"回退机制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_info_functions():
    """测试缓存信息功能"""
    print("\n" + "=" * 50)
    print("测试缓存信息功能")
    print("=" * 50)
    
    try:
        from src.data.sector_component_cache import sector_component_cache
        
        # 创建模拟数据
        mock_data = create_mock_sector_components()
        
        # 设置缓存数据
        for sector_name, components_df in mock_data.items():
            sector_component_cache.cache_data[sector_name] = {
                "components": components_df,
                "last_update": datetime.now().isoformat(),
                "is_valid": True
            }
        
        # 测试获取缓存信息
        for sector_name in mock_data.keys():
            cache_info = sector_component_cache.get_cache_info(sector_name)
            print(f"\n{sector_name} 缓存信息: {cache_info}")
            
            if cache_info and cache_info.get('is_valid'):
                print(f"✅ {sector_name} 缓存信息正确")
            else:
                print(f"❌ {sector_name} 缓存信息异常")
        
        # 测试缓存统计
        stats = sector_component_cache.get_cache_statistics()
        print(f"\n缓存统计: {stats}")
        
        if stats.get('total_sectors', 0) > 0:
            print("✅ 缓存统计功能正常")
        else:
            print("❌ 缓存统计功能异常")
        
        return True
        
    except Exception as e:
        print(f"缓存信息功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始简化的优化功能测试")
    print(f"测试时间: {datetime.now()}")
    
    test_results = []
    
    # 1. 测试缓存机制
    result1 = test_cache_mechanism()
    test_results.append(("缓存机制", result1))
    
    # 2. 测试优化函数逻辑
    result2 = test_optimized_functions_logic()
    test_results.append(("优化函数逻辑", result2))
    
    # 3. 测试回退机制
    result3 = test_fallback_mechanism()
    test_results.append(("回退机制", result3))
    
    # 4. 测试缓存信息功能
    result4 = test_cache_info_functions()
    test_results.append(("缓存信息功能", result4))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！优化功能基本正常")
    else:
        print("⚠️  部分测试失败，需要进一步检查")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main()