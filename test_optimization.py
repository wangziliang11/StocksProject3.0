"""
测试优化后的板块统计计算功能
验证缓存机制是否正常工作，计算结果是否准确
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_sector_component_cache():
    """测试板块成分股缓存功能"""
    print("=" * 50)
    print("测试板块成分股缓存功能")
    print("=" * 50)
    
    try:
        from src.data.sector_component_cache import sector_component_cache
        
        # 测试板块列表
        test_sectors = ["银行", "证券", "保险", "房地产", "钢铁"]
        
        for sector in test_sectors:
            print(f"\n测试板块: {sector}")
            
            # 获取缓存信息
            cache_info = sector_component_cache.get_cache_info(sector)
            print(f"缓存信息: {cache_info}")
            
            # 获取成分股（不强制刷新）
            start_time = time.time()
            components = sector_component_cache.get_sector_components(sector, force_refresh=False)
            end_time = time.time()
            
            if components is not None and not components.empty:
                print(f"成分股数量: {len(components)}")
                print(f"获取耗时: {end_time - start_time:.2f}秒")
                print(f"前5只成分股:")
                print(components.head())
            else:
                print("未获取到成分股数据")
                
                # 尝试强制刷新
                print("尝试强制刷新...")
                start_time = time.time()
                components = sector_component_cache.get_sector_components(sector, force_refresh=True)
                end_time = time.time()
                
                if components is not None and not components.empty:
                    print(f"强制刷新后成分股数量: {len(components)}")
                    print(f"刷新耗时: {end_time - start_time:.2f}秒")
                else:
                    print("强制刷新后仍未获取到数据")
        
        # 获取缓存统计
        stats = sector_component_cache.get_cache_stats()
        print(f"\n缓存统计: {stats}")
        
    except Exception as e:
        print(f"测试板块成分股缓存失败: {e}")
        import traceback
        traceback.print_exc()

def test_optimized_calculations():
    """测试优化后的计算函数"""
    print("\n" + "=" * 50)
    print("测试优化后的计算函数")
    print("=" * 50)
    
    try:
        from src.ui.optimized_industry_calculations import (
            compute_industry_volume_metrics_optimized,
            compute_industry_volume_metrics_period_optimized,
            compute_industry_amount_metrics_optimized,
            compute_industry_amount_metrics_period_optimized,
            get_cached_sector_components
        )
        
        # 测试板块
        test_sector = "银行"
        N = 20
        
        print(f"\n测试板块: {test_sector}")
        
        # 测试获取成分股
        print("\n1. 测试获取成分股")
        start_time = time.time()
        components = get_cached_sector_components(test_sector)
        end_time = time.time()
        
        if components is not None and not components.empty:
            print(f"成分股数量: {len(components)}")
            print(f"获取耗时: {end_time - start_time:.2f}秒")
        else:
            print("未获取到成分股数据")
            return
        
        # 测试成交量统计（近N日）
        print(f"\n2. 测试成交量统计（近{N}日）")
        start_time = time.time()
        volume_metrics = compute_industry_volume_metrics_optimized(test_sector, N)
        end_time = time.time()
        
        print(f"计算耗时: {end_time - start_time:.2f}秒")
        print(f"统计结果: {volume_metrics}")
        
        # 测试成交量统计（周期）
        print(f"\n3. 测试成交量统计（周期）")
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        start_time = time.time()
        volume_period_metrics = compute_industry_volume_metrics_period_optimized(test_sector, start_date, end_date)
        end_time = time.time()
        
        print(f"计算耗时: {end_time - start_time:.2f}秒")
        print(f"统计结果: {volume_period_metrics}")
        
        # 测试成交额统计（近N日）
        print(f"\n4. 测试成交额统计（近{N}日）")
        start_time = time.time()
        amount_metrics = compute_industry_amount_metrics_optimized(test_sector, N)
        end_time = time.time()
        
        print(f"计算耗时: {end_time - start_time:.2f}秒")
        print(f"统计结果: {amount_metrics}")
        
        # 测试成交额统计（周期）
        print(f"\n5. 测试成交额统计（周期）")
        start_time = time.time()
        amount_period_metrics = compute_industry_amount_metrics_period_optimized(test_sector, start_date, end_date)
        end_time = time.time()
        
        print(f"计算耗时: {end_time - start_time:.2f}秒")
        print(f"统计结果: {amount_period_metrics}")
        
    except Exception as e:
        print(f"测试优化后的计算函数失败: {e}")
        import traceback
        traceback.print_exc()

def test_performance_comparison():
    """性能对比测试"""
    print("\n" + "=" * 50)
    print("性能对比测试")
    print("=" * 50)
    
    try:
        # 导入原始函数和优化函数
        from src.ui.app import (
            compute_industry_volume_metrics,
            compute_industry_amount_metrics
        )
        from src.ui.optimized_industry_calculations import (
            compute_industry_volume_metrics_optimized,
            compute_industry_amount_metrics_optimized
        )
        
        test_sector = "银行"
        N = 20
        
        print(f"测试板块: {test_sector}, N={N}")
        
        # 测试原始成交量计算
        print("\n原始成交量计算:")
        start_time = time.time()
        try:
            original_volume = compute_industry_volume_metrics(test_sector, N)
            original_volume_time = time.time() - start_time
            print(f"耗时: {original_volume_time:.2f}秒")
            print(f"结果: {original_volume}")
        except Exception as e:
            print(f"原始计算失败: {e}")
            original_volume_time = None
            original_volume = None
        
        # 测试优化成交量计算
        print("\n优化成交量计算:")
        start_time = time.time()
        try:
            optimized_volume = compute_industry_volume_metrics_optimized(test_sector, N)
            optimized_volume_time = time.time() - start_time
            print(f"耗时: {optimized_volume_time:.2f}秒")
            print(f"结果: {optimized_volume}")
        except Exception as e:
            print(f"优化计算失败: {e}")
            optimized_volume_time = None
            optimized_volume = None
        
        # 性能对比
        if original_volume_time and optimized_volume_time:
            improvement = (original_volume_time - optimized_volume_time) / original_volume_time * 100
            print(f"\n性能提升: {improvement:.1f}%")
            if improvement > 0:
                print("✅ 优化版本更快")
            else:
                print("❌ 优化版本较慢")
        
        # 测试原始成交额计算
        print("\n原始成交额计算:")
        start_time = time.time()
        try:
            original_amount = compute_industry_amount_metrics(test_sector, N)
            original_amount_time = time.time() - start_time
            print(f"耗时: {original_amount_time:.2f}秒")
            print(f"结果: {original_amount}")
        except Exception as e:
            print(f"原始计算失败: {e}")
            original_amount_time = None
            original_amount = None
        
        # 测试优化成交额计算
        print("\n优化成交额计算:")
        start_time = time.time()
        try:
            optimized_amount = compute_industry_amount_metrics_optimized(test_sector, N)
            optimized_amount_time = time.time() - start_time
            print(f"耗时: {optimized_amount_time:.2f}秒")
            print(f"结果: {optimized_amount}")
        except Exception as e:
            print(f"优化计算失败: {e}")
            optimized_amount_time = None
            optimized_amount = None
        
        # 性能对比
        if original_amount_time and optimized_amount_time:
            improvement = (original_amount_time - optimized_amount_time) / original_amount_time * 100
            print(f"\n成交额性能提升: {improvement:.1f}%")
            if improvement > 0:
                print("✅ 优化版本更快")
            else:
                print("❌ 优化版本较慢")
        
    except Exception as e:
        print(f"性能对比测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("开始测试优化后的板块统计计算功能")
    print(f"测试时间: {datetime.now()}")
    
    # 1. 测试板块成分股缓存
    test_sector_component_cache()
    
    # 2. 测试优化后的计算函数
    test_optimized_calculations()
    
    # 3. 性能对比测试
    test_performance_comparison()
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main()