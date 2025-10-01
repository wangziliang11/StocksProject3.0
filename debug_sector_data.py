# -*- coding: utf-8 -*-
"""
调试脚本：检查板块成分股数据获取问题
用于诊断 AKShare API 调用和数据检索的根本原因
"""

import pandas as pd
import akshare as ak
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_akshare_functions():
    """测试 AKShare 相关函数的可用性"""
    print("=== 测试 AKShare 函数可用性 ===")
    
    # 检查关键函数是否存在
    functions_to_check = [
        "stock_board_industry_name_ths",
        "stock_board_industry_cons_ths", 
        "stock_board_industry_cons_em",
        "stock_board_concept_cons_em",
        "stock_board_concept_cons_ths",
        "stock_board_concept_name_ths",
        "stock_board_industry_name_url",
        "stock_board_industry_name_em",
        "stock_board_concept_name_em",
        "stock_board_concept_name_url"
    ]
    
    available_functions = []
    for func_name in functions_to_check:
        if hasattr(ak, func_name):
            available_functions.append(func_name)
            print(f"✓ {func_name} - 可用")
        else:
            print(f"✗ {func_name} - 不可用")
    
    return available_functions

def test_industry_name_retrieval():
    """测试行业名称获取"""
    print("\n=== 测试行业名称获取 ===")
    
    # 测试同花顺行业板块名称
    try:
        if hasattr(ak, "stock_board_industry_name_ths"):
            df = ak.stock_board_industry_name_ths()
            print(f"同花顺行业板块数量: {len(df) if df is not None else 0}")
            if df is not None and not df.empty:
                print(f"列名: {list(df.columns)}")
                print("前5个行业:")
                print(df.head())
        else:
            print("stock_board_industry_name_ths 不可用")
    except Exception as e:
        print(f"获取同花顺行业名称失败: {e}")
    
    # 测试东方财富行业名称
    try:
        if hasattr(ak, "stock_board_industry_name_em"):
            df = ak.stock_board_industry_name_em()
            print(f"\n东方财富行业板块数量: {len(df) if df is not None else 0}")
            if df is not None and not df.empty:
                print(f"列名: {list(df.columns)}")
                print("前5个行业:")
                print(df.head())
        else:
            print("stock_board_industry_name_em 不可用")
    except Exception as e:
        print(f"获取东方财富行业名称失败: {e}")

def test_concept_name_retrieval():
    """测试概念名称获取"""
    print("\n=== 测试概念名称获取 ===")
    
    # 测试东方财富概念名称
    try:
        if hasattr(ak, "stock_board_concept_name_em"):
            df = ak.stock_board_concept_name_em()
            print(f"东方财富概念板块数量: {len(df) if df is not None else 0}")
            if df is not None and not df.empty:
                print(f"列名: {list(df.columns)}")
                print("前5个概念:")
                print(df.head())
        else:
            print("stock_board_concept_name_em 不可用")
    except Exception as e:
        print(f"获取东方财富概念名称失败: {e}")

def test_sector_component_retrieval():
    """测试具体板块成分股获取"""
    print("\n=== 测试板块成分股获取 ===")
    
    # 测试几个常见板块
    test_sectors = ["半导体", "人工智能", "电池", "光伏设备", "新能源汽车"]
    
    for sector in test_sectors:
        print(f"\n--- 测试板块: {sector} ---")
        
        # 1. 测试同花顺行业成分股
        try:
            if hasattr(ak, "stock_board_industry_cons_ths"):
                # 先获取行业代码映射
                if hasattr(ak, "stock_board_industry_name_ths"):
                    name_df = ak.stock_board_industry_name_ths()
                    if name_df is not None and not name_df.empty:
                        # 查找匹配的行业代码
                        name_col = None
                        code_col = None
                        for c in ["行业名称", "板块名称", "板块简称", "名称", "name"]:
                            if c in name_df.columns:
                                name_col = c
                                break
                        for c in ["代码", "板块代码", "指数代码", "symbol", "code"]:
                            if c in name_df.columns:
                                code_col = c
                                break
                        
                        if name_col and code_col:
                            matched_rows = name_df[name_df[name_col].str.contains(sector, na=False)]
                            if not matched_rows.empty:
                                sector_code = matched_rows.iloc[0][code_col]
                                print(f"找到行业代码: {sector_code}")
                                
                                # 获取成分股
                                cons_df = ak.stock_board_industry_cons_ths(symbol=sector_code)
                                print(f"同花顺行业成分股数量: {len(cons_df) if cons_df is not None else 0}")
                                if cons_df is not None and not cons_df.empty:
                                    print(f"列名: {list(cons_df.columns)}")
                                    print("前3只股票:")
                                    print(cons_df.head(3))
                            else:
                                print(f"未找到匹配的行业: {sector}")
        except Exception as e:
            print(f"同花顺行业成分股获取失败: {e}")
        
        # 2. 测试东方财富行业成分股
        try:
            if hasattr(ak, "stock_board_industry_cons_em"):
                cons_df = ak.stock_board_industry_cons_em(symbol=sector)
                print(f"东方财富行业成分股数量: {len(cons_df) if cons_df is not None else 0}")
                if cons_df is not None and not cons_df.empty:
                    print(f"列名: {list(cons_df.columns)}")
                    print("前3只股票:")
                    print(cons_df.head(3))
        except Exception as e:
            print(f"东方财富行业成分股获取失败: {e}")
        
        # 3. 测试概念成分股
        try:
            if hasattr(ak, "stock_board_concept_cons_em"):
                cons_df = ak.stock_board_concept_cons_em(symbol=sector)
                print(f"东方财富概念成分股数量: {len(cons_df) if cons_df is not None else 0}")
                if cons_df is not None and not cons_df.empty:
                    print(f"列名: {list(cons_df.columns)}")
                    print("前3只股票:")
                    print(cons_df.head(3))
        except Exception as e:
            print(f"东方财富概念成分股获取失败: {e}")

def test_data_client_integration():
    """测试与数据客户端的集成"""
    print("\n=== 测试数据客户端集成 ===")
    
    try:
        # 导入数据客户端（修正导入路径）
        from src.data.ak_client import AKDataClient
        client = AKDataClient()
        
        # 测试获取单只股票数据
        test_symbol = "000001"  # 平安银行
        print(f"正在获取股票 {test_symbol} 数据...")
        df = client.get_hist(market="A", symbol=test_symbol, period="daily")
        
        if df is not None and not df.empty:
            print(f"✓ 成功获取股票 {test_symbol} 数据，共 {len(df)} 条记录")
            print(f"数据列: {list(df.columns)}")
            
            # 检查关键字段
            required_fields = ["volume", "amount"]
            for field in required_fields:
                if field in df.columns:
                    non_zero_count = (pd.to_numeric(df[field], errors='coerce').fillna(0) > 0).sum()
                    total_count = len(df)
                    print(f"  {field} 字段: 存在，非零记录数: {non_zero_count}/{total_count}")
                else:
                    print(f"  {field} 字段: 缺失")
                    
            # 显示最近几条数据
            print("\n最近5条数据:")
            recent_data = df.tail(5)[['date', 'close', 'volume', 'amount']]
            for _, row in recent_data.iterrows():
                print(f"  {row['date']}: 收盘={row['close']}, 成交量={row['volume']}, 成交额={row['amount']}")
        else:
            print(f"✗ 无法获取股票 {test_symbol} 数据")
            
    except Exception as e:
        print(f"✗ 数据客户端测试失败: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")

def main():
    """主函数"""
    print("开始调试板块成分股数据获取问题...")
    print(f"AKShare 版本: {ak.__version__}")
    
    # 运行所有测试
    available_functions = test_akshare_functions()
    test_industry_name_retrieval()
    test_concept_name_retrieval()
    test_sector_component_retrieval()
    test_data_client_integration()
    
    print("\n=== 调试总结 ===")
    print(f"可用的 AKShare 函数数量: {len(available_functions)}")
    print("建议检查:")
    print("1. AKShare 版本是否为最新")
    print("2. 网络连接是否正常")
    print("3. API 调用频率是否过高")
    print("4. 板块名称是否准确匹配")

if __name__ == "__main__":
    main()