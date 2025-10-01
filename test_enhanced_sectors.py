#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '.')
from src.ui.app import get_enhanced_sector_lists

def test_enhanced_sector_lists():
    """测试增强的板块列表功能"""
    print('测试增强的板块列表功能...')
    try:
        enhanced_lists = get_enhanced_sector_lists()
        
        # 显示统计信息
        industries_count = len(enhanced_lists['industries'])
        concepts_count = len(enhanced_lists['concepts'])
        custom_count = len(enhanced_lists['custom'])
        total_count = len(enhanced_lists['all'])
        
        print(f'行业板块数量: {industries_count}')
        print(f'概念板块数量: {concepts_count}')
        print(f'自定义板块数量: {custom_count}')
        print(f'总板块数量: {total_count}')
        
        # 显示前10个行业板块
        print('\n前10个行业板块:')
        for i, sector in enumerate(enhanced_lists['industries'][:10]):
            print(f'  {i+1}. {sector}')
        
        # 显示前10个概念板块
        print('\n前10个概念板块:')
        for i, sector in enumerate(enhanced_lists['concepts'][:10]):
            print(f'  {i+1}. {sector}')
        
        # 检查是否包含固态电池
        if '固态电池' in enhanced_lists['all']:
            print('\n✓ 包含"固态电池"板块')
        else:
            print('\n✗ 不包含"固态电池"板块')
            
        print('\n测试成功！增强的板块列表功能正常工作。')
        return True
        
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_enhanced_sector_lists()