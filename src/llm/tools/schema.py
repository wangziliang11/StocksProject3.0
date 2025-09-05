import json
from typing import List, Dict, Any

# 工具：查询个股基本信息（A股 / 港股）
# 说明：作为 LLM function calling 的工具定义（OpenAI tools 格式）

def get_tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch_stock_info_a",
                "description": "获取A股个股基本信息，返回表格数据（东方财富源）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "A股股票代码，例如 600519"}
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_stock_info_hk",
                "description": "获取港股个股实时与基础信息（尽可能包含所属行业），返回表格数据（东方财富/新浪源）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "港股股票代码，例如 00700、00005"}
                    },
                    "required": ["symbol"]
                }
            }
        }
    ]