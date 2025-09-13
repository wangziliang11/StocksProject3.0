# A/H 股票分析系统（Streamlit + AKShare + LLM）

一个面向 A 股与港股的量化分析与研报辅助工具，支持：
- 数据获取与持久化缓存：AKShare 拉取历史行情，按市场/代码/复权分桶缓存为 CSV，非日线周期统一从日线重采样
- 技术指标与简易回测：SMA/EMA/MACD、双均线金叉/死叉策略的基础回测统计
- 可视化：Plotly K 线 + 成交量，叠加常用均线
- LLM 多提供商与路由：通过 providers/models/routing 配置统一管理模型，支持 OpenAI 兼容接口
- LLM 工具调用（Function Calling）：示例工具“获取 A 股个股基本信息”，可扩展更多联网工具
- 前后端分层结构：数据层、业务逻辑层、LLM 层、可视化层、UI 层彻底隔离


## 目录结构

```
.
├── data/                      # 本地数据目录
│   └── cache/                 # 日线缓存：data/cache/{A|H}/{symbol}/daily_{qfq|hfq|none}.csv
├── secrets/                   # 本地敏感信息（不提交）
├── models.yaml                # 公共模型与提供商配置（可提交）
├── models.local.yaml          # 本地敏感配置（api_key/base_url 覆盖，已忽略）
├── routing.yaml               # 公共路由配置（模型注册与路由）
├── routing.local.yaml         # 本地路由覆盖（已忽略）
├── requirements.txt
└── src/
    ├── data/
    │   └── ak_client.py       # AKDataClient：数据获取 + 本地缓存 + 重采样
    ├── logic/
    │   └── indicators.py      # 指标与回测逻辑
    ├── llm/
    │   ├── mcp/
    │   │   └── adapter.py     # MCPRouter（适配层，占位实现）
    │   └── tools/
    │       ├── schema.py      # Tools Schema（OpenAI tools 规范）
    │       └── executor.py    # Tools 执行器（应用端实际调用）
    ├── viz/
    │   └── charts.py          # K 线与成交量可视化
    └── ui/
        └── app.py             # Streamlit 前端入口
```


## 快速开始

1) 安装依赖（建议使用虚拟环境）

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) 配置大模型

- 填写 models.local.yaml（不会提交到 Git），示例：

```yaml
version: 1
providers:
  openai_compat:
    api_key: "sk-..."         # 建议从环境读取，这里示例直填
    base_url: "https://api.openai.com"  # 可按需覆盖
  dashscope:
    api_key: "your-dashscope-key"
  deepseek:
    api_key: "your-deepseek-key"
# 可按需添加 routing 覆盖
routing:
  default:
    provider: openai_compat
    model: gpt-4o-mini
```

- 公共 providers/models/routing 配置在 models.yaml
- 路由注册建议放到 routing.yaml，支持按文件合并与本地覆盖（见下文）

3) 运行前端

```
streamlit run src/ui/app.py --server.port 8501
```

打开浏览器访问 http://localhost:8501/ 即可。


## 配置说明

本项目通过“公共配置 + 本地覆盖”实现灵活的多模型路由：

- models.yaml（可提交）定义：
  - providers：提供商基础信息（type/base_url）
  - models：模型名称与归属提供商
  - routing：默认路由（也可放在 routing.yaml）
- models.local.yaml（忽略提交）用于：
  - providers 的 api_key、本地 base_url 覆盖
  - routing 的本地覆盖（可选）
- routing.yaml / routing.local.yaml：
  - 路由注册文件（公共/本地），最终合并优先级：
    models.yaml.routing < routing.yaml < models.local.yaml.routing < routing.local.yaml

在代码侧，已切换为基于 MCP 的调用路径，由 <mcfile name="adapter.py" path="src/llm/mcp/adapter.py"></mcfile> 提供 <mcsymbol name="MCPRouter" filename="adapter.py" path="src/llm/mcp/adapter.py" startline="6" type="class"></mcsymbol> 统一发起对话。


## 数据持久化与重采样

- AKDataClient 统一以“日线”为基准，所有周/月/季/年均由日线重采样得到，以减少对外请求、统一数据口径。
- 缓存位置：data/cache/{market}/{symbol}/daily_{adj}.csv
  - market: A 或 H
  - symbol: 股票代码（如 600519、00700）
  - adj: qfq/hfq/none（仅 A 股）
- 过期策略：默认 3 天过期（expire_days=3）。
- 可控参数（代码层）：
  - use_cache=True|False：是否使用缓存
  - refresh=True|False：强制刷新（忽略过期策略）
  - expire_days=3：缓存过期天数

接口定义见 <mcfile name="ak_client.py" path="src/data/ak_client.py"></mcfile> 中的 <mcsymbol name="AKDataClient" filename="ak_client.py" path="src/data/ak_client.py" startline="9" type="class"></mcsymbol>。


## LLM 调用与工具（Function Calling）

- 前端可选择路由名（如 default/analysis），由 <mcsymbol name="MCPRouter" filename="adapter.py" path="src/llm/mcp/adapter.py" startline="6" type="class"></mcsymbol> 统一转发调用，底层实际模型/工具由 MCP 客户端配置决定（可在 routing.yaml 中登记路由名）。
- 工具定义：<mcfile name="schema.py" path="src/llm/tools/schema.py"></mcfile>（OpenAI tools 规范）
- 工具执行：<mcfile name="executor.py" path="src/llm/tools/executor.py"></mcfile>（应用端实际联网查询）
- Demo 工具：fetch_stock_info_a（东方财富源：个股基本信息）
- 在前端 <mcfile name="app.py" path="src/ui/app.py"></mcfile> 中，勾选“启用联网工具”后，模型可触发工具调用，应用端执行并回显结果

扩展新工具的步骤：
1. 在 schema.py 增加工具 schema（tools 数组中新对象）
2. 在 executor.py 实现对应函数
3. 在前端根据函数名调用执行器（可后续抽象为统一工具注册表）


## 如何新增提供商/模型/路由

- 新增提供商：在 models.yaml 的 providers 下增加一项，至少含 type 和 base_url。api_key 在 models.local.yaml 覆盖。
- 新增模型：在 models.yaml 的 models 列表中增加一项，指定 name 和 provider。
- 新增路由：推荐在 routing.yaml 中配置如下：

```yaml
routing:
  analysis:
    provider: deepseek
    model: deepseek-chat
```

前端在“路由名(route)”输入 analysis 即可使用该路由。


## 运行与开发

- 启动开发服务：
  - `streamlit run src/ui/app.py --server.port 8501`
- 可选优化：安装 watchdog 提升热重载性能：`pip install watchdog`
- 若 AKShare 拉取失败：请检查网络、代理与依赖版本；可尝试开启刷新或延长过期时间


## 安全与提交

- models.local.yaml 和 routing.local.yaml 已加入 .gitignore，不会提交敏感信息
- 建议不要在代码或仓库中明文打印/记录 API Key


## 已知限制与路线图

- 港股个股/行业信息工具尚未接入，将在后续版本补充
- 工具执行目前为 Demo 方式（按名称 if 分支），后续将引入“工具注册表”统一管理
- 回测策略较为基础，可进一步扩展交易成本、风控与绩效指标

欢迎 Issue/PR，共同完善！