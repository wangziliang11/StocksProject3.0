# StocksProject3.0 — A/H 股票分析与可视化（Streamlit + AKShare）

一个面向 A 股与港股的轻量级行情分析与可视化项目，聚焦“数据获取、持久化缓存、基础技术指标、可视化与简单回测”。前端采用 Streamlit，数据源使用 AKShare。

- 数据获取与缓存：统一封装 AKDataClient，按市场/代码/复权分桶缓存为 CSV；非日线周期通过日线重采样生成
- 技术指标与回测：提供 SMA/EMA/MACD/RSI 以及均线金叉、MACD 交叉、RSI 区间等基础回测
- 可视化：Plotly K 线 + 成交量，支持叠加均线、展示 MACD / RSI 子图
- 多页面：主页面与“股票详情”页（支持 URL 参数），自选列表与行业板块管理
- LLM 模块（OpenAI 兼容）：提供基础客户端与工具模块，便于后续扩展到研报辅助


## 目录结构

```
.
├── requirements.txt
├── models.yaml            # 占位：模型/提供商/路由的公共配置
├── routing.yaml           # 占位：路由注册（未来版本将对接）
├── src/
│   ├── data/
│   │   └── ak_client.py   # AKDataClient：A/H 行情获取 + 缓存 + 重采样
│   ├── logic/
│   │   └── indicators.py  # 指标与回测
│   ├── viz/
│   │   └── charts.py      # K 线与子图可视化
│   ├── llm/
│   │   ├── client.py      # OpenAI 兼容客户端（基础实现）
│   │   ├── tools/
│   │   │   ├── schema.py  # 工具 schema（占位示例）
│   │   │   └── executor.py# 工具执行器（占位示例）
│   └── ui/
│       ├── app.py         # Streamlit 前端入口
│       └── pages/
│           └── StockDetail.py  # 股票详情页（支持 URL 参数）
└── .gitignore             # 忽略 data/、secrets/ 与本地路由/模型覆盖
```


## 快速开始

1) 安装依赖（建议使用虚拟环境）

```
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

2) 运行前端

```
streamlit run src/ui/app.py --server.port 8501
```
打开浏览器访问 http://localhost:8501/ 即可。项目采用 Streamlit 多页面结构，左侧侧栏可进入“股票详情”等页面。


## 使用说明与功能概览

- 主页：
  - 查询 A 股 / 港股个股历史数据（支持日/周/月/季/年周期）
  - 配置均线窗口，选择是否显示 MACD / RSI 子图
  - 查看基础回测统计（均线金叉、MACD 交叉、RSI 区间策略）
  - 管理自选列表、行业板块（含自定义行业）

- 股票详情页（src/ui/pages/StockDetail.py）：
  - 支持 URL 参数：?market=A&symbol=600519 或 ?market=H&symbol=00700
  - 可加入自选；复用主页面的单股模块（历史数据、指标与回测、图表）

- 图表（src/viz/charts.py）：
  - Plotly K 线 + 成交量，支持叠加多条均线
  - 子图可选择 MACD / RSI，并提供统一 hover 与坐标样式

- 技术指标与回测（src/logic/indicators.py）：
  - 指标：SMA、EMA、MACD、RSI
  - 回测：均线金叉/死叉、MACD 交叉、RSI 区间策略（含收益、最大回撤、胜率与交易次数估计）


## API/模块文档

### 数据客户端（src/data/ak_client.py）
<mcfile name="ak_client.py" path="src/data/ak_client.py"></mcfile>

- 目标：统一封装 A 股 / 港股历史行情获取、标准化字段、日线缓存与非日线重采样
- 返回数据格式（DataFrame 列）：
  - [date, open, high, low, close, volume, amount, adj_factor, market, symbol]
- 公共方法：
  - get_hist(market, symbol, period="daily", start=None, end=None, adjust=None, use_cache=True, refresh=False, expire_days=3)
    - market："A" 或 "H"
    - symbol：股票代码（A 股示例 600519；港股示例 00700）
    - period："daily" | "weekly" | "monthly" | "quarterly" | "yearly"
    - start/end：可选的起止日期（YYYY-MM-DD 或 YYYYMMDD）
    - adjust：仅 A 股有效，"qfq"（前复权）| "hfq"（后复权）| None（不复权）
    - use_cache：是否使用本地缓存（默认 True）
    - refresh：是否强制刷新（忽略过期策略，默认 False）
    - expire_days：缓存过期天数，默认 3
  - get_a_hist(symbol, period="daily", start=None, end=None, adjust=None, use_cache=True, refresh=False, expire_days=3)
  - get_hk_hist(symbol, period="daily", start=None, end=None, use_cache=True, refresh=False, expire_days=3)
- 字段与数据口径：
  - 非日线均由日线重采样得到（开盘首、最高最大、最低最小、收盘末、量额求和、复权因子取末）
  - A 股代码自动规范化为 shXXXXXX / szXXXXXX；港股纯数字自动补零为 5 位
- 示例：

```python
from src.data.ak_client import AKDataClient
client = AKDataClient(cache_dir="data/cache")
# A 股：不复权日线
df_a = client.get_a_hist(symbol="600519", period="daily")
# A 股：前复权周线
df_a_w = client.get_a_hist(symbol="600519", period="weekly", adjust="qfq")
# 港股：腾讯 00700 月线
df_h = client.get_hk_hist(symbol="00700", period="monthly")
```

### 可视化（src/viz/charts.py）
<mcfile name="charts.py" path="src/viz/charts.py"></mcfile>

- 函数：kline_with_volume(df, title="K线图", ma_windows=None, show_macd=False, period="daily", second_rows=None) -> plotly.graph_objects.Figure
  - df：标准化行情数据 DataFrame（至少包含 date/open/high/low/close，可选 volume）
  - title：图标题
  - ma_windows：均线窗口列表（如 [5, 10, 20, 60]）
  - show_macd：兼容旧参数，若为 True 将在子图中展示 MACD（second_rows 未指定时生效）
  - period：用于设置坐标轴 rangebreak（daily 会隐藏周末并补工作日缺失；非日线只隐藏周末）
  - second_rows：价格下方子图选择，允许 ["成交量", "MACD", "RSI"]，最多两项
- 特性：
  - Plotly 蜡烛图 + 成交量柱；涨跌颜色区分（红涨绿跌）
  - MACD/RSI 子图，含参考线与统一 hover 文案
- 示例：

```python
from src.viz.charts import kline_with_volume
fig = kline_with_volume(df_a, title="贵州茅台(日线)", ma_windows=[5,10,20], second_rows=["成交量","MACD"], period="daily")
fig.show()
```

### 指标与回测（src/logic/indicators.py）
<mcfile name="indicators.py" path="src/logic/indicators.py"></mcfile>

- 指标：
  - sma(series, window) -> Series：简单移动平均
  - ema(series, span) -> Series：指数移动平均
  - macd(series, fast=12, slow=26, signal=9) -> Dict[str, Series]：返回 {"dif","dea","hist"}
  - rsi(series, period=14) -> Series：相对强弱指数（EWMA 版本）
- 回测：
  - backtest_ma_cross(df, short=10, long=20, fee=0.0005) -> Dict[str, float]
  - backtest_macd_cross(df, fast=12, slow=26, signal=9, fee=0.0005) -> Dict[str, float]
  - backtest_rsi(df, period=14, low=30, high=70, fee=0.0005) -> Dict[str, float]
- 回传统计字段：{"trades", "return", "max_drawdown", "win_rate"}
- 示例：

```python
from src.logic.indicators import backtest_ma_cross, backtest_macd_cross, backtest_rsi
res_ma = backtest_ma_cross(df_a, short=10, long=20)
res_macd = backtest_macd_cross(df_a)
res_rsi = backtest_rsi(df_a, period=14, low=30, high=70)
```


## 部署到云端指南（Streamlit Community Cloud / Railway / Render）

### 1) Streamlit Community Cloud（免费托管）
- 将仓库推送到公共 GitHub 仓库
- 登录 Streamlit Community Cloud，点击“New app”，选择仓库与分支
- 指定入口文件：src/ui/app.py；确保 requirements.txt 存在
- 可在“Advanced settings”添加 Secrets（如 API Key）与环境变量
- 部署后自动生成公共 URL；更新代码会自动触发重新部署

提示：若使用自定义端口，请保持默认或改为 8501；Cloud 会自动管理端口并以 headless 模式运行。

### 2) Railway（通用 PaaS）
- 创建新项目并连接 GitHub 仓库
- 设置 Build Command：`pip install -r requirements.txt`
- 设置 Start Command：`streamlit run src/ui/app.py --server.port $PORT --server.headless true`
- 在环境变量中配置必要的 Secrets（如 API Key）
- 部署完成后，访问 Railway 提供的域名

### 3) Render（通用 PaaS）
- 创建“Web Service”，连接 GitHub 仓库
- Runtime 选择 Python；Build Command：`pip install -r requirements.txt`
- Start Command：`streamlit run src/ui/app.py --server.port $PORT --server.headless true`
- 在“Environment”中设置环境变量与 Secrets
- 点击部署，等待服务上线

### 通用注意事项
- 端口：很多 PaaS 通过 `$PORT` 环境变量分配端口；Streamlit 启动命令应使用 `--server.port $PORT` 并开启 `--server.headless true`
- 依赖：requirements.txt 需完整，当前包含 pandas/numpy/akshare/plotly/streamlit/requests 等
- 缓存与数据：云端的本地文件系统可能是临时的；若需持久化，请改为外部存储（对象存储或数据库）
- Secrets：不要将敏感信息提交到仓库，使用平台的 Secret/Environment 功能管理

- 市场支持：A 股（market="A"）、港股（market="H"）
- 周期：daily/weekly/monthly/quarterly/yearly（非日线通过日线重采样）
- A 股复权：qfq/hfq/none 三种模式；当选择 qfq/hfq 时，会同时抓取未复权数据以估算复权因子
- 代码规范化：
  - A 股：支持 600519、SH600519、sz000001、000001.SZ 等输入，统一规范为 shXXXXXX / szXXXXXX
  - 港股：纯数字自动补零至 5 位（如 700 -> 00700），也兼容 00700.HK 等格式
- 缓存结构：data/cache/{A|H}/{symbol}/daily_{qfq|hfq|none}.csv
- 过期策略：默认 3 天（expire_days=3），可通过 use_cache / refresh 控制缓存使用与强制刷新


## LLM 模块（OpenAI 兼容）

- 位置：src/llm/client.py（基础客户端实现）
- 用途：兼容 OpenAI 风格 /v1/chat/completions 的接口；通过 base_url、api_key、model 配置调用
- 工具模块：src/llm/tools/schema.py、src/llm/tools/executor.py 提供示例结构，便于后续扩展研报辅助
- 说明：仓库根目录的 models.yaml / routing.yaml 当前作为占位配置，未来版本将对接前端路由与提供商注册


## DEMO 模式与缓存清理

- 在本地演示模式下可自动清空 Streamlit 缓存，触发条件：
  - 环境变量：LOCAL_DEMO / DEMO_MODE / STREAMLIT_DEMO 任一为 1/true/yes/on
  - 或 URL 参数：?demo=1（true/yes/on）
- 该功能仅在会话首次触发时清理，避免频繁清理影响性能


## 提交与安全

- .gitignore 已忽略以下本地文件/目录：
  - data/（缓存与自定义数据）
  - secrets/（本地敏感信息）
  - models.local.yaml、routing.local.yaml（本地覆盖配置）
  - src/ui/data/（前端侧的临时/本地数据）
- 建议不要在仓库中明文记录 API Keys；如需集成 LLM，请通过环境变量或本地未提交的配置文件管理


## 常见问题（FAQ）

- AKShare 拉取失败或超时：
  - 检查网络与依赖版本；必要时重试或在 UI 中启用“刷新/延长过期”
- 图表展示异常：
  - 请确认数据列包含 date/open/high/low/close/volume 等标准字段；项目会做字段名兼容与类型转换
- 港股代码格式：
  - 支持 700、00700、00700.HK 等输入，最终统一为五位数字存储与使用


## 路线图

- 行业与概念的更丰富数据源接入
- LLM 工具与研报工作流的前端集成与路由管理
- 回测模块扩展（交易成本、风控、绩效指标更完整）
- 数据质量与异常值处理的完善