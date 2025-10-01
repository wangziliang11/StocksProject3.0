# StocksProject3.0 — A/H 股票分析与可视化系统

一个面向 A 股与港股的轻量级行情分析与可视化项目，聚焦"数据获取、持久化缓存、基础技术指标、可视化与简单回测"。前端采用 Streamlit，数据源使用 AKShare，支持多种大语言模型集成。

## 核心特性

- **数据获取与缓存**：统一封装 AKDataClient，按市场/代码/复权分桶缓存为 CSV；非日线周期通过日线重采样生成
- **技术指标与回测**：提供 SMA/EMA/MACD/RSI 以及均线金叉、MACD 交叉、RSI 区间等基础回测
- **可视化**：Plotly K 线 + 成交量，支持叠加均线、展示 MACD / RSI 子图
- **多页面**：主页面与"股票详情"页（支持 URL 参数），自选列表与行业板块管理
- **LLM 模块**：支持多种 OpenAI 兼容的大语言模型，提供基础客户端与工具模块

## 项目结构

```
.
├── requirements.txt       # 项目依赖包
├── models.yaml           # 大语言模型配置文件
├── routing.yaml          # 模型路由配置文件
├── src/                  # 源代码目录
│   ├── __init__.py
│   ├── config/           # 配置模块
│   │   └── __init__.py
│   ├── data/             # 数据获取模块
│   │   ├── __init__.py
│   │   └── ak_client.py  # AKShare数据客户端
│   ├── logic/            # 业务逻辑模块
│   │   ├── __init__.py
│   │   └── indicators.py # 技术指标与回测逻辑
│   ├── viz/              # 可视化模块
│   │   ├── __init__.py
│   │   └── charts.py     # K线图表绘制
│   ├── llm/              # 大语言模型模块
│   │   ├── __init__.py
│   │   ├── client.py     # LLM客户端
│   │   ├── mcp/          # MCP协议支持
│   │   ├── providers/    # 模型提供商管理
│   │   │   ├── __init__.py
│   │   │   └── registry.py
│   │   └── tools/        # LLM工具集
│   │       ├── __init__.py
│   │       ├── schema.py
│   │       └── executor.py
│   └── ui/               # 用户界面模块
│       ├── __init__.py
│       ├── app.py        # Streamlit主应用
│       └── pages/        # 页面模块
│           └── StockDetail.py
└── .gitignore            # Git忽略文件配置
```

## 详细脚本说明

### 数据模块 (src/data/)

#### ak_client.py - AKShare数据客户端
**功能**：统一封装A股和港股历史行情数据获取，提供标准化的数据接口

**核心技术**：
- **数据源**：基于AKShare库获取股票数据
- **缓存机制**：按市场/代码/复权类型分桶存储CSV文件，支持过期检查
- **数据标准化**：统一输出格式 [date, open, high, low, close, volume, amount, adj_factor, market, symbol]
- **复权处理**：支持前复权(qfq)、后复权(hfq)，自动计算复权因子
- **重采样**：非日线周期通过日线数据重采样生成（周线、月线、季线、年线）

**主要类和方法**：
- `AKDataClient`: 主要客户端类
  - `get_a_hist()`: 获取A股历史数据
  - `get_hk_hist()`: 获取港股历史数据
  - `get_hist()`: 统一接口获取历史数据
  - `_normalize_a_prefixed()`: A股代码标准化（支持多种格式）
  - `_normalize_hk_symbol()`: 港股代码标准化
  - `_resample_ohlcv()`: OHLCV数据重采样

**实现方式**：
- 使用pandas进行数据处理和时间序列操作
- 文件系统缓存避免重复请求
- 异常处理确保数据获取的稳定性
- 支持多种AKShare接口的兼容性处理

### 业务逻辑模块 (src/logic/)

#### indicators.py - 技术指标与回测
**功能**：实现常用技术指标计算和简单回测策略

**核心技术**：
- **技术指标**：SMA、EMA、MACD、RSI等经典指标
- **回测框架**：基于pandas的向量化回测，支持手续费计算
- **策略实现**：均线金叉死叉、MACD交叉、RSI区间策略

**主要函数**：
- `sma(series, window)`: 简单移动平均线
- `ema(series, span)`: 指数移动平均线
- `macd(series, fast, slow, signal)`: MACD指标计算
- `rsi(series, period)`: RSI相对强弱指标
- `backtest_ma_cross()`: 均线交叉策略回测
- `backtest_macd_cross()`: MACD交叉策略回测
- `backtest_rsi()`: RSI区间策略回测

**实现方式**：
- 使用pandas的rolling和ewm方法进行指标计算
- 向量化操作提高计算效率
- 返回统一的回测结果格式（收益率、最大回撤、胜率、交易次数）

### 可视化模块 (src/viz/)

#### charts.py - K线图表绘制
**功能**：基于Plotly创建交互式K线图表，支持多种技术指标叠加

**核心技术**：
- **图表库**：Plotly Graph Objects，支持交互式操作
- **子图布局**：支持价格、成交量、MACD、RSI等多子图显示
- **样式定制**：红涨绿跌配色方案，符合中国股市习惯
- **数据兼容**：自动处理缺失数据和类型转换

**主要函数**：
- `kline_with_volume()`: 主要的K线图绘制函数
- `_calc_macd()`: 内部MACD计算（避免循环依赖）
- `_calc_rsi()`: 内部RSI计算

**实现方式**：
- 使用plotly.subplots创建多子图布局
- 支持动态子图选择（成交量、MACD、RSI）
- 自动隐藏非交易日，优化显示效果
- 响应式设计，适配不同屏幕尺寸

### 大语言模型模块 (src/llm/)

#### client.py - 基础LLM客户端
**功能**：提供OpenAI兼容的大语言模型客户端

**核心技术**：
- **协议兼容**：支持OpenAI API格式
- **多供应商**：可接入通义千问、DeepSeek、豆包等
- **HTTP请求**：基于requests库的RESTful API调用

**主要类**：
- `OpenAICompatConfig`: 配置数据类
- `LLMClient`: 基础客户端类
  - `chat()`: 聊天接口方法

#### providers/registry.py - 模型提供商注册中心
**功能**：管理多个LLM提供商的配置和路由

**核心技术**：
- **配置管理**：YAML文件解析，支持本地覆盖
- **路由系统**：动态选择模型提供商
- **代理支持**：HTTP/HTTPS代理配置
- **重试机制**：网络请求失败重试

**主要类**：
- `ProviderConfig`: 提供商配置
- `Route`: 路由配置
- `ProviderRegistry`: 注册中心管理器
- `OpenAICompatClient`: 兼容客户端
- `LLMRouter`: 路由器

**实现方式**：
- 支持公共配置(models.yaml)和本地配置(models.local.yaml)分离
- 自动合并配置文件，本地配置优先级更高
- 支持超时和重试配置，提高稳定性

#### tools/ - LLM工具集
**功能**：为LLM提供Function Calling工具

**schema.py**：
- 定义工具的JSON Schema格式
- 支持A股和港股基本信息查询工具

**executor.py**：
- 实现具体的工具执行逻辑
- `fetch_stock_info_a()`: A股信息查询
- `fetch_stock_info_hk()`: 港股信息查询

### 用户界面模块 (src/ui/)

#### app.py - Streamlit主应用
**功能**：提供完整的Web界面，包含股票查询、分析、可视化等功能

**核心技术**：
- **Web框架**：Streamlit构建交互式Web应用
- **缓存机制**：@st.cache_data装饰器优化性能
- **多页面**：支持主页和详情页导航
- **数据持久化**：JSON文件存储自选股和行业关注

**主要功能模块**：
- 单股查询与分析
- 自选股管理
- 行业板块分析
- 技术指标回测
- LLM智能问答

**实现方式**：
- 使用Streamlit的组件系统构建UI
- 集成所有后端模块提供完整功能
- 支持URL参数传递，便于分享和书签
- 响应式布局适配不同设备

#### pages/StockDetail.py - 股票详情页
**功能**：独立的股票详情页面，支持URL参数访问

**核心技术**：
- **参数解析**：支持新旧版本Streamlit的查询参数API
- **状态管理**：使用session_state保持页面状态
- **模块复用**：复用主应用的功能模块

## 配置文件说明

### requirements.txt - 项目依赖
包含所有必需的Python包及版本要求：
- `PyYAML>=6.0.1`: YAML配置文件解析
- `akshare>=1.11.95`: 股票数据获取
- `pandas>=2.0.0`: 数据处理和分析
- `numpy>=1.24.0`: 数值计算
- `plotly>=5.20.0`: 交互式图表
- `streamlit>=1.37.0`: Web应用框架
- `requests>=2.31.0`: HTTP请求库

### models.yaml - 大语言模型配置
定义支持的LLM提供商和模型：
- **providers**: 提供商配置（base_url、api_key等）
- **models**: 可用模型列表
- **routing**: 路由规则（default、fast、analysis等）

支持的提供商：
- SiliconFlow（硅基流动）
- Doubao（豆包）
- Qwen（通义千问）
- OpenAI兼容接口
- DeepSeek

### routing.yaml - 路由配置
独立的路由配置文件，支持环境无关的路由规则定义。

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

创建 `models.local.yaml` 文件配置API密钥：

```yaml
providers:
  siliconflow:
    api_key: "your_api_key_here"
  qwen:
    api_key: "your_qwen_api_key"
  # 其他提供商配置...
```

### 3. 运行应用

```bash
streamlit run src/ui/app.py
```

应用将在 `http://localhost:8501` 启动。

## 使用说明

### 股票查询
1. 选择市场（A股/港股）
2. 输入股票代码
3. 选择时间周期和复权方式
4. 查看K线图和技术指标

### 技术分析
- 支持多种移动平均线叠加
- MACD、RSI等技术指标子图显示
- 简单回测策略验证

### 行业分析
- 查看行业板块成分股
- 自定义行业分组
- 行业关注列表管理

### LLM问答
- 股票基本信息查询
- 技术分析建议
- 市场数据解读

## 技术架构

### 数据层
- **AKShare**: 数据源接口
- **CSV缓存**: 本地数据持久化
- **Pandas**: 数据处理引擎

### 业务层
- **技术指标**: 纯函数式计算
- **回测引擎**: 向量化策略验证
- **LLM集成**: 多提供商统一接口

### 表现层
- **Streamlit**: Web应用框架
- **Plotly**: 交互式图表
- **响应式设计**: 多设备适配

### 扩展性设计
- **模块化架构**: 松耦合组件设计
- **配置驱动**: YAML配置文件管理
- **插件化LLM**: 易于添加新的模型提供商

## 开发指南

### 添加新的技术指标
1. 在 `src/logic/indicators.py` 中实现指标函数
2. 在 `src/viz/charts.py` 中添加可视化支持
3. 在UI中添加相应的控制选项

### 集成新的LLM提供商
1. 在 `models.yaml` 中添加提供商配置
2. 确保API兼容OpenAI格式
3. 在 `models.local.yaml` 中配置API密钥

### 扩展数据源
1. 在 `src/data/` 中创建新的客户端类
2. 实现统一的数据接口格式
3. 添加相应的缓存机制

## 注意事项

1. **API限制**: 注意各数据源和LLM提供商的API调用限制
2. **数据质量**: AKShare数据可能存在延迟或缺失，建议结合多个数据源
3. **回测局限**: 简单回测不考虑滑点、冲击成本等实际交易因素
4. **风险提示**: 本系统仅供学习和研究使用，不构成投资建议

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。请确保：
1. 代码符合项目的编码规范
2. 添加必要的测试用例
3. 更新相关文档

## 更新日志

### v3.0
- 重构项目架构，采用模块化设计
- 集成多种大语言模型支持
- 优化数据缓存机制
- 增强可视化功能
- 添加行业分析功能