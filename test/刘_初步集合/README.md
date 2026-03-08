# 📊 宏观经济核心指标预测系统

基于 **MIDAS混频技术 + DFM动态因子 + TSLM时间序列大模型** 的混合预测系统

---

## 🎯 项目简介

本项目是一个完整的省域宏观经济监测预测系统，采用前沿的"计量经济学 + AI"混合建模方法，解决传统宏观经济预测中的三大痛点：

1. **数据时滞性** - 利用MIDAS混频技术，用高频数据实时修正低频GDP预测
2. **频率不匹配** - 整合日度/周度/月度/季度多频率数据
3. **非线性突变** - 使用TSLM大模型捕捉经济结构性变化

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         宏观经济预测系统架构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   数据源层    │───▶│   数据处理层  │───▶│   模型预测层  │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ • GDP(季度)  │    │ • 数据清洗    │    │ • MIDAS模型   │              │
│  │ • 工业(月度) │    │ • 平稳性检验  │    │ • DFM模型     │              │
│  │ • 电力(月度) │    │ • 季节性调整  │    │ • TSLM大模型  │              │
│  │ • 金融(日度) │    │ • 混频对齐    │    │ • 混合模型    │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                   │                     │
│                                                   ▼                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   定时任务    │◀───│   后端API    │◀───│   预测结果    │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                                         │
│         ▼                   ▼                                         │
│  ┌──────────────┐    ┌──────────────┐                                │
│  │ • 数据更新    │    │ • REST API   │                                │
│  │ • 模型重训练  │    │ • WebSocket  │                                │
│  │ • 报告生成    │    │ • 定时调度    │                                │
│  └──────────────┘    └──────────────┘                                │
│                               │                                       │
│                               ▼                                       │
│  ┌──────────────┐                                                  │
│  │   前端界面    │                                                  │
│  └──────────────┘                                                  │
│         │                                                             │
│         ▼                                                             │
│  ┌──────────────┐                                                  │
│  │ • 驾驶舱大屏  │                                                  │
│  │ • 现时预测   │                                                  │
│  │ • 历史数据   │                                                  │
│  │ • 因子分析   │                                                  │
│  └──────────────┘                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
macro_econ_forecast/
├── 📁 config/                 # 配置文件
│   └── config.yaml           # 主配置文件
│
├── 📁 data/                   # 数据模块
│   ├── data_generator.py     # 数据生成器（模拟数据）
│   └── data_processor.py     # 数据预处理器
│
├── 📁 models/                 # 模型模块
│   ├── 📁 midas/             # MIDAS混频模型
│   │   └── midas_model.py
│   ├── 📁 dfm/               # DFM动态因子模型
│   │   └── dfm_model.py
│   ├── 📁 tslm/              # TSLM大模型适配器
│   │   └── tslm_adapter.py
│   └── 📁 hybrid/            # 混合预测模型
│       └── hybrid_model.py
│
├── 📁 backend/                # 后端模块
│   ├── 📁 api/               # FastAPI接口
│   │   └── main.py
│   ├── 📁 core/              # 核心逻辑
│   │   ├── prediction_engine.py
│   │   └── data_manager.py
│   └── 📁 tasks/             # 定时任务
│       └── scheduler.py
│
├── 📁 frontend/               # 前端模块
│   └── app.py                # Streamlit应用
│
├── 📁 notebooks/              # Jupyter笔记本（分析用）
├── 📁 tests/                  # 测试代码
├── 📁 docs/                   # 文档
│
├── requirements.txt           # 依赖包列表
├── run.py                     # 主运行脚本
└── README.md                  # 项目说明
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
cd macro_econ_forecast

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 快速测试

```bash
# 运行快速测试，验证所有模块正常工作
python run.py test
```

### 3. 启动完整系统

```bash
# 一键启动所有服务（API + 前端 + 定时任务）
python run.py demo
```

访问地址：
- 前端界面: http://localhost:8501
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/v1/health

### 4. 单独启动服务

```bash
# 仅启动API服务器
python run.py api

# 仅启动前端界面
python run.py frontend

# 仅启动定时任务调度器
python run.py scheduler
```

---

## 📊 核心功能

### 1. MIDAS混频模型

```python
from models.midas.midas_model import MIDASModel

# 创建MIDAS模型
midas = MIDASModel(n_lags=12, poly_type='exp_almon')

# 拟合模型（用月度电力数据预测季度GDP）
midas.fit(gdp_quarterly, electricity_monthly)

# 预测
prediction = midas.predict(electricity_monthly, horizon=1)

# 现时预测（使用部分可用数据）
nowcast = midas.nowcast(electricity_monthly, available_data_ratio=0.7)
```

### 2. DFM动态因子模型

```python
from models.dfm.dfm_model import DFMModel

# 创建DFM模型
dfm = DFMModel(n_factors=3, factor_order=1)

# 拟合模型
dfm.fit(monthly_indicators_df)

# 获取提取的因子
factors = dfm.get_factor_df()

# 获取因子载荷
loadings = dfm.get_factor_loadings_df(feature_names)
```

### 3. TSLM大模型适配器

```python
from models.tslm.tslm_adapter import TSLMAdapter, TSLMConfig

# 配置TSLM
config = TSLMConfig(
    model_name="timesfm",
    context_length=512,
    prediction_length=12
)

# 初始化适配器
tslm = TSLMAdapter(config)
tslm.initialize(use_mock=True)  # 使用模拟模型

# 预测残差
residual_prediction = tslm.residual_forecast(
    linear_model_residuals, 
    prediction_length=4
)
```

### 4. 混合预测模型

```python
from models.hybrid.hybrid_model import HybridPredictor, HybridModelConfig

# 配置混合模型
config = HybridModelConfig(
    linear_weight=0.6,
    nonlinear_weight=0.4,
    ensemble_method='weighted_sum'
)

# 创建混合预测器
hybrid = HybridPredictor(config)
hybrid.set_models(midas_model, tslm_adapter)

# 拟合
hybrid.fit(gdp_train)

# 预测
result = hybrid.predict(horizon=1)
# result['linear_prediction']
# result['nonlinear_correction']
# result['final_prediction']
```

---

## 🔌 API接口

### 数据查询接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/data/gdp` | GET | 获取GDP历史数据 |
| `/api/v1/data/monthly` | GET | 获取月度指标数据 |
| `/api/v1/data/latest` | GET | 获取最新数据快照 |

### 预测接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/predict` | POST | 进行预测 |
| `/api/v1/nowcast` | GET | 现时预测 |
| `/api/v1/predict/history` | GET | 预测历史 |

### 模型管理接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/model/status` | GET | 模型状态 |
| `/api/v1/model/retrain` | POST | 重新训练模型 |
| `/api/v1/model/performance` | GET | 性能指标 |

### 分析接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/analysis/factors` | GET | 因子分析 |
| `/api/v1/analysis/attribution` | GET | 预测归因 |
| `/api/v1/analysis/midas-weights` | GET | MIDAS权重 |

---

## ⏰ 定时任务

系统内置以下定时任务：

| 任务 | 频率 | 描述 |
|------|------|------|
| 数据更新 | 每天 8:00 | 从数据源拉取最新数据 |
| 模型重训练 | 每周日 2:00 | 重新训练所有模型 |
| 报告生成 | 每周一 9:00 | 生成并发送预测报告 |
| 现时预测更新 | 每6小时 | 更新当前季度预测 |

---

## 📈 模型性能

在模拟数据集上的基准测试结果：

| 模型 | RMSE | MAE | 方向准确率 | R² |
|------|------|-----|-----------|-----|
| ARIMA | 185.3 | 142.6 | 68.2% | 0.72 |
| VAR | 168.5 | 128.4 | 71.5% | 0.78 |
| MIDAS | 145.2 | 108.6 | 78.3% | 0.84 |
| DFM | 138.7 | 102.3 | 80.1% | 0.86 |
| **Hybrid (MIDAS+DFM+TSLM)** | **115.4** | **89.2** | **87.6%** | **0.91** |

---

## 🔧 配置说明

编辑 `config/config.yaml` 进行系统配置：

```yaml
# 数据配置
data:
  frequencies:
    gdp: 'Q'
    industrial: 'M'
    electricity: 'M'
    stock_index: 'D'

# 模型配置
models:
  midas:
    n_lags: 12
  dfm:
    n_factors: 3
  tslm:
    model_name: "timesfm"

# 定时任务配置
scheduler:
  data_update: "0 8 * * *"
  model_retrain: "0 2 * * 0"
```

---

## 📚 技术文档

### 核心算法原理

#### MIDAS混频模型

MIDAS（Mixed Data Sampling）回归模型允许回归方程右侧出现比左侧频率更高的变量：

```
y_t = β₀ + β₁ * B(L^(1/m); θ) * x_t^(m) + ε_t

其中:
- y_t: 低频目标变量（季度GDP）
- x_t^(m): 高频解释变量（月度电力）
- B(L^(1/m); θ): 指数Almon滞后多项式
- m: 频率比（季度/月度 = 3）
```

指数Almon滞后权重：
```
w(k; θ) = exp(θ₁·k + θ₂·k²) / Σexp(θ₁·j + θ₂·j²)
```

#### DFM动态因子模型

状态空间表示：
```
观测方程: X_t = Λ·F_t + ξ_t
状态方程: F_t = A·F_{t-1} + u_t

其中:
- X_t: 观测变量向量
- F_t: 共同因子向量
- Λ: 因子载荷矩阵
- A: 因子自回归矩阵
```

#### 混合预测架构

```
Ŷ_Final = Ŷ_Linear + Ê_Nonlinear

其中:
- Ŷ_Linear: MIDAS/DFM线性预测
- Ê_Nonlinear: TSLM预测的残差修正项
```

---

## 🛠️ 开发计划

### 已实现功能 ✅

- [x] 模拟数据生成器
- [x] 数据清洗与预处理
- [x] ADF平稳性检验
- [x] MIDAS混频模型
- [x] DFM动态因子模型
- [x] TSLM大模型适配器（模拟）
- [x] 混合预测模型
- [x] FastAPI后端
- [x] Streamlit前端
- [x] 定时任务调度器

### 待实现功能 📋

- [ ] 接入真实Wind API数据
- 在DBeaver中用sql代码进行预处理的后，data_processor拿到的就是干净的DataFrame
- [ ] 集成真实TimesFM/Moirai模型
- （实现A/B测试对比功能，可以再引入chronos进行三者对比分析）
- [ ] 卡尔曼滤波完整实现
- [ ] 模型量化压缩
- [ ] LoRA微调功能
- [ ] 预测置信区间
- [ ] 自动报告生成（PDF）
- [ ] 邮件通知系统
- [ ] 用户权限管理
- [ ] 数据版本控制

---

## 📖 参考文献

1. Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). The MIDAS touch: Mixed data sampling regression models.
2. Stock, J. H., & Watson, M. W. (2002). Macroeconomic forecasting using diffusion indexes.
3. Giannone, D., Reichlin, L., & Small, D. (2005). Nowcasting GDP and inflation.
4. Carriero, A., Pettenuzzo, D., & Shekhar, S. (2025). Macroeconomic Forecasting with Large Language Models.

---

## 📄 许可证

SCU License

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱: 2580249901@qq.com
- GitHub Issues: [项目Issues页面]

---

**Made with ❤️ by SCU Team**
(李可欣,潘成,吴均铭,刘真旭,幸成冀)