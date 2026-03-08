# TSLM时间序列大模型部署指南

本指南详细说明如何部署真实的TimesFM、Moirai等时间序列大模型，替代当前demo中的模拟模型。

---

## 📋 目录

1. [TimesFM部署](#timesfm部署)
2. [Moirai部署](#moirai部署)
3. [Chronos部署](#chronos部署)
4. [模型量化与优化](#模型量化与优化)
5. [LoRA微调](#lora微调)
6. [性能优化](#性能优化)

---

## TimesFM部署

### 1. 环境准备

```bash
# 创建新环境
conda create -n timesfm python=3.10
conda activate timesfm

# 安装基础依赖
pip install torch>=2.0.0
pip install jax jaxlib  # TimesFM依赖JAX

# 安装TimesFM
pip install timesfm
```

### 2. 下载预训练权重

#### 方式一：从Google Cloud Storage下载

```bash
# 安装gsutil
pip install gsutil

# 下载模型权重
gsutil cp -r gs://timesfm/models/ ./models/timesfm/

# 目录结构
models/timesfm/
├── checkpoint_1100000/
│   ├── checkpoint
│   ├── checkpoint_1100000
│   └── ...
└── ...
```

#### 方式二：从HuggingFace下载

```python
from huggingface_hub import snapshot_download

# 下载模型
snapshot_download(
    repo_id="google/timesfm-1.0-200m",
    local_dir="./models/timesfm",
    local_dir_use_symlinks=False
)
```

### 3. 加载模型

```python
import timesfm

# 初始化模型
model = timesfm.TimesFm(
    context_len=512,          # 上下文长度
    horizon_len=12,           # 预测长度
    input_patch_len=32,       # 输入分块长度
    output_patch_len=128,     # 输出分块长度
    num_layers=20,            # Transformer层数
    model_dims=1280,          # 模型维度
    backend='gpu'             # 后端: 'gpu' 或 'cpu'
)

# 加载权重
model.load_from_checkpoint("./models/timesfm/checkpoint_1100000")

print("✅ TimesFM模型加载成功")
```

### 4. 进行预测

```python
import numpy as np

# 准备数据
history = np.array([...])  # 历史时间序列，长度 >= context_len

# 预测
forecast = model.forecast(
    inputs=[history],
    freq=[0]  # 频率编码: 0=高频, 1=日度, 2=周度, 3=月度, 4=季度
)

# forecast.shape: (num_series, horizon_len)
print(f"预测结果: {forecast[0]}")
```

### 5. 批量预测

```python
# 多个序列批量预测
histories = [
    np.array([...]),  # 序列1
    np.array([...]),  # 序列2
    np.array([...]),  # 序列3
]

forecasts = model.forecast(
    inputs=histories,
    freq=[0, 0, 0]  # 每个序列的频率
)
```

---

## Moirai部署

### 1. 环境准备

```bash
# 创建新环境
conda create -n moirai python=3.10
conda activate moirai

# 安装依赖
pip install torch>=2.0.0
pip install einops
pip install uni2ts  # Moirai官方库
```

### 2. 下载预训练权重

```python
from huggingface_hub import snapshot_download

# 可选模型:
# - Salesforce/moirai-1.0-R-small
# - Salesforce/moirai-1.0-R-base
# - Salesforce/moirai-1.0-R-large

snapshot_download(
    repo_id="Salesforce/moirai-1.0-R-small",
    local_dir="./models/moirai",
    local_dir_use_symlinks=False
)
```

### 3. 加载模型

```python
from uni2ts.model import MoiraiForecast, MoiraiModule
import torch

# 加载模型模块
module = MoiraiModule.from_pretrained("./models/moirai")

# 创建预测模型
model = MoiraiForecast(
    module=module,
    prediction_length=12,     # 预测长度
    context_length=512,       # 上下文长度
    patch_size=32,            # 分块大小
    num_samples=100,          # 采样次数（用于不确定性估计）
)

print("✅ Moirai模型加载成功")
```

### 4. 进行预测

```python
from einops import rearrange
import numpy as np

# 准备数据
history = np.array([...])  # 历史序列

# 转换为tensor
past_target = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
past_observed = torch.ones_like(past_target, dtype=torch.bool)

# 预测
forecast = model(
    past_target=rearrange(past_target, '... -> ... 1'),
    past_observed_target=rearrange(past_observed, '... -> ... 1'),
)

# 提取预测结果
predictions = forecast.mean(dim=0).numpy()
print(f"预测结果: {predictions}")
```

---

## Chronos部署

### 1. 环境准备

```bash
# 创建新环境
conda create -n chronos python=3.10
conda activate chronos

# 安装Chronos
pip install chronos-forecasting
```

### 2. 下载预训练权重

```python
from chronos import ChronosPipeline

# 加载预训练模型
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # 可选: tiny, small, base, large
    device_map="cuda",           # 使用GPU
    torch_dtype=torch.bfloat16,  # 使用混合精度
)

print("✅ Chronos模型加载成功")
```

### 3. 进行预测

```python
import torch
import numpy as np

# 准备数据
history = np.array([...])
context = torch.tensor(history, dtype=torch.float32)

# 预测
forecast = pipeline.predict(
    context=context,
    prediction_length=12,      # 预测长度
    num_samples=20,            # 采样次数
    temperature=1.0,           # 采样温度
    top_k=50,                  # Top-K采样
    top_p=1.0,                 # Top-P采样
)

# forecast.shape: (num_samples, prediction_length)
predictions = forecast.mean(dim=0).numpy()
print(f"预测结果: {predictions}")
```

---

## 模型量化与优化

### 1. 动态量化（CPU推理优化）

```python
import torch

# 对模型进行动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 保存量化模型
torch.save(quantized_model.state_dict(), "model_quantized.pth")
```

### 2. ONNX导出

```python
import torch.onnx

# 导出为ONNX格式
dummy_input = torch.randn(1, 512)  # 根据模型输入调整
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence"},
        "output": {0: "batch_size"}
    }
)
```

### 3. TensorRT加速（NVIDIA GPU）

```python
import tensorrt as trt
import pycuda.driver as cuda

# 构建TensorRT引擎
def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    # 配置builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)  # 启用FP16
    
    # 构建引擎
    engine = builder.build_engine(network, config)
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine

# 构建引擎
build_engine("model.onnx", "model.trt")
```

---

## LoRA微调

### 1. 安装PEFT库

```bash
pip install peft
```

### 2. LoRA配置

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,                    # LoRA秩
    lora_alpha=32,          # 缩放参数
    lora_dropout=0.1,       # Dropout率
    bias="none",
    target_modules=[        # 目标模块
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj"
    ]
)

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### 3. 微调训练

```python
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# 自定义数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_length, prediction_length):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
    
    def __len__(self):
        return len(self.data) - self.context_length - self.prediction_length
    
    def __getitem__(self, idx):
        context = self.data[idx:idx + self.context_length]
        target = self.data[idx + self.context_length:
                          idx + self.context_length + self.prediction_length]
        return torch.tensor(context, dtype=torch.float32), \
               torch.tensor(target, dtype=torch.float32)

# 准备数据
train_data = [...]  # 省级历史数据
dataset = TimeSeriesDataset(train_data, context_length=512, prediction_length=12)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 训练
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
num_epochs = 10

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (context, target) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(context)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(predictions, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 保存微调后的模型
model.save_pretrained("./models/timesfm_lora_finetuned")
```

### 4. 加载微调模型

```python
from peft import PeftModel

# 加载基础模型
base_model = timesfm.TimesFm(...)
base_model.load_from_checkpoint("./models/timesfm/checkpoint_1100000")

# 加载LoRA权重
model = PeftModel.from_pretrained(
    base_model,
    "./models/timesfm_lora_finetuned"
)

print("✅ 微调模型加载成功")
```

---

## 性能优化

### 1. 批处理优化

```python
# 使用批处理提高吞吐量
def batch_predict(model, data_list, batch_size=32):
    results = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        forecasts = model.forecast(inputs=batch, freq=[0] * len(batch))
        results.extend(forecasts)
    return results
```

### 2. 缓存机制

```python
import functools
import hashlib

# 预测结果缓存
def cache_prediction(func):
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 生成缓存键
        key = hashlib.md5(
            str(args).encode() + str(kwargs).encode()
        ).hexdigest()
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return wrapper

@cache_prediction
def predict_with_cache(model, data):
    return model.forecast(inputs=[data], freq=[0])
```

### 3. 异步推理

```python
import asyncio
import concurrent.futures

async def async_predict(model, data_list):
    loop = asyncio.get_event_loop()
    
    with concurrent.futures.ThreadPoolExecutor() as pool:
        tasks = [
            loop.run_in_executor(pool, model.forecast, [data], [0])
            for data in data_list
        ]
        results = await asyncio.gather(*tasks)
    
    return results

# 使用
results = asyncio.run(async_predict(model, data_list))
```

---

## 部署检查清单

- [ ] 安装所有依赖
- [ ] 下载预训练权重
- [ ] 测试模型加载
- [ ] 验证预测功能
- [ ] 配置GPU/CPU
- [ ] 设置批处理大小
- [ ] 启用缓存机制
- [ ] 配置监控日志
- [ ] 测试性能基准
- [ ] 部署到生产环境

---

## 故障排除

### 显存不足

```python
# 减少批处理大小
batch_size = 4  # 从32减少到4

# 使用混合精度
from torch.cuda.amp import autocast

with autocast():
    forecast = model.forecast(...)

# 清理缓存
torch.cuda.empty_cache()
```

### 模型加载失败

```bash
# 检查模型文件完整性
ls -lh models/timesfm/checkpoint_1100000/

# 重新下载
rm -rf models/timesfm/
gsutil cp -r gs://timesfm/models/ ./models/timesfm/
```

### 预测结果异常

```python
# 检查输入数据
print(f"Input shape: {history.shape}")
print(f"Input range: [{history.min()}, {history.max()}]")
print(f"Input mean: {history.mean()}, std: {history.std()}")

# 数据标准化
history_normalized = (history - history.mean()) / history.std()
```

---

## 参考资源

- [TimesFM GitHub](https://github.com/google-research/timesfm)
- [Moirai GitHub](https://github.com/SalesforceAIResearch/uni2ts)
- [Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [HuggingFace Time Series](https://huggingface.co/blog/time-series-transformers)
