"""
TSLM (Time Series Large Model) 时间序列大模型适配器
支持TimesFM、Moirai等预训练大模型
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TSLMConfig:
    """TSLM配置"""
    model_name: str = "timesfm"  # timesfm, moirai, chronos
    context_length: int = 512
    prediction_length: int = 12
    patch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MockTSLM(nn.Module):
    """
    模拟TSLM大模型（用于demo演示）
    实际部署时需要替换为真实的TimesFM/Moirai模型
    """
    
    def __init__(self, config: TSLMConfig):
        super().__init__()
        self.config = config
        
        # 模拟Transformer架构
        self.embedding = nn.Linear(config.patch_size, 128)
        
        # 多层Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 输出层
        self.output = nn.Linear(128, config.patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, seq_len, patch_size)
            
        Returns:
            预测序列
        """
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, 128)
        
        # Transformer编码
        x = self.transformer(x)  # (batch, seq_len, 128)
        
        # 输出
        x = self.output(x)  # (batch, seq_len, patch_size)
        
        return x
    
    def predict(self, context: np.ndarray, prediction_length: int) -> np.ndarray:
        """
        预测未来值
        
        Args:
            context: 历史序列
            prediction_length: 预测长度
            
        Returns:
            预测值
        """
        self.eval()
        with torch.no_grad():
            # 将序列分块
            patches = self._create_patches(context)
            patches_tensor = torch.FloatTensor(patches).unsqueeze(0).to(next(self.parameters()).device)
            
            # 自回归预测
            predictions = []
            current_patches = patches_tensor.clone()
            
            for _ in range(prediction_length // self.config.patch_size + 1):
                output = self.forward(current_patches)
                last_patch = output[0, -1, :].cpu().numpy()
                predictions.extend(last_patch)
                
                # 滑动窗口更新
                new_patch = torch.FloatTensor(last_patch).unsqueeze(0).unsqueeze(0).to(next(self.parameters()).device)
                current_patches = torch.cat([current_patches[:, 1:, :], new_patch], dim=1)
            
            return np.array(predictions[:prediction_length])
    
    def _create_patches(self, series: np.ndarray) -> np.ndarray:
        """将序列分块"""
        patch_size = self.config.patch_size
        n_patches = len(series) // patch_size
        patches = series[-n_patches * patch_size:].reshape(n_patches, patch_size)
        return patches


class TSLMAdapter:
    """
    时间序列大模型适配器
    提供统一的接口来使用不同的TSLM模型
    """
    
    def __init__(self, config: Optional[TSLMConfig] = None):
        """
        Args:
            config: TSLM配置
        """
        self.config = config or TSLMConfig()
        self.model = None
        self.is_initialized = False
        
    def initialize(self, use_mock: bool = True):
        """
        初始化模型
        
        Args:
            use_mock: 是否使用模拟模型（用于demo）
        """
        if use_mock:
            print(f"🔄 初始化模拟TSLM模型...")
            self.model = MockTSLM(self.config)
            self.model.to(self.config.device)
            self.is_initialized = True
            print(f"✅ 模拟TSLM模型初始化完成 (device: {self.config.device})")
        else:
            # 实际部署时加载真实模型
            self._load_real_model()
            
    def _load_real_model(self):
        """
        加载真实的TSLM模型
        需要安装相应的库和下载模型权重
        """
        model_name = self.config.model_name.lower()
        
        if model_name == "timesfm":
            self._load_timesfm()
        elif model_name == "moirai":
            self._load_moirai()
        elif model_name == "chronos":
            self._load_chronos()
        else:
            raise ValueError(f"不支持的模型: {model_name}")
            
    def _load_timesfm(self):
        """
        加载Google TimesFM模型
        
        安装命令:
        pip install timesfm
        
        需要下载模型权重到本地
        """
        try:
            import timesfm
            
            print("🔄 加载TimesFM模型...")
            # 实际加载代码（需要模型权重）
            # self.model = timesfm.TimesFm(
            #     context_len=self.config.context_length,
            #     horizon_len=self.config.prediction_length,
            #     input_patch_len=self.config.patch_size,
            #     output_patch_len=self.config.patch_size,
            #     num_layers=20,
            #     model_dims=1280,
            # )
            # self.model.load_from_checkpoint("path/to/checkpoint")
            
            print("⚠️ TimesFM模型加载需要预训练权重，当前使用模拟模型")
            self.model = MockTSLM(self.config)
            self.is_initialized = True
            
        except ImportError:
            print("⚠️ timesfm库未安装，使用模拟模型")
            print("   安装命令: pip install timesfm")
            self.model = MockTSLM(self.config)
            self.is_initialized = True
            
    def _load_moirai(self):
        """
        加载Salesforce Moirai模型
        
        安装命令:
        pip install uni2ts
        """
        try:
            from uni2ts.model import MoiraiForecast
            
            print("🔄 加载Moirai模型...")
            # 实际加载代码
            # self.model = MoiraiForecast.load_from_checkpoint("path/to/checkpoint")
            
            print("⚠️ Moirai模型加载需要预训练权重，当前使用模拟模型")
            self.model = MockTSLM(self.config)
            self.is_initialized = True
            
        except ImportError:
            print("⚠️ uni2ts库未安装，使用模拟模型")
            print("   安装命令: pip install uni2ts")
            self.model = MockTSLM(self.config)
            self.is_initialized = True
            
    def _load_chronos(self):
        """
        加载Amazon Chronos模型
        
        安装命令:
        pip install chronos-forecasting
        """
        try:
            from chronos import ChronosPipeline
            
            print("🔄 加载Chronos模型...")
            # 实际加载代码
            # self.model = ChronosPipeline.from_pretrained("path/to/model")
            
            print("⚠️ Chronos模型加载需要预训练权重，当前使用模拟模型")
            self.model = MockTSLM(self.config)
            self.is_initialized = True
            
        except ImportError:
            print("⚠️ chronos库未安装，使用模拟模型")
            print("   安装命令: pip install chronos-forecasting")
            self.model = MockTSLM(self.config)
            self.is_initialized = True
    
    def forecast(self, series: Union[pd.Series, np.ndarray], 
                 prediction_length: Optional[int] = None) -> np.ndarray:
        """
        单变量预测
        
        Args:
            series: 历史时间序列
            prediction_length: 预测长度（默认使用配置值）
            
        Returns:
            预测值数组
        """
        if not self.is_initialized:
            raise RuntimeError("模型尚未初始化，请先调用initialize()")
            
        pred_len = prediction_length or self.config.prediction_length
        
        # 转换为numpy数组
        if isinstance(series, pd.Series):
            values = series.values
        else:
            values = series
            
        # 标准化
        mean = np.mean(values)
        std = np.std(values)
        values_normalized = (values - mean) / (std + 1e-8)
        
        # 预测
        predictions_normalized = self.model.predict(values_normalized, pred_len)
        
        # 反标准化
        predictions = predictions_normalized * std + mean
        
        return predictions
    
    def forecast_batch(self, series_list: List[Union[pd.Series, np.ndarray]], 
                       prediction_length: Optional[int] = None) -> List[np.ndarray]:
        """
        批量预测
        
        Args:
            series_list: 时间序列列表
            prediction_length: 预测长度
            
        Returns:
            预测结果列表
        """
        return [self.forecast(s, prediction_length) for s in series_list]
    
    def residual_forecast(self, residuals: Union[pd.Series, np.ndarray],
                          prediction_length: Optional[int] = None) -> np.ndarray:
        """
        残差预测（用于混合模型）
        
        这是TSLM在混合架构中的核心用途：
        预测线性模型未能捕捉的非线性残差
        
        Args:
            residuals: 线性模型的残差序列
            prediction_length: 预测长度
            
        Returns:
            残差预测值
        """
        print(f"🔄 TSLM预测非线性残差...")
        predictions = self.forecast(residuals, prediction_length)
        print(f"✅ 残差预测完成，预测长度: {len(predictions)}")
        return predictions


class TSLMTrainer:
    """
    TSLM微调训练器（使用LoRA等技术）
    """
    
    def __init__(self, adapter: TSLMAdapter):
        self.adapter = adapter
        
    def finetune_lora(self, train_data: List[np.ndarray], 
                      val_data: Optional[List[np.ndarray]] = None,
                      epochs: int = 10, 
                      learning_rate: float = 1e-4,
                      lora_rank: int = 8):
        """
        使用LoRA微调模型
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            epochs: 训练轮数
            learning_rate: 学习率
            lora_rank: LoRA秩
        """
        print(f"🔄 使用LoRA微调TSLM (rank={lora_rank})...")
        
        # 这里实现LoRA微调逻辑
        # 实际部署时需要使用peft库
        
        print(f"⚠️ LoRA微调需要peft库，当前为演示模式")
        print(f"   安装命令: pip install peft")
        
        # 模拟训练过程
        for epoch in range(epochs):
            print(f"   Epoch {epoch+1}/{epochs} - 模拟训练...")
            
        print(f"✅ LoRA微调完成")


def install_timesfm_instructions():
    """
    输出TimesFM安装指导
    """
    instructions = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                     TimesFM 模型安装与部署指导                                ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  1. 安装依赖库                                                                ║
    ║     pip install timesfm                                                      ║
    ║     pip install jax jaxlib  # TimesFM依赖JAX                                 ║
    ║                                                                              ║
    ║  2. 下载预训练权重                                                            ║
    ║     # 从Google Cloud Storage下载                                              ║
    ║     gsutil cp -r gs://timesfm/models/ ./models/timesfm/                      ║
    ║                                                                              ║
    ║     或使用HuggingFace                                                         ║
    ║     from huggingface_hub import snapshot_download                            ║
    ║     snapshot_download(repo_id="google/timesfm-1.0-200m")                     ║
    ║                                                                              ║
    ║  3. 加载模型                                                                  ║
    ║     import timesfm                                                           ║
    ║                                                                              ║
    ║     model = timesfm.TimesFm(                                                 ║
    ║         context_len=512,                                                     ║
    ║         horizon_len=12,                                                      ║
    ║         input_patch_len=32,                                                  ║
    ║         output_patch_len=128,                                                ║
    ║         num_layers=20,                                                       ║
    ║         model_dims=1280,                                                     ║
    ║         backend='gpu'  # 或 'cpu'                                            ║
    ║     )                                                                        ║
    ║     model.load_from_checkpoint("./models/timesfm/checkpoint_1100000")        ║
    ║                                                                              ║
    ║  4. 进行预测                                                                  ║
    ║     forecast = model.forecast(                                               ║
    ║         inputs=[train_data],                                                 ║
    ║         freq=[0]  # 0表示高频数据                                            ║
    ║     )                                                                        ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


def install_moirai_instructions():
    """
    输出Moirai安装指导
    """
    instructions = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                     Moirai 模型安装与部署指导                                 ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  1. 安装依赖库                                                                ║
    ║     pip install uni2ts                                                       ║
    ║     pip install torch>=2.0.0                                                 ║
    ║                                                                              ║
    ║  2. 下载预训练权重                                                            ║
    ║     from huggingface_hub import snapshot_download                            ║
    ║     snapshot_download(                                                       ║
    ║         repo_id="Salesforce/moirai-1.0-R-small",                             ║
    ║         local_dir="./models/moirai"                                          ║
    ║     )                                                                        ║
    ║                                                                              ║
    ║  3. 加载模型                                                                  ║
    ║     from uni2ts.model import MoiraiForecast, MoiraiModule                    ║
    ║                                                                              ║
    ║     model = MoiraiForecast(                                                  ║
    ║         module=MoiraiModule.from_pretrained("./models/moirai"),              ║
    ║         prediction_length=12,                                                ║
    ║         context_length=512,                                                  ║
    ║     )                                                                        ║
    ║                                                                              ║
    ║  4. 进行预测                                                                  ║
    ║     from einops import rearrange                                             ║
    ║     import torch                                                             ║
    ║                                                                              ║
    ║     # 准备数据                                                                ║
    ║     past_target = torch.tensor(train_data).unsqueeze(0)                      ║
    ║     past_observed = torch.ones_like(past_target, dtype=torch.bool)           ║
    ║                                                                              ║
    ║     # 预测                                                                    ║
    ║     forecast = model(                                                        ║
    ║         past_target=rearrange(past_target, '... -> ... 1'),                  ║
    ║         past_observed_target=rearrange(past_observed, '... -> ... 1'),       ║
    ║     )                                                                        ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


if __name__ == "__main__":
    # 测试TSLM适配器
    print("=" * 60)
    print("TSLM时间序列大模型适配器测试")
    print("=" * 60)
    
    # 初始化适配器
    config = TSLMConfig(model_name="timesfm", context_length=128, prediction_length=12)
    adapter = TSLMAdapter(config)
    adapter.initialize(use_mock=True)
    
    # 生成测试数据
    np.random.seed(42)
    test_series = np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.1, 200)
    
    # 预测
    forecast = adapter.forecast(test_series, prediction_length=12)
    print(f"\n📊 预测结果: {forecast}")
    
    # 输出安装指导
    print("\n")
    install_timesfm_instructions()
