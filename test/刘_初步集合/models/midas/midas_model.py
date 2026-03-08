"""
MIDAS (Mixed Data Sampling) 混频数据抽样模型
用于高频数据预测低频目标变量（如用月度数据预测季度GDP）
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MIDASParams:
    """MIDAS模型参数"""
    beta0: float  # 截距
    beta1: float  # 高频变量系数
    theta1: float  # Almon多项式参数1
    theta2: float  # Almon多项式参数2
    

class ExponentialAlmonLags:
    """
    指数Almon滞后多项式权重函数
    这是MIDAS的核心，用于对高频滞后项进行加权
    """
    
    @staticmethod
    def weights(theta1: float, theta2: float, n_lags: int) -> np.ndarray:
        """
        计算指数Almon滞后权重
        
        公式: w(k; θ) = exp(θ1*k + θ2*k^2) / Σexp(θ1*j + θ2*j^2)
        
        Args:
            theta1: 一阶参数（控制权重衰减速度）
            theta2: 二阶参数（控制权重曲率）
            n_lags: 滞后阶数
            
        Returns:
            归一化权重数组
        """
        k = np.arange(1, n_lags + 1)
        # 计算未归一化权重
        unnorm_weights = np.exp(theta1 * k + theta2 * k**2)
        # 归一化
        weights = unnorm_weights / np.sum(unnorm_weights)
        return weights
    
    @staticmethod
    def gradient(theta1: float, theta2: float, n_lags: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算权重对参数的梯度（用于优化）
        """
        k = np.arange(1, n_lags + 1)
        w = ExponentialAlmonLags.weights(theta1, theta2, n_lags)
        
        # dw/dtheta1 = w * (k - sum(w*k))
        dw_dtheta1 = w * (k - np.sum(w * k))
        
        # dw/dtheta2 = w * (k^2 - sum(w*k^2))
        dw_dtheta2 = w * (k**2 - np.sum(w * k**2))
        
        return dw_dtheta1, dw_dtheta2


class MIDASModel:
    """
    MIDAS混频回归模型
    
    模型形式: y_t = β0 + β1 * B(L^(1/m); θ) * x_t^(m) + ε_t
    
    其中:
    - y_t: 低频目标变量（如季度GDP）
    - x_t^(m): 高频解释变量（如月度用电量）
    - B(L^(1/m); θ): 指数Almon滞后多项式
    - m: 频率比（如季度/月度=3）
    """
    
    def __init__(self, n_lags: int = 12, poly_type: str = 'exp_almon'):
        """
        Args:
            n_lags: 高频滞后阶数
            poly_type: 多项式类型 ('exp_almon')
        """
        self.n_lags = n_lags
        self.poly_type = poly_type
        self.params: Optional[MIDASParams] = None
        self.weights: Optional[np.ndarray] = None
        self.fitted_values: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
        
    def _aggregate_high_freq(self, x_high: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
    `    使用权重聚合高频数据
        
        Args:
            x_high: 高频数据数组
            weights: 滞后权重
            
        Returns:
            聚合后的低频数据
        """
        n_obs = len(x_high) - len(weights) + 1
        aggregated = np.zeros(n_obs)
        
        for i in range(n_obs):
            # 加权求和
            aggregated[i] = np.sum(x_high[i:i+len(weights)] * weights[::-1])
            
        return aggregated
    
    def _objective_function(self, params: np.ndarray, 
                           y_low: np.ndarray, 
                           x_high: np.ndarray) -> float:
        """
        目标函数（残差平方和）
        
        Args:
            params: [beta0, beta1, theta1, theta2]
            y_low: 低频目标变量
            x_high: 高频解释变量
            
        Returns:
            残差平方和
        """
        beta0, beta1, theta1, theta2 = params
        
        # 计算权重
        weights = ExponentialAlmonLags.weights(theta1, theta2, self.n_lags)
        
        # 聚合高频数据
        x_aggregated = self._aggregate_high_freq(x_high, weights)
        
        # 确保长度匹配
        min_len = min(len(y_low), len(x_aggregated))
        y_low = y_low[-min_len:]
        x_aggregated = x_aggregated[-min_len:]
        
        # 预测
        y_pred = beta0 + beta1 * x_aggregated
        
        # 残差平方和
        residuals = y_low - y_pred
        rss = np.sum(residuals**2)
        
        # 添加正则化防止过拟合
        rss += 0.001 * (theta1**2 + theta2**2)
        
        return rss
    
    def fit(self, y_low: pd.Series, x_high: pd.Series, 
            init_params: Optional[np.ndarray] = None) -> 'MIDASModel':
        """
        拟合MIDAS模型
        
        Args:
            y_low: 低频目标变量（如季度GDP）
            x_high: 高频解释变量（如月度用电量）
            init_params: 初始参数 [beta0, beta1, theta1, theta2]
            
        Returns:
            self
        """
        # 转换为numpy数组
        y_values = y_low.values if isinstance(y_low, pd.Series) else y_low
        x_values = x_high.values if isinstance(x_high, pd.Series) else x_high
        
        # 默认初始参数
        if init_params is None:
            init_params = np.array([np.mean(y_values), 0.5, -0.1, 0.01])
        
        # 参数约束
        bounds = [(None, None),  # beta0
                  (None, None),  # beta1
                  (-2, 2),       # theta1
                  (-0.5, 0.5)]   # theta2
        
        # 优化
        result = minimize(
            fun=self._objective_function,
            x0=init_params,
            args=(y_values, x_values),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        if result.success:
            self.params = MIDASParams(
                beta0=result.x[0],
                beta1=result.x[1],
                theta1=result.x[2],
                theta2=result.x[3]
            )
            self.weights = ExponentialAlmonLags.weights(
                self.params.theta1, self.params.theta2, self.n_lags
            )
            
            # 计算拟合值和残差
            x_aggregated = self._aggregate_high_freq(x_values, self.weights)
            min_len = min(len(y_values), len(x_aggregated))
            y_trimmed = y_values[-min_len:]
            x_trimmed = x_aggregated[-min_len:]
            
            self.fitted_values = self.params.beta0 + self.params.beta1 * x_trimmed
            self.residuals = y_trimmed - self.fitted_values
            
            print(f"✅ MIDAS模型拟合成功!")
            print(f"   参数: beta0={self.params.beta0:.4f}, beta1={self.params.beta1:.4f}")
            print(f"   参数: theta1={self.params.theta1:.4f}, theta2={self.params.theta2:.4f}")
            print(f"   RSS: {result.fun:.4f}")
        else:
            print(f"❌ 优化失败: {result.message}")
            
        return self
    
    def predict(self, x_high: pd.Series, horizon: int = 1) -> np.ndarray:
        """
        使用MIDAS模型进行预测
        
        Args:
            x_high: 高频解释变量
            horizon: 预测期数
            
        Returns:
            预测值
        """
        if self.params is None:
            raise ValueError("模型尚未拟合，请先调用fit()")
            
        x_values = x_high.values if isinstance(x_high, pd.Series) else x_high
        
        # 聚合高频数据
        x_aggregated = self._aggregate_high_freq(x_values, self.weights)
        
        # 预测
        predictions = self.params.beta0 + self.params.beta1 * x_aggregated
        
        return predictions[-horizon:]
    
    def nowcast(self, x_high: pd.Series, 
                available_data_ratio: float = 1.0) -> float:
        """
        现时预测（Nowcasting）
        使用部分可用的高频数据进行当季GDP预测
        
        Args:
            x_high: 高频解释变量
            available_data_ratio: 当前周期内可用数据比例（0-1）
            
        Returns:
            现时预测值
        """
        if self.params is None:
            raise ValueError("模型尚未拟合")
            
        x_values = x_high.values if isinstance(x_high, pd.Series) else x_high
        
        # 根据可用数据比例调整权重
        n_available = int(self.n_lags * available_data_ratio)
        adjusted_weights = self.weights.copy()
        
        if n_available < self.n_lags:
            # 将不可用部分的权重设为0，并重新归一化
            adjusted_weights[:self.n_lags - n_available] = 0
            adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
        
        # 使用调整后的权重聚合
        x_aggregated = self._aggregate_high_freq(x_values, adjusted_weights)
        
        return self.params.beta0 + self.params.beta1 * x_aggregated[-1]
    
    def get_model_summary(self) -> Dict:
        """
        获取模型摘要信息
        """
        if self.params is None:
            return {"error": "模型尚未拟合"}
            
        summary = {
            "parameters": {
                "beta0": self.params.beta0,
                "beta1": self.params.beta1,
                "theta1": self.params.theta1,
                "theta2": self.params.theta2
            },
            "weights": self.weights.tolist(),
            "goodness_of_fit": {
                "rmse": np.sqrt(np.mean(self.residuals**2)) if self.residuals is not None else None,
                "mae": np.mean(np.abs(self.residuals)) if self.residuals is not None else None,
                "r2": 1 - np.sum(self.residuals**2) / np.sum((self.fitted_values - np.mean(self.fitted_values))**2) \
                      if self.residuals is not None else None
            }
        }
        
        return summary


class MIDASEnsemble:
    """
    MIDAS模型集成
    使用多个高频指标构建集成预测
    """
    
    def __init__(self, n_lags: int = 12):
        self.n_lags = n_lags
        self.models: Dict[str, MIDASModel] = {}
        self.weights: Dict[str, float] = {}
        
    def add_model(self, name: str, model: MIDASModel, weight: float = 1.0):
        """添加子模型"""
        self.models[name] = model
        self.weights[name] = weight
        
    def fit(self, y_low: pd.Series, x_high_dict: Dict[str, pd.Series]):
        """
        拟合所有子模型
        
        Args:
            y_low: 低频目标变量
            x_high_dict: 高频变量字典 {名称: 序列}
        """
        for name, x_high in x_high_dict.items():
            print(f"\n📊 拟合MIDAS模型: {name}")
            model = MIDASModel(n_lags=self.n_lags)
            model.fit(y_low, x_high)
            self.models[name] = model
            
            # 根据模型拟合优度设置权重
            if model.residuals is not None:
                rmse = np.sqrt(np.mean(model.residuals**2))
                self.weights[name] = 1.0 / (rmse + 1e-6)
                
        # 归一化权重
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        print(f"\n✅ 集成模型权重: {self.weights}")
        
    def predict(self, x_high_dict: Dict[str, pd.Series]) -> float:
        """
        集成预测
        """
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if name in x_high_dict:
                pred = model.predict(x_high_dict[name], horizon=1)[0]
                predictions.append(pred)
                weights.append(self.weights[name])
                
        # 加权平均
        return np.average(predictions, weights=weights)


if __name__ == "__main__":
    # 测试MIDAS模型
    import sys
    sys.path.append('../..')
    from data.data_generator import MacroDataGenerator
    from data.data_processor import DataProcessor
    
    # 生成数据
    generator = MacroDataGenerator()
    data = generator.generate_all_data()
    
    processor = DataProcessor()
    processed = processor.preprocess_pipeline(data)
    
    # 准备数据
    gdp = processed['gdp']['gdp_value_clean']
    electricity = processed['monthly']['electricity']
    
    # 对齐数据
    gdp_aligned, elec_aligned = processor.align_frequencies(
        pd.DataFrame(gdp), pd.DataFrame(electricity),
        low_freq='Q', high_freq='M'
    )
    
    # 拟合MIDAS模型
    midas = MIDASModel(n_lags=12)
    midas.fit(gdp_aligned['gdp_value_clean'], elec_aligned['electricity'])
    
    # 查看模型摘要
    summary = midas.get_model_summary()
    print("\n📈 模型摘要:")
    print(f"   RMSE: {summary['goodness_of_fit']['rmse']:.4f}")
    print(f"   MAE: {summary['goodness_of_fit']['mae']:.4f}")
    print(f"   R²: {summary['goodness_of_fit']['r2']:.4f}")
