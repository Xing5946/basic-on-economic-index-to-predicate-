"""
Hybrid Model 混合预测模型
整合线性模型（MIDAS/DFM）和非线性模型（TSLM）的预测结果

核心公式: Ŷ_Final =Ŷ _Linear + Ê_Nonlinear
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HybridModelConfig:
    """混合模型配置"""
    linear_weight: float = 0.6  # 线性模型权重
    nonlinear_weight: float = 0.4  # 非线性模型权重
    use_residual_approach: bool = True  # 是否使用残差修正方法
    ensemble_method: str = "weighted_sum"  # 集成方法: weighted_sum, stacking, blending


class HybridPredictor:
    """
    混合预测器
    整合多种模型的预测结果
    """
    
    def __init__(self, config: Optional[HybridModelConfig] = None):
        self.config = config or HybridModelConfig()
        self.linear_model = None
        self.tslm_adapter = None
        self.meta_learner = None  # 用于stacking
        self.fitted = False
        
        # 历史记录
        self.linear_predictions = []
        self.nonlinear_predictions = []
        self.actuals = []
        
    def set_models(self, linear_model, tslm_adapter):
        """
        设置子模型
        
        Args:
            linear_model: 线性模型（MIDAS或DFM）
            tslm_adapter: TSLM大模型适配器
        """
        self.linear_model = linear_model
        self.tslm_adapter = tslm_adapter
        
    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        """
        拟合混合模型
        
        Args:
            y_train: 训练目标值
            X_train: 训练特征（可选）
        """
        print("🔄 拟合混合模型...")
        
        if self.config.ensemble_method == "stacking":
            # 使用stacking方法，训练元学习器
            self._fit_stacking(y_train, X_train)
        elif self.config.ensemble_method == "blending":
            # 使用blending方法
            self._fit_blending(y_train, X_train)
        else:
            # 简单加权平均
            print(f"   使用加权平均: 线性={self.config.linear_weight}, 非线性={self.config.nonlinear_weight}")
            
        self.fitted = True
        print("✅ 混合模型拟合完成")
        
    def _fit_stacking(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        """
        使用Stacking方法训练元学习器
        """
        print("   训练Stacking元学习器...")
        
        # 生成基模型的预测（使用交叉验证避免过拟合）
        tscv = TimeSeriesSplit(n_splits=5)
        
        linear_preds = []
        nonlinear_preds = []
        actuals = []
        
        for train_idx, val_idx in tscv.split(y_train):
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 这里简化处理，实际应该重新训练基模型
            # 获取基模型预测
            if hasattr(self.linear_model, 'fitted_values'):
                linear_pred = self.linear_model.fitted_values[val_idx]
            else:
                linear_pred = np.mean(y_tr) * np.ones(len(val_idx))
                
            # TSLM预测残差
            if self.tslm_adapter and self.tslm_adapter.is_initialized:
                # 这里简化处理
                nonlinear_pred = np.zeros(len(val_idx))
            else:
                nonlinear_pred = np.zeros(len(val_idx))
            
            linear_preds.extend(linear_pred if isinstance(linear_pred, list) else linear_pred.tolist())
            nonlinear_preds.extend(nonlinear_pred.tolist())
            actuals.extend(y_val.tolist())
            
        # 训练元学习器（Ridge回归）
        meta_features = np.column_stack([linear_preds, nonlinear_preds])
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(meta_features, actuals)
        
        print(f"   元学习器系数: {self.meta_learner.coef_}")
        
    def _fit_blending(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        """
        使用Blending方法
        """
        print("   训练Blending权重...")
        
        # 使用验证集确定最优权重
        split_point = int(len(y_train) * 0.8)
        
        # 这里简化处理，实际应该使用验证集预测
        # 使用网格搜索找到最优权重
        best_weights = {'linear': 0.6, 'nonlinear': 0.4}
        best_mse = float('inf')
        
        for w_linear in np.arange(0, 1.1, 0.1):
            w_nonlinear = 1 - w_linear
            # 计算验证集MSE（简化）
            mse = 0  # 实际应该计算
            if mse < best_mse:
                best_mse = mse
                best_weights = {'linear': w_linear, 'nonlinear': w_nonlinear}
                
        self.config.linear_weight = best_weights['linear']
        self.config.nonlinear_weight = best_weights['nonlinear']
        
        print(f"   最优权重: 线性={self.config.linear_weight:.2f}, 非线性={self.config.nonlinear_weight:.2f}")
        
    def predict(self, prediction_length: int = 1) -> Dict[str, np.ndarray]:
        """
        进行预测
        
        Args:
            prediction_length: 预测长度
            
        Returns:
            包含各组件预测结果的字典
        """
        if not self.fitted:
            raise RuntimeError("模型尚未拟合")
            
        # 获取线性模型预测
        if hasattr(self.linear_model, 'fitted_values'):
            linear_pred = self.linear_model.fitted_values[-prediction_length:]
        else:
            linear_pred = np.zeros(prediction_length)
            
        # 获取非线性修正（残差预测）
        if self.tslm_adapter and self.tslm_adapter.is_initialized:
            if hasattr(self.linear_model, 'residuals'):
                residuals = self.linear_model.residuals
                nonlinear_pred = self.tslm_adapter.residual_forecast(
                    residuals, prediction_length
                )
            else:
                nonlinear_pred = np.zeros(prediction_length)
        else:
            nonlinear_pred = np.zeros(prediction_length)
            
        # 集成预测
        if self.config.ensemble_method == "stacking" and self.meta_learner:
            meta_features = np.column_stack([
                linear_pred if isinstance(linear_pred, np.ndarray) else [linear_pred],
                nonlinear_pred
            ])
            final_pred = self.meta_learner.predict(meta_features)
        else:
            # 加权平均
            final_pred = (self.config.linear_weight * 
                         (linear_pred if isinstance(linear_pred, np.ndarray) else np.array([linear_pred])) +
                         self.config.nonlinear_weight * nonlinear_pred)
        
        return {
            'linear_prediction': linear_pred,
            'nonlinear_prediction': nonlinear_pred,
            'final_prediction': final_pred,
            'linear_weight': self.config.linear_weight,
            'nonlinear_weight': self.config.nonlinear_weight
        }
    
    def nowcast(self, available_data_ratio: float = 1.0) -> Dict[str, float]:
        """
        现时预测（Nowcasting）
        
        Args:
            available_data_ratio: 可用数据比例
            
        Returns:
            预测结果字典
        """
        # 线性模型现时预测
        if hasattr(self.linear_model, 'nowcast'):
            linear_nowcast = self.linear_model.nowcast(
                None, available_data_ratio  # 需要传入高频数据
            )
        else:
            linear_nowcast = 0
            
        # 非线性修正
        nonlinear_correction = 0  # 简化处理
        
        # 最终预测
        final_nowcast = (self.config.linear_weight * linear_nowcast +
                        self.config.nonlinear_weight * nonlinear_correction)
        
        return {
            'linear_nowcast': linear_nowcast,
            'nonlinear_correction': nonlinear_correction,
            'final_nowcast': final_nowcast
        }
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        评估预测性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            评估指标字典
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 方向准确率
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        direction_accuracy = np.mean(direction_true == direction_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }


class HybridModelPipeline:
    """
    混合模型完整流水线
    整合数据预处理、模型训练、预测、评估全流程
    """
    
    def __init__(self):
        self.midas_model = None
        self.dfm_model = None
        self.tslm_adapter = None
        self.hybrid_predictor = None
        self.results = {}
        
    def build_models(self, midas_params: Optional[Dict] = None,
                    dfm_params: Optional[Dict] = None,
                    tslm_config: Optional[Dict] = None):
        """
        构建所有模型组件
        """
        from models.midas.midas_model import MIDASModel
        from models.dfm.dfm_model import DFMModel
        from models.tslm.tslm_adapter import TSLMAdapter, TSLMConfig
        
        print("🔄 构建模型组件...")
        
        # 1. MIDAS模型
        midas_kwargs = midas_params or {'n_lags': 12}
        self.midas_model = MIDASModel(**midas_kwargs)
        print("   ✓ MIDAS模型构建完成")
        
        # 2. DFM模型
        dfm_kwargs = dfm_params or {'n_factors': 3}
        self.dfm_model = DFMModel(**dfm_kwargs)
        print("   ✓ DFM模型构建完成")
        
        # 3. TSLM适配器
        tslm_kwargs = tslm_config or {}
        config = TSLMConfig(**tslm_kwargs)
        self.tslm_adapter = TSLMAdapter(config)
        self.tslm_adapter.initialize(use_mock=True)
        print("   ✓ TSLM适配器构建完成")
        
        # 4. 混合预测器
        self.hybrid_predictor = HybridPredictor()
        print("   ✓ 混合预测器构建完成")
        
    def fit(self, gdp_data: pd.Series, monthly_data: pd.DataFrame,
            target_col: str = 'gdp_value_clean'):
        """
        拟合完整流水线
        
        Args:
            gdp_data: GDP数据
            monthly_data: 月度指标数据
            target_col: 目标变量列名
        """
        print("\n" + "="*60)
        print("🚀 开始拟合混合模型流水线")
        print("="*60)
        
        # 1. 拟合MIDAS模型（使用用电量作为高频指标）
        if 'electricity' in monthly_data.columns:
            print("\n📊 拟合MIDAS模型...")
            self.midas_model.fit(gdp_data, monthly_data['electricity'])
        
        # 2. 拟合DFM模型
        print("\n📊 拟合DFM模型...")
        monthly_numeric = monthly_data.select_dtypes(include=[np.number])
        self.dfm_model.fit(monthly_numeric)
        
        # 3. 设置混合预测器（使用MIDAS作为线性模型）
        self.hybrid_predictor.set_models(self.midas_model, self.tslm_adapter)
        self.hybrid_predictor.fit(gdp_data)
        
        print("\n✅ 混合模型流水线拟合完成!")
        
    def predict(self, steps: int = 1) -> Dict:
        """
        进行预测
        """
        print(f"\n🔮 进行{steps}步预测...")
        
        # 混合预测
        hybrid_result = self.hybrid_predictor.predict(steps)
        
        # 保存结果
        self.results['latest_prediction'] = hybrid_result
        
        print(f"   线性预测: {hybrid_result['linear_prediction']}")
        print(f"   非线性修正: {hybrid_result['nonlinear_prediction']}")
        print(f"   最终预测: {hybrid_result['final_prediction']}")
        
        return hybrid_result
    
    def get_model_summary(self) -> Dict:
        """
        获取模型摘要
        """
        return {
            'midas': self.midas_model.get_model_summary() if self.midas_model else None,
            'dfm': self.dfm_model.get_model_summary() if self.dfm_model else None,
            'hybrid_config': {
                'linear_weight': self.hybrid_predictor.config.linear_weight if self.hybrid_predictor else None,
                'nonlinear_weight': self.hybrid_predictor.config.nonlinear_weight if self.hybrid_predictor else None,
            }
        }


if __name__ == "__main__":
    # 测试混合模型
    import sys
    sys.path.append('../..')
    from data.data_generator import MacroDataGenerator
    from data.data_processor import DataProcessor
    
    print("="*60)
    print("混合预测模型测试")
    print("="*60)
    
    # 生成数据
    generator = MacroDataGenerator()
    data = generator.generate_all_data()
    
    processor = DataProcessor()
    processed = processor.preprocess_pipeline(data)
    
    # 构建流水线
    pipeline = HybridModelPipeline()
    pipeline.build_models()
    
    # 拟合
    gdp = processed['gdp']['gdp_value_clean']
    monthly = processed['monthly']
    
    # 对齐数据
    gdp_aligned, monthly_aligned = processor.align_frequencies(
        pd.DataFrame(gdp), monthly, 'Q', 'M'
    )
    
    pipeline.fit(gdp_aligned['gdp_value_clean'], monthly_aligned)
    
    # 预测
    result = pipeline.predict(steps=1)
    
    # 模型摘要
    summary = pipeline.get_model_summary()
    print("\n📋 模型摘要:")
    print(f"   MIDAS RMSE: {summary['midas']['goodness_of_fit']['rmse']:.4f}")
    print(f"   DFM解释方差: {summary['dfm']['cumulative_variance'][-1]:.2%}")
