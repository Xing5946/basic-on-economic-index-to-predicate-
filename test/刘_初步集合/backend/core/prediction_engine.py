"""
预测引擎核心模块
整合所有模型，提供统一的预测接口
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import pickle

from models.midas.midas_model import MIDASModel, MIDASEnsemble
from models.dfm.dfm_model import DFMModel
from models.tslm.tslm_adapter import TSLMAdapter, TSLMConfig
from models.hybrid.hybrid_model import HybridPredictor, HybridModelConfig
from data.data_generator import MacroDataGenerator
from data.data_processor import DataProcessor


class PredictionEngine:
    """
    预测引擎
    整合MIDAS、DFM、TSLM模型，提供预测服务
    """
    
    def __init__(self):
        self.midas_model = None
        self.dfm_model = None
        self.tslm_adapter = None
        self.hybrid_predictor = None
        
        self.data = {}
        self.processed_data = {}
        self.is_initialized = False
        
        # 预测历史
        self.prediction_history = []
        
    def initialize(self, use_mock_data: bool = True):
        """
        初始化预测引擎
        
        Args:
            use_mock_data: 是否使用模拟数据
        """
        print("🔄 初始化预测引擎...")
        
        # 1. 加载数据
        if use_mock_data:
            self._load_mock_data()
        else:
            self._load_real_data()
            
        # 2. 预处理数据
        self._preprocess_data()
        
        # 3. 构建和训练模型
        self._build_models()
        
        self.is_initialized = True
        print("✅ 预测引擎初始化完成")
        
    def _load_mock_data(self):
        """加载模拟数据"""
        print("   加载模拟数据...")
        generator = MacroDataGenerator()
        self.data = generator.generate_all_data()
        
    def _load_real_data(self):
        """加载真实数据（实际部署时实现）"""
        print("   加载真实数据...")
        # 从数据库或文件加载
        # 这里简化处理，使用模拟数据
        self._load_mock_data()
        
    def _preprocess_data(self):
        """预处理数据"""
        print("   预处理数据...")
        processor = DataProcessor()
        self.processed_data = processor.preprocess_pipeline(self.data)
        
    def _build_models(self):
        """构建和训练模型"""
        print("   构建模型...")
        
        # 准备数据
        gdp = self.processed_data['gdp']['gdp_value_clean']
        monthly = self.processed_data['monthly']
        
        # 对齐数据
        processor = DataProcessor()
        gdp_aligned, monthly_aligned = processor.align_frequencies(
            pd.DataFrame(gdp), monthly, 'Q', 'M'
        )
        
        # 1. MIDAS模型
        print("      训练MIDAS模型...")
        self.midas_model = MIDASModel(n_lags=12)
        if 'electricity' in monthly_aligned.columns:
            self.midas_model.fit(
                gdp_aligned['gdp_value_clean'], 
                monthly_aligned['electricity']
            )
        
        # 2. DFM模型
        print("      训练DFM模型...")
        monthly_numeric = monthly_aligned.select_dtypes(include=[np.number])
        self.dfm_model = DFMModel(n_factors=3)
        self.dfm_model.fit(monthly_numeric)
        
        # 3. TSLM适配器
        print("      初始化TSLM适配器...")
        config = TSLMConfig(model_name="timesfm", context_length=128, prediction_length=12)
        self.tslm_adapter = TSLMAdapter(config)
        self.tslm_adapter.initialize(use_mock=True)
        
        # 4. 混合预测器
        print("      构建混合预测器...")
        hybrid_config = HybridModelConfig(
            linear_weight=0.6,
            nonlinear_weight=0.4,
            use_residual_approach=True
        )
        self.hybrid_predictor = HybridPredictor(hybrid_config)
        self.hybrid_predictor.set_models(self.midas_model, self.tslm_adapter)
        self.hybrid_predictor.fit(gdp_aligned['gdp_value_clean'])
        
    def predict(self, target: str = "gdp", horizon: int = 1, 
                use_hybrid: bool = True) -> Dict[str, Any]:
        """
        进行预测
        
        Args:
            target: 预测目标
            horizon: 预测期数
            use_hybrid: 是否使用混合模型
            
        Returns:
            预测结果字典
        """
        if not self.is_initialized:
            raise RuntimeError("预测引擎尚未初始化")
            
        if use_hybrid and self.hybrid_predictor:
            result = self.hybrid_predictor.predict(horizon)
        else:
            # 仅使用线性模型
            linear_pred = self.midas_model.predict(
                self.processed_data['monthly']['electricity'], 
                horizon
            )
            result = {
                'linear_prediction': linear_pred,
                'nonlinear_correction': np.zeros(horizon),
                'final_prediction': linear_pred,
                'model_weights': {'linear': 1.0, 'nonlinear': 0.0}
            }
            
        # 记录预测历史
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'target': target,
            'horizon': horizon,
            'result': result
        }
        self.prediction_history.append(prediction_record)
        
        return result
        
    def nowcast(self, target: str = "gdp") -> Dict[str, Any]:
        """
        现时预测
        
        Returns:
            现时预测结果
        """
        if not self.is_initialized:
            raise RuntimeError("预测引擎尚未初始化")
            
        # 获取当前季度
        now = datetime.now()
        current_quarter = f"{now.year}Q{(now.month-1)//3 + 1}"
        
        # 计算数据可用性
        monthly_data = self.processed_data['monthly']
        last_month = monthly_data.index[-1]
        days_in_quarter = 90
        days_elapsed = (now - last_month).days
        data_availability = {
            'overall': min(days_elapsed / days_in_quarter, 1.0),
            'electricity': 0.85,
            'industrial': 0.80,
            'retail': 0.75
        }
        
        # 进行预测
        prediction = self.predict(target, horizon=1, use_hybrid=True)
        
        return {
            'current_quarter': current_quarter,
            'linear_nowcast': float(prediction['linear_prediction'][0]),
            'nonlinear_correction': float(prediction['nonlinear_prediction'][0]),
            'final_nowcast': float(prediction['final_prediction'][0]),
            'data_availability': data_availability
        }
        
    def get_prediction_history(self, target: str = "gdp", 
                               n_periods: int = 10) -> List[Dict]:
        """
        获取预测历史
        """
        return self.prediction_history[-n_periods:]
        
    def get_status(self) -> Dict[str, Any]:
        """
        获取引擎状态
        """
        return {
            "status": "ready" if self.is_initialized else "initializing",
            "models_loaded": {
                "midas": self.midas_model is not None,
                "dfm": self.dfm_model is not None,
                "tslm": self.tslm_adapter is not None and self.tslm_adapter.is_initialized
            },
            "last_trained": datetime.now().isoformat() if self.is_initialized else None,
            "performance_metrics": self._get_performance_metrics()
        }
        
    def _get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        if self.midas_model and self.midas_model.residuals is not None:
            residuals = self.midas_model.residuals
            return {
                "midas_rmse": float(np.sqrt(np.mean(residuals**2))),
                "midas_mae": float(np.mean(np.abs(residuals)))
            }
        return {}
        
    def get_performance_metrics(self, target: str = "gdp", 
                                metric: str = "rmse") -> Dict[str, Any]:
        """
        获取详细性能指标
        """
        metrics = self._get_performance_metrics()
        return {
            "target": target,
            "metric": metric,
            "value": metrics.get(f"midas_{metric}", 0),
            "all_metrics": metrics
        }
        
    def get_factor_analysis(self) -> Dict[str, Any]:
        """
        获取因子分析结果
        """
        if not self.dfm_model:
            return {"error": "DFM模型未加载"}
            
        summary = self.dfm_model.get_model_summary()
        factors_df = self.dfm_model.get_factor_df()
        
        return {
            "n_factors": summary["n_factors"],
            "explained_variance_ratio": summary["explained_variance_ratio"],
            "cumulative_variance": summary["cumulative_variance"],
            "factors_preview": factors_df.tail(5).to_dict()
        }
        
    def get_attribution(self, target: str = "gdp", 
                        date: Optional[str] = None) -> Dict[str, Any]:
        """
        获取预测归因分析
        """
        # 简化实现
        return {
            "target": target,
            "date": date or datetime.now().isoformat(),
            "attributions": {
                "electricity": 0.35,
                "industrial_output": 0.25,
                "retail_sales": 0.20,
                "fixed_investment": 0.15,
                "other": 0.05
            }
        }
        
    def get_midas_weights(self) -> Dict[str, Any]:
        """
        获取MIDAS滞后权重
        """
        if not self.midas_model or self.midas_model.weights is None:
            return {"error": "MIDAS模型未训练"}
            
        weights = self.midas_model.weights
        return {
            "n_lags": len(weights),
            "weights": weights.tolist(),
            "weight_distribution": {
                "recent_3m": float(np.sum(weights[-3:])),
                "recent_6m": float(np.sum(weights[-6:])),
                "recent_12m": float(np.sum(weights))
            }
        }
        
    def retrain(self):
        """
        重新训练所有模型
        """
        print("🔄 重新训练模型...")
        self._build_models()
        print("✅ 模型重训练完成")
