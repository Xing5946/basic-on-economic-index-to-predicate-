"""
DFM (Dynamic Factor Model) 动态因子模型
使用PCA提取共同因子，结合卡尔曼滤波处理缺失数据
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.linalg import solve_discrete_are
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DFMParams:
    """DFM模型参数"""
    n_factors: int  # 因子数量
    factor_order: int  # 因子自回归阶数
    error_order: int  # 误差自回归阶数


class KalmanFilterDFM:
    """
    用于DFM的卡尔曼滤波器
    处理数据末端缺失（Ragged Edge）问题
    """
    
    def __init__(self, n_factors: int, n_series: int):
        self.n_factors = n_factors
        self.n_series = n_series
        
        # 状态空间模型参数
        self.A = None  # 状态转移矩阵
        self.C = None  # 观测矩阵（因子载荷）
        self.Q = None  # 状态噪声协方差
        self.R = None  # 观测噪声协方差
        
        # 滤波结果
        self.filtered_states = None
        self.filtered_covs = None
        
    def initialize_params(self, factors: np.ndarray, loadings: np.ndarray):
        """
        初始化状态空间参数
        
        Args:
            factors: 提取的因子 (T x n_factors)
            loadings: 因子载荷矩阵 (n_series x n_factors)
        """
        T = factors.shape[0]
        
        # 观测矩阵 = 因子载荷
        self.C = loadings
        
        # 估计状态转移矩阵A（因子自回归系数）
        # 使用VAR(1)近似: F_t = A * F_{t-1} + u_t
        F_lag = factors[:-1]  # F_{t-1}
        F_curr = factors[1:]   # F_t
        
        # 最小二乘估计: A = (F'F)^{-1} F'F_lag
        self.A = np.linalg.lstsq(F_lag, F_curr, rcond=None)[0].T
        
        # 估计状态噪声协方差Q
        residuals = F_curr - F_lag @ self.A.T
        self.Q = np.cov(residuals.T)
        
        # 估计观测噪声协方差R
        predicted = factors @ self.C.T
        # 这里简化处理，实际应该使用完整的数据
        self.R = np.eye(self.n_series) * 0.1
        
    def filter(self, observations: np.ndarray, 
               missing_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        卡尔曼滤波
        
        Args:
            observations: 观测数据 (T x n_series)
            missing_mask: 缺失数据掩码 (T x n_series)，True表示缺失
            
        Returns:
            (滤波后的状态, 状态协方差)
        """
        T = observations.shape[0]
        
        # 初始化
        F_pred = np.zeros((T, self.n_factors))
        P_pred = np.zeros((T, self.n_factors, self.n_factors))
        F_filt = np.zeros((T, self.n_factors))
        P_filt = np.zeros((T, self.n_factors, self.n_factors))
        
        # 初始状态
        F_filt[0] = np.zeros(self.n_factors)
        P_filt[0] = np.eye(self.n_factors) * 10
        
        for t in range(1, T):
            # ===== 预测步骤 =====
            F_pred[t] = self.A @ F_filt[t-1]
            P_pred[t] = self.A @ P_filt[t-1] @ self.A.T + self.Q
            
            # ===== 更新步骤 =====
            if missing_mask is not None and np.any(missing_mask[t]):
                # 处理缺失数据：只使用可用的观测
                available = ~missing_mask[t]
                if np.any(available):
                    C_avail = self.C[available]
                    y_avail = observations[t, available]
                    R_avail = self.R[np.ix_(available, available)]
                    
                    # 卡尔曼增益
                    S = C_avail @ P_pred[t] @ C_avail.T + R_avail
                    K = P_pred[t] @ C_avail.T @ np.linalg.inv(S)
                    
                    # 状态更新
                    y_pred = C_avail @ F_pred[t]
                    F_filt[t] = F_pred[t] + K @ (y_avail - y_pred)
                    P_filt[t] = (np.eye(self.n_factors) - K @ C_avail) @ P_pred[t]
                else:
                    # 全部缺失，只使用预测
                    F_filt[t] = F_pred[t]
                    P_filt[t] = P_pred[t]
            else:
                # 完整观测
                S = self.C @ P_pred[t] @ self.C.T + self.R
                K = P_pred[t] @ self.C.T @ np.linalg.inv(S)
                
                y_pred = self.C @ F_pred[t]
                F_filt[t] = F_pred[t] + K @ (observations[t] - y_pred)
                P_filt[t] = (np.eye(self.n_factors) - K @ self.C) @ P_pred[t]
        
        self.filtered_states = F_filt
        self.filtered_covs = P_filt
        
        return F_filt, P_filt
    
    def smooth(self) -> np.ndarray:
        """
        卡尔曼平滑（Rauch-Tung-Striebel平滑器）
        """
        if self.filtered_states is None:
            raise ValueError("请先运行filter()")
            
        T = self.filtered_states.shape[0]
        F_smooth = self.filtered_states.copy()
        
        for t in range(T-2, -1, -1):
            # 平滑增益
            P_pred = self.A @ self.filtered_covs[t] @ self.A.T + self.Q
            J = self.filtered_covs[t] @ self.A.T @ np.linalg.inv(P_pred)
            
            # 平滑状态
            F_smooth[t] = self.filtered_states[t] + J @ (F_smooth[t+1] - self.A @ self.filtered_states[t])
            
        return F_smooth


class DFMModel:
    """
    动态因子模型
    
    模型结构:
    - 观测方程: X_t = Λ * F_t + ξ_t
    - 状态方程: F_t = A * F_{t-1} + u_t
    
    其中:
    - X_t: 观测变量向量
    - F_t: 共同因子向量
    - Λ: 因子载荷矩阵
    - A: 因子自回归矩阵
    """
    
    def __init__(self, n_factors: int = 3, factor_order: int = 1, 
                 error_order: int = 1):
        """
        Args:
            n_factors: 共同因子数量
            factor_order: 因子自回归阶数
            error_order: 误差自回归阶数
        """
        self.params = DFMParams(n_factors, factor_order, error_order)
        self.scaler = StandardScaler()
        self.pca = None
        self.kalman = None
        self.loadings = None
        self.factors = None
        self.target_loading = None
        
    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        数据预处理：标准化
        """
        # 处理缺失值（用均值填充用于PCA）
        df_filled = df.fillna(df.mean())
        
        # 标准化
        X_scaled = self.scaler.fit_transform(df_filled)
        
        return X_scaled
    
    def extract_factors_pca(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用PCA提取初始因子
        
        Returns:
            (因子矩阵, 因子载荷矩阵)
        """
        self.pca = PCA(n_components=self.params.n_factors)
        factors = self.pca.fit_transform(X)
        loadings = self.pca.components_.T
        
        return factors, loadings
    
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """
        拟合DFM模型
        
        Args:
            df: 多变量时间序列DataFrame
            target_col: 目标变量列名（用于预测）
        """
        print(f"🔄 拟合DFM模型 (n_factors={self.params.n_factors})...")
        
        # 1. 数据预处理
        X_scaled = self.preprocess_data(df)
        
        # 2. PCA提取初始因子
        factors_init, self.loadings = self.extract_factors_pca(X_scaled)
        
        # 3. 初始化卡尔曼滤波器
        self.kalman = KalmanFilterDFM(
            self.params.n_factors, 
            X_scaled.shape[1]
        )
        self.kalman.initialize_params(factors_init, self.loadings)
        
        # 4. 使用卡尔曼滤波处理缺失数据
        missing_mask = df.isna().values
        self.factors, _ = self.kalman.filter(X_scaled, missing_mask)
        
        # 5. 如果指定了目标变量，估计目标方程
        if target_col and target_col in df.columns:
            target_idx = df.columns.get_loc(target_col)
            self.target_loading = self.loadings[target_idx]
            
        print(f"✅ DFM模型拟合完成!")
        print(f"   解释的方差比例: {self.pca.explained_variance_ratio_.sum():.2%}")
        print(f"   各因子解释方差: {self.pca.explained_variance_ratio_}")
        
        return self
    
    def predict_target(self, steps_ahead: int = 1) -> np.ndarray:
        """
        预测目标变量
        
        Args:
            steps_ahead: 预测步数
            
        Returns:
            预测值
        """
        if self.target_loading is None:
            raise ValueError("未指定目标变量")
            
        if self.factors is None:
            raise ValueError("模型尚未拟合")
            
        # 使用最后的状态预测未来因子
        last_factor = self.factors[-1]
        predictions = []
        
        current_factor = last_factor
        for _ in range(steps_ahead):
            # 状态预测: F_{t+1} = A * F_t
            current_factor = self.kalman.A @ current_factor
            # 观测预测: y = Λ' * F
            y_pred = self.target_loading @ current_factor
            predictions.append(y_pred)
            
        # 反标准化
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_orig = self.scaler.inverse_transform(
            np.zeros((steps_ahead, self.loadings.shape[0]))
        )
        
        # 这里简化处理，实际应该只反标准化目标变量
        return np.array(predictions)
    
    def get_factor_df(self) -> pd.DataFrame:
        """
        获取因子DataFrame
        """
        if self.factors is None:
            raise ValueError("模型尚未拟合")
            
        factor_cols = [f'Factor_{i+1}' for i in range(self.params.n_factors)]
        return pd.DataFrame(self.factors, columns=factor_cols)
    
    def get_factor_loadings_df(self, feature_names: List[str]) -> pd.DataFrame:
        """
        获取因子载荷DataFrame
        """
        if self.loadings is None:
            raise ValueError("模型尚未拟合")
            
        factor_cols = [f'Factor_{i+1}' for i in range(self.params.n_factors)]
        return pd.DataFrame(
            self.loadings, 
            columns=factor_cols,
            index=feature_names
        )
    
    def get_model_summary(self) -> Dict:
        """
        获取模型摘要
        """
        if self.pca is None:
            return {"error": "模型尚未拟合"}
            
        return {
            "n_factors": self.params.n_factors,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(self.pca.explained_variance_ratio_).tolist(),
            "singular_values": self.pca.singular_values_.tolist(),
        }


if __name__ == "__main__":
    # 测试DFM模型
    import sys
    sys.path.append('../..')
    from data.data_generator import MacroDataGenerator
    from data.data_processor import DataProcessor
    
    # 生成数据
    generator = MacroDataGenerator()
    data = generator.generate_all_data()
    
    processor = DataProcessor()
    processed = processor.preprocess_pipeline(data)
    
    # 使用月度数据拟合DFM
    monthly = processed['monthly'].select_dtypes(include=[np.number])
    
    # 拟合DFM
    dfm = DFMModel(n_factors=3)
    dfm.fit(monthly)
    
    # 查看因子
    factors_df = dfm.get_factor_df()
    print("\n📊 提取的共同因子:")
    print(factors_df.head(10))
    
    # 模型摘要
    summary = dfm.get_model_summary()
    print(f"\n📈 模型摘要:")
    print(f"   累计解释方差: {summary['cumulative_variance'][-1]:.2%}")