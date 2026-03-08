"""
数据清洗与预处理模块
包含：平稳性检验、季节性调整、缺失值处理、混频对齐
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.x13 import x13_arima_analysis
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    宏观经济数据预处理器
    """
    
    def __init__(self):
        self.processed_data = {}
        
    def adf_test(self, series: pd.Series, significance: float = 0.05) -> Dict:
        """
        ADF单位根检验（平稳性检验）
        
        Args:
            series: 时间序列
            significance: 显著性水平
            
        Returns:
            检验结果字典
        """
        result = adfuller(series.dropna(), autolag='AIC')
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < significance,
            'lag_used': result[2],
            'n_obs': result[3]
        }
    
    def kpss_test(self, series: pd.Series, significance: float = 0.05) -> Dict:
        """
        KPSS平稳性检验
        """
        result = kpss(series.dropna(), regression='c')
        
        return {
            'kpss_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[3],
            'is_stationary': result[1] > significance
        }
    
    def make_stationary(self, series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """
        自动差分使序列平稳
        
        Returns:
            (差分后序列, 差分阶数)
        """
        diff_count = 0
        current_series = series.copy()
        
        while diff_count < max_diff:
            adf_result = self.adf_test(current_series)
            if adf_result['is_stationary']:
                break
            current_series = current_series.diff().dropna()
            diff_count += 1
            
        return current_series, diff_count
    
    def seasonal_decompose_stl(self, series: pd.Series, period: int = 12) -> pd.DataFrame:
        """
        使用STL分解季节性
        
        Args:
            series: 时间序列
            period: 季节周期（月度=12，季度=4）
            
        Returns:
            分解结果DataFrame
        """
        stl = STL(series.dropna(), period=period, robust=True)
        result = stl.fit()
        
        return pd.DataFrame({
            'original': series,
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid,
            'deseasonalized': result.trend + result.resid
        })
    
    def remove_outliers(self, series: pd.Series, method: str = 'iqr', 
                       threshold: float = 3.0) -> pd.Series:
        """
        异常值检测与处理
        
        Args:
            series: 时间序列
            method: 方法 ('iqr', 'zscore', 'mad')
            threshold: 阈值
        """
        series_clean = series.copy()
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            outliers = pd.Series(z_scores > threshold, index=series.dropna().index)
            outliers = outliers.reindex(series.index).fillna(False)
            
        elif method == 'mad':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z = 0.6745 * (series - median) / mad
            outliers = np.abs(modified_z) > threshold
            
        # 用插值替换异常值
        series_clean[outliers] = np.nan
        series_clean = series_clean.interpolate(method='linear')
        
        return series_clean
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        缺失值处理
        
        Args:
            df: DataFrame
            method: 处理方法 ('interpolate', 'forward_fill', 'backward_fill', 'mean')
        """
        df_clean = df.copy()
        
        if method == 'interpolate':
            df_clean = df_clean.interpolate(method='linear')
            # 边界缺失值用前后填充
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        elif method == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill')
        elif method == 'backward_fill':
            df_clean = df_clean.fillna(method='bfill')
        elif method == 'mean':
            df_clean = df_clean.fillna(df_clean.mean())
            
        return df_clean
    
    def ragged_edge_process(self, df: pd.DataFrame, target_date: pd.Timestamp,
                           max_lag: int = 3) -> pd.DataFrame:
        """
        处理数据末端锯齿状缺失（Ragged Edge Problem）
        这是混频数据的关键问题
        
        Args:
            df: 高频数据DataFrame
            target_date: 目标预测日期
            max_lag: 最大滞后阶数
        """
        df_processed = df.copy()
        
        # 对于每个指标，检查在target_date之前的可用性
        for col in df.columns:
            series = df[col]
            # 找到最后一个非空值
            last_valid = series.last_valid_index()
            
            if last_valid is not None and last_valid < target_date:
                # 使用指数平滑进行短期外推
                n_missing = pd.date_range(last_valid, target_date, freq=df.index.freq).shape[0] - 1
                if n_missing > 0 and n_missing <= max_lag:
                    # 使用最后几个观测值的指数加权平均
                    recent_values = series.dropna().tail(5)
                    weights = np.exp(np.linspace(-1, 0, len(recent_values)))
                    weights /= weights.sum()
                    forecast_value = np.average(recent_values, weights=weights)
                    
                    # 填充缺失值
                    missing_dates = pd.date_range(last_valid + pd.Timedelta(days=1), 
                                                  target_date, 
                                                  freq=df.index.freq)
                    for i, date in enumerate(missing_dates):
                        if date in df_processed.index:
                            # 添加衰减的随机扰动
                            decay_factor = 0.9 ** (i + 1)
                            df_processed.loc[date, col] = forecast_value + \
                                np.random.normal(0, series.std() * 0.1 * decay_factor)
                            
        return df_processed
    
    def align_frequencies(self, low_freq_df: pd.DataFrame, 
                         high_freq_df: pd.DataFrame,
                         low_freq: str = 'Q', 
                         high_freq: str = 'M') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        对齐不同频率的数据
        
        Args:
            low_freq_df: 低频数据（如季度GDP）
            high_freq_df: 高频数据（如月度指标）
            low_freq: 低频类型
            high_freq: 高频类型
            
        Returns:
            (对齐后的低频数据, 对齐后的高频数据)
        """
        # 确保索引是datetime
        low_freq_df.index = pd.to_datetime(low_freq_df.index)
        high_freq_df.index = pd.to_datetime(high_freq_df.index)
        
        # 找到共同的时间范围
        start_date = max(low_freq_df.index.min(), high_freq_df.index.min())
        end_date = min(low_freq_df.index.max(), high_freq_df.index.max())
        
        # 截取共同时间段
        low_freq_aligned = low_freq_df[(low_freq_df.index >= start_date) & 
                                       (low_freq_df.index <= end_date)]
        high_freq_aligned = high_freq_df[(high_freq_df.index >= start_date) & 
                                         (high_freq_df.index <= end_date)]
        
        return low_freq_aligned, high_freq_aligned
    
    def preprocess_pipeline(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        完整的数据预处理流水线
        
        Args:
            data_dict: 原始数据字典
            
        Returns:
            处理后的数据字典
        """
        print("🔄 开始数据预处理...")
        processed = {}
        
        # 1. 处理GDP数据
        if 'gdp' in data_dict:
            gdp = data_dict['gdp'].copy()
            gdp['gdp_value_clean'] = self.remove_outliers(gdp['gdp_value'], method='iqr')
            gdp['gdp_value_clean'] = self.handle_missing_values(
                gdp[['gdp_value_clean']], method='interpolate'
            )['gdp_value_clean']
            processed['gdp'] = gdp
            print("   ✓ GDP数据清洗完成")
        
        # 2. 处理月度数据
        if 'monthly' in data_dict:
            monthly = data_dict['monthly'].copy()
            numeric_cols = monthly.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # 异常值处理
                monthly[col] = self.remove_outliers(monthly[col], method='iqr')
                # 缺失值处理
                monthly[col] = self.handle_missing_values(
                    monthly[[col]], method='interpolate'
                )[col]
                
            processed['monthly'] = monthly
            print("   ✓ 月度指标清洗完成")
        
        # 3. 处理日度数据
        if 'daily' in data_dict:
            daily = data_dict['daily'].copy()
            numeric_cols = daily.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                daily[col] = self.remove_outliers(daily[col], method='zscore', threshold=4)
                daily[col] = self.handle_missing_values(
                    daily[[col]], method='interpolate'
                )[col]
                
            processed['daily'] = daily
            print("   ✓ 日度金融数据清洗完成")
        
        # 4. 处理周度数据
        if 'weekly' in data_dict:
            weekly = data_dict['weekly'].copy()
            numeric_cols = weekly.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                weekly[col] = self.remove_outliers(weekly[col], method='iqr')
                weekly[col] = self.handle_missing_values(
                    weekly[[col]], method='interpolate'
                )[col]
                
            processed['weekly'] = weekly
            print("   ✓ 周度指标清洗完成")
        
        print("✅ 数据预处理完成!")
        return processed
    
    def generate_stationarity_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成平稳性检验报告
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = []
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 20:
                adf_result = self.adf_test(series)
                kpss_result = self.kpss_test(series)
                
                results.append({
                    'variable': col,
                    'adf_statistic': adf_result['adf_statistic'],
                    'adf_pvalue': adf_result['p_value'],
                    'adf_stationary': adf_result['is_stationary'],
                    'kpss_statistic': kpss_result['kpss_statistic'],
                    'kpss_pvalue': kpss_result['p_value'],
                    'kpss_stationary': kpss_result['is_stationary'],
                    'conclusion': 'Stationary' if (adf_result['is_stationary'] and 
                                                   kpss_result['is_stationary']) else 'Non-stationary'
                })
                
        return pd.DataFrame(results)


if __name__ == "__main__":
    # 测试数据预处理
    from data_generator import MacroDataGenerator
    
    generator = MacroDataGenerator()
    data = generator.generate_all_data()
    
    processor = DataProcessor()
    processed_data = processor.preprocess_pipeline(data)
    
    # 生成平稳性报告
    print("\n📊 月度指标平稳性检验报告:")
    report = processor.generate_stationarity_report(processed_data['monthly'])
    print(report.to_string())
