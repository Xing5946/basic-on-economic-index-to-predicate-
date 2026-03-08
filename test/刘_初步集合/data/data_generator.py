"""
宏观经济数据生成器
生成模拟的省级宏观经济数据用于demo演示
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MacroDataGenerator:
    """
    宏观经济数据生成器
    生成包含季度GDP、月度指标、日度金融数据的混频数据集
    """
    
    def __init__(self, start_date: str = "2015-01-01", end_date: str = "2024-12-31", 
                 province: str = "DemoProvince", random_seed: int = 42):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.province = province
        np.random.seed(random_seed)
        
        # 基础参数设置（模拟真实经济特征）
        self.gdp_trend = 0.015  # 季度GDP趋势增长率
        self.gdp_volatility = 0.008  # GDP波动率
        
    def generate_gdp(self) -> pd.DataFrame:
        """
        生成季度GDP数据（目标变量）
        包含趋势、周期、季节性成分
        """
        # 生成季度日期
        quarters = pd.date_range(start=self.start_date, end=self.end_date, freq='QE')
        n_quarters = len(quarters)
        
        # 趋势成分,是numpy.ndarray类型
        trend = 100 * (1 + self.gdp_trend) ** np.arange(n_quarters)
        
        # 周期成分（模拟经济周期）,同上
        cycle = 3 * np.sin(2 * np.pi * np.arange(n_quarters) / 40) + \
                1.5 * np.sin(2 * np.pi * np.arange(n_quarters) / 12)
        
        # 季节性成分
        seasonal = np.tile([0, 1.5, -0.5, -1.0], n_quarters // 4 + 1)[:n_quarters]
        
        # 随机冲击
        noise = np.random.normal(0, self.gdp_volatility * 100, n_quarters)
        
        # 合成GDP
        gdp_value = trend + cycle + seasonal + noise
        gdp_yoy = np.diff(gdp_value, 4) / gdp_value[:-4] * 100
        gdp_yoy = np.concatenate([[np.nan] * 4, gdp_yoy])
        
        df = pd.DataFrame({
            'date': quarters,
            'gdp_value': gdp_value,
            'gdp_yoy': gdp_yoy,
            'province': self.province
        })
        df.set_index('date', inplace=True)
        return df
    
    def generate_monthly_indicators(self, gdp: pd.DataFrame) -> pd.DataFrame:
        """
        生成月度宏观经济指标
        包括：工业增加值、社消零、固投、用电量等
        """
        months = pd.date_range(start=self.start_date, end=self.end_date, freq='ME')
        n_months = len(months)
        
        # 将季度GDP插值为月度（用于生成相关指标）
        gdp_monthly = gdp.resample('ME').interpolate(method='linear')
        gdp_values = gdp_monthly['gdp_value'].reindex(months).fillna(method='ffill')
        
        indicators = {}
        
        # 1. 工业增加值（领先GDP约1个月，相关系数高）
        indicators['industrial_output'] = (
            gdp_values * 0.35 + 
            np.random.normal(0, 2, n_months) +
            np.sin(2 * np.pi * np.arange(n_months) / 12) * 3
        )
        
        # 2. 社会消费品零售总额
        indicators['retail_sales'] = (
            gdp_values * 0.42 + 
            np.random.normal(0, 1.5, n_months) +
            np.sin(2 * np.pi * np.arange(n_months) / 12 + np.pi/4) * 5  # 春节效应
        )
        
        # 3. 固定资产投资
        indicators['fixed_investment'] = (
            gdp_values * 0.55 + 
            np.random.normal(0, 3, n_months) +
            np.sin(2 * np.pi * np.arange(n_months) / 12 + np.pi/2) * 2
        )
        
        # 4. 全社会用电量（高频先行指标）
        indicators['electricity'] = (
            gdp_values * 0.28 + 
            np.random.normal(0, 1.8, n_months) +
            np.sin(2 * np.pi * np.arange(n_months) / 12) * 8  # 强季节性
        )
        
        # 5. PMI采购经理指数
        indicators['pmi'] = 50 + (
            (gdp_values.pct_change().fillna(0) * 500) + 
            np.random.normal(0, 1.5, n_months)
        ).clip(-10, 10)
        
        # 6. CPI消费者价格指数
        indicators['cpi'] = 100 + np.cumsum(np.random.normal(0.02, 0.3, n_months))
        
        # 7. PPI生产者价格指数
        indicators['ppi'] = 100 + np.cumsum(np.random.normal(0.01, 0.5, n_months)) + \
                           np.sin(2 * np.pi * np.arange(n_months) / 36) * 5
        
        # 8. 出口总额
        indicators['exports'] = (
            gdp_values * 0.15 + 
            np.random.normal(0, 2, n_months) +
            np.sin(2 * np.pi * np.arange(n_months) / 12 + np.pi) * 3
        )
        
        # 9. 信贷投放
        indicators['credit'] = (
            gdp_values * 0.45 + 
            np.random.normal(0, 4, n_months) +
            np.sin(2 * np.pi * np.arange(n_months) / 3) * 2  # 季末冲量
        )
        
        df = pd.DataFrame(indicators, index=months)
        df['province'] = self.province
        return df
    
    def generate_daily_financial(self) -> pd.DataFrame:
        """
        生成日度金融数据
        包括：股指、利率、汇率等
        """
        days = pd.date_range(start=self.start_date, end=self.end_date, freq='B')  # 工作日
        n_days = len(days)
        
        financial = {}
        
        # 1. 股票指数（随机游走+趋势）
        returns = np.random.normal(0.0002, 0.015, n_days)
        financial['stock_index'] = 3000 * np.exp(np.cumsum(returns))
        
        # 2. 10年期国债收益率
        financial['bond_yield_10y'] = 3.0 + np.cumsum(np.random.normal(0, 0.02, n_days)) + \
                                     np.sin(2 * np.pi * np.arange(n_days) / 252) * 0.5
        
        # 3. 银行间同业拆借利率
        financial['interbank_rate'] = 2.5 + np.random.normal(0, 0.3, n_days) + \
                                     np.sin(2 * np.pi * np.arange(n_days) / 21) * 0.3  # 月度波动
        
        # 4. 汇率
        financial['exchange_rate'] = 6.8 + np.cumsum(np.random.normal(0, 0.005, n_days))
        
        # 5. 成交量
        financial['trading_volume'] = np.exp(15 + np.random.normal(0, 0.3, n_days))
        
        df = pd.DataFrame(financial, index=days)
        df['province'] = self.province
        return df
    
    def generate_weekly_indicators(self) -> pd.DataFrame:
        """
        生成周度指标
        如：生产资料价格、航运指数等
        """
        weeks = pd.date_range(start=self.start_date, end=self.end_date, freq='W')
        n_weeks = len(weeks)
        
        weekly = {}
        
        # 1. 生产资料价格指数
        weekly['material_price'] = 100 + np.cumsum(np.random.normal(0.01, 0.8, n_weeks))
        
        # 2. 物流运价指数
        weekly['logistics_index'] = 1000 + np.random.normal(0, 50, n_weeks) + \
                                   np.sin(2 * np.pi * np.arange(n_weeks) / 52) * 100
        
        # 3. 房地产成交面积
        weekly['property_volume'] = np.exp(10 + np.random.normal(0, 0.2, n_weeks) + \
                                          np.sin(2 * np.pi * np.arange(n_weeks) / 52) * 0.3)
        
        df = pd.DataFrame(weekly, index=weeks)
        df['province'] = self.province
        return df
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        生成完整的混频数据集
        """
        print("🔄 正在生成宏观经济数据...")
        
        # 先生成GDP（目标变量）
        gdp = self.generate_gdp()
        
        # 生成其他频率数据
        monthly = self.generate_monthly_indicators(gdp)
        daily = self.generate_daily_financial()
        weekly = self.generate_weekly_indicators()
        
        data_dict = {
            'gdp': gdp,
            'monthly': monthly,
            'daily': daily,
            'weekly': weekly
        }
        
        print(f"✅ 数据生成完成!")
        print(f"   GDP数据: {len(gdp)} 条 (季度)")
        print(f"   月度指标: {len(monthly)} 条")
        print(f"   日度金融: {len(daily)} 条")
        print(f"   周度指标: {len(weekly)} 条")
        
        return data_dict
    
    def save_data(self, data_dict: Dict[str, pd.DataFrame], output_dir: str = "./data/raw"):
        """
        保存数据到文件
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data_dict.items():
            filepath = f"{output_dir}/{name}_data.csv"
            df.to_csv(filepath)
            print(f"💾 已保存: {filepath}")


if __name__ == "__main__":
    # 生成demo数据
    generator = MacroDataGenerator(
        start_date="2015-01-01",
        end_date="2024-12-31",
        province="DemoProvince"
    )
    
    data = generator.generate_all_data()
    generator.save_data(data, output_dir="./raw")
    
    print("\n📊 数据预览:")
    print("\nGDP数据:")
    print(data['gdp'].head(10))
    print("\n月度指标:")
    print(data['monthly'].head(10))
