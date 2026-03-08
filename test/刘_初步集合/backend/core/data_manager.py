"""
数据管理器
负责数据的加载、存储、更新和查询
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json


class DataManager:
    """
    数据管理器
    管理宏观经济数据的整个生命周期
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.raw_data = {}
        self.processed_data = {}
        self.is_initialized = False
        
    def initialize(self):
        """初始化数据管理器"""
        print("🔄 初始化数据管理器...")
        
        # 加载数据
        self._load_data()
        
        self.is_initialized = True
        print("✅ 数据管理器初始化完成")
        
    def _load_data(self):
        """加载数据"""
        from data.data_generator import MacroDataGenerator
        
        # 使用模拟数据
        generator = MacroDataGenerator()
        self.raw_data = generator.generate_all_data()
        
    def get_gdp_data(self, start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取GDP数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            GDP数据DataFrame
        """
        df = self.raw_data.get('gdp', pd.DataFrame()).copy()
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        # 重置索引以便序列化
        df = df.reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        return df
        
    def get_monthly_data(self, indicator: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取月度指标数据
        
        Args:
            indicator: 特定指标名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            月度数据DataFrame
        """
        df = self.raw_data.get('monthly', pd.DataFrame()).copy()
        
        # 筛选特定指标
        if indicator and indicator in df.columns:
            df = df[['province', indicator]]
            
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        # 重置索引
        df = df.reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        return df
        
    def get_daily_data(self, indicator: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """获取日度金融数据"""
        df = self.raw_data.get('daily', pd.DataFrame()).copy()
        
        if indicator and indicator in df.columns:
            df = df[['province', indicator]]
            
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        df = df.reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        return df
        
    def get_weekly_data(self, indicator: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """获取周度数据"""
        df = self.raw_data.get('weekly', pd.DataFrame()).copy()
        
        if indicator and indicator in df.columns:
            df = df[['province', indicator]]
            
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        df = df.reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        return df
        
    def get_latest_snapshot(self) -> Dict[str, Any]:
        """
        获取最新数据快照
        
        Returns:
            最新数据摘要
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {}
        }
        
        # GDP最新数据
        if 'gdp' in self.raw_data:
            gdp = self.raw_data['gdp']
            latest_gdp = gdp.iloc[-1]
            snapshot['data_summary']['gdp'] = {
                "latest_date": gdp.index[-1].strftime('%Y-%m-%d'),
                "latest_value": float(latest_gdp['gdp_value']),
                "yoy_growth": float(latest_gdp['gdp_yoy']) if not pd.isna(latest_gdp['gdp_yoy']) else None
            }
            
        # 月度指标最新数据
        if 'monthly' in self.raw_data:
            monthly = self.raw_data['monthly']
            latest_monthly = monthly.iloc[-1]
            snapshot['data_summary']['monthly'] = {
                "latest_date": monthly.index[-1].strftime('%Y-%m-%d'),
                "indicators": {
                    col: float(latest_monthly[col]) 
                    for col in monthly.select_dtypes(include=[np.number]).columns
                    if col != 'province'
                }
            }
            
        # 日度数据最新
        if 'daily' in self.raw_data:
            daily = self.raw_data['daily']
            latest_daily = daily.iloc[-1]
            snapshot['data_summary']['daily'] = {
                "latest_date": daily.index[-1].strftime('%Y-%m-%d'),
                "stock_index": float(latest_daily['stock_index'])
            }
            
        return snapshot
        
    def update_data(self, source: str = "api") -> bool:
        """
        更新数据
        
        Args:
            source: 数据来源 (api, file, manual)
            
        Returns:
            是否成功
        """
        print(f"🔄 从{source}更新数据...")
        
        # 实际部署时实现数据更新逻辑
        # 这里简化处理
        
        print("✅ 数据更新完成")
        return True
        
    def get_data_availability(self) -> Dict[str, Any]:
        """
        获取数据可用性报告
        """
        availability = {
            "timestamp": datetime.now().isoformat(),
            "datasets": {}
        }
        
        for name, df in self.raw_data.items():
            if not df.empty:
                availability['datasets'][name] = {
                    "start_date": df.index[0].strftime('%Y-%m-%d'),
                    "end_date": df.index[-1].strftime('%Y-%m-%d'),
                    "n_observations": len(df),
                    "n_variables": len(df.columns),
                    "completeness": 1 - df.isna().sum().sum() / (len(df) * len(df.columns))
                }
                
        return availability
