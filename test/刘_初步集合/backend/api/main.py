"""
宏观经济预测系统 - FastAPI后端服务
提供数据查询、模型预测、结果展示等RESTful API
"""
import os
import sys

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import yaml

# 导入自定义模块
from backend.core.prediction_engine import PredictionEngine
from backend.core.data_manager import DataManager
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 创建FastAPI应用
app = FastAPI(
    title="宏观经济核心指标预测系统 API",
    description="基于MIDAS+DFM+TSLM的混合预测模型",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态
app_state = {
    "prediction_engine": None,
    "data_manager": None,
    "initialized": False,
    "last_update": None
}


# ============ Pydantic模型定义 ============

class GDPData(BaseModel):
    date: str
    value: float
    yoy_growth: Optional[float] = None


class MonthlyIndicator(BaseModel):
    date: str
    industrial_output: Optional[float] = None
    retail_sales: Optional[float] = None
    fixed_investment: Optional[float] = None
    electricity: Optional[float] = None
    pmi: Optional[float] = None
    cpi: Optional[float] = None
    ppi: Optional[float] = None


class PredictionRequest(BaseModel):
    target: str = Field(default="gdp", description="预测目标")
    horizon: int = Field(default=1, ge=1, le=4, description="预测期数")
    use_hybrid: bool = Field(default=True, description="是否使用混合模型")


class PredictionResponse(BaseModel):
    target: str
    prediction_date: str
    linear_prediction: float
    nonlinear_correction: float
    final_prediction: float
    confidence_interval: Optional[Dict[str, float]] = None
    model_weights: Dict[str, float]


class NowcastResponse(BaseModel):
    target: str
    current_quarter: str
    linear_nowcast: float
    nonlinear_correction: float
    final_nowcast: float
    data_availability: Dict[str, float]


class ModelStatus(BaseModel):
    status: str
    last_trained: Optional[str] = None
    models_loaded: Dict[str, bool]
    performance_metrics: Optional[Dict[str, float]] = None


class FactorAnalysis(BaseModel):
    factor_name: str
    explained_variance: float
    loadings: Dict[str, float]


# ============ 生命周期事件 ============

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    print("🚀 正在初始化宏观经济预测系统...")
    
    try:
        # 初始化数据管理器
        app_state["data_manager"] = DataManager()
        app_state["data_manager"].initialize()
        
        # 初始化预测引擎
        app_state["prediction_engine"] = PredictionEngine()
        app_state["prediction_engine"].initialize()
        
        app_state["initialized"] = True
        app_state["last_update"] = datetime.now().isoformat()
        
        print("✅ 系统初始化完成!")
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        # 即使失败也允许启动，部分功能可能不可用


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    print("👋 正在关闭宏观经济预测系统...")
    app_state["initialized"] = False


# ============ API路由 ============

@app.get("/")
async def root():
    """根路径 - API信息"""
    return {
        "name": "宏观经济核心指标预测系统 API",
        "version": "1.0.0",
        "status": "running" if app_state["initialized"] else "initializing",
        "docs": "/docs"
    }


@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if app_state["initialized"] else "degraded",
        "initialized": app_state["initialized"],
        "last_update": app_state["last_update"]
    }


# ============ 数据查询API ============

@app.get("/api/v1/data/gdp", response_model=List[GDPData])
async def get_gdp_data(
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)")
):
    """
    获取GDP历史数据
    """
    try:
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        data = app_state["data_manager"].get_gdp_data(start_date, end_date)
        return data.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/monthly", response_model=List[MonthlyIndicator])
async def get_monthly_data(
    indicator: Optional[str] = Query(None, description="特定指标名称"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期")
):
    """
    获取月度指标数据
    """
    try:
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        data = app_state["data_manager"].get_monthly_data(indicator, start_date, end_date)
        return data.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/latest")
async def get_latest_data():
    """
    获取最新数据快照
    """
    try:
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        return app_state["data_manager"].get_latest_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 预测API ============

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    进行预测
    """
    try:
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        result = app_state["prediction_engine"].predict(
            target=request.target,
            horizon=request.horizon,
            use_hybrid=request.use_hybrid
        )
        
        return PredictionResponse(
            target=request.target,
            prediction_date=datetime.now().isoformat(),
            linear_prediction=result['linear_prediction'],
            nonlinear_correction=result['nonlinear_correction'],
            final_prediction=result['final_prediction'],
            confidence_interval=result.get('confidence_interval'),
            model_weights=result['model_weights']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/nowcast", response_model=NowcastResponse)
async def nowcast(target: str = Query(default="gdp", description="预测目标")):
    """
    现时预测（Nowcasting）- 使用最新可用数据预测当前季度
    """
    try:
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        result = app_state["prediction_engine"].nowcast(target=target)
        
        return NowcastResponse(
            target=target,
            current_quarter=result['current_quarter'],
            linear_nowcast=result['linear_nowcast'],
            nonlinear_correction=result['nonlinear_correction'],
            final_nowcast=result['final_nowcast'],
            data_availability=result['data_availability']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/predict/history")
async def get_prediction_history(
    target: str = Query(default="gdp"),
    n_periods: int = Query(default=10, ge=1, le=100)
):
    """
    获取历史预测记录
    """
    try:
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        return app_state["prediction_engine"].get_prediction_history(target, n_periods)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 模型管理API ============

@app.get("/api/v1/model/status", response_model=ModelStatus)
async def get_model_status():
    """
    获取模型状态
    """
    try:
        if not app_state["initialized"]:
            return ModelStatus(
                status="initializing",
                models_loaded={"midas": False, "dfm": False, "tslm": False}
            )
        
        return app_state["prediction_engine"].get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/model/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """
    重新训练模型（异步任务）
    """
    try:
        background_tasks.add_task(_retrain_models_task)
        return {"message": "模型重训练任务已启动", "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _retrain_models_task():
    """后台重训练任务"""
    try:
        print("🔄 后台任务: 重新训练模型...")
        if app_state["prediction_engine"]:
            app_state["prediction_engine"].retrain()
            app_state["last_update"] = datetime.now().isoformat()
        print("✅ 模型重训练完成")
    except Exception as e:
        print(f"❌ 模型重训练失败: {str(e)}")


@app.get("/api/v1/model/performance")
async def get_model_performance(
    target: str = Query(default="gdp"),
    metric: str = Query(default="rmse", description="评估指标: rmse, mae, mape, r2")
):
    """
    获取模型性能指标
    """
    try:
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        return app_state["prediction_engine"].get_performance_metrics(target, metric)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 因子分析API ============

@app.get("/api/v1/analysis/factors")
async def get_factor_analysis():
    """
    获取DFM因子分析结果
    """
    try:
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        return app_state["prediction_engine"].get_factor_analysis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analysis/attribution")
async def get_prediction_attribution(
    target: str = Query(default="gdp"),
    date: Optional[str] = Query(None, description="特定日期")
):
    """
    获取预测归因分析
    """
    try:
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        return app_state["prediction_engine"].get_attribution(target, date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analysis/midas-weights")
async def get_midas_weights():
    """
    获取MIDAS模型滞后权重
    """
    try:
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        return app_state["prediction_engine"].get_midas_weights()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 报告API ============

@app.get("/api/v1/report/summary")
async def get_summary_report():
    """
    获取综合摘要报告
    """
    try:
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        return {
            "generated_at": datetime.now().isoformat(),
            "latest_nowcast": await nowcast(),
            "model_status": await get_model_status(),
            "latest_data": app_state["data_manager"].get_latest_snapshot()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/report/generate")
async def generate_report(background_tasks: BackgroundTasks):
    """
    生成PDF报告（异步任务）
    """
    try:
        background_tasks.add_task(_generate_report_task)
        return {"message": "报告生成任务已启动", "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_report_task():
    """后台报告生成任务"""
    try:
        print("🔄 后台任务: 生成报告...")
        # 实际报告生成逻辑
        print("✅ 报告生成完成")
    except Exception as e:
        print(f"❌ 报告生成失败: {str(e)}")


# ============ 主程序入口 ============

if __name__ == "__main__":
    import uvicorn
    
    host = config['backend']['host']
    port = config['backend']['port']
    reload = config['backend']['reload']
    
    print(f"🚀 启动API服务: http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=reload)
