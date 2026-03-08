"""
宏观经济预测系统 - 主运行脚本
提供一键启动所有服务的功能
"""
import os
import sys
import subprocess
import argparse
import time
import signal
from pathlib import Path
import pandas as pd
import numpy as np
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()


def print_banner():
    """打印启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║           📊 宏观经济核心指标预测系统 📊                                        ║
    ║                                                                              ║
    ║      基于 MIDAS + DFM + TSLM 的混合预测模型                                     ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  技术架构:                                                                    ║
    ║    • MIDAS  - 混频数据抽样模型（处理高频数据）                                    ║
    ║    • DFM    - 动态因子模型（提取共同因子）                                       ║
    ║    • TSLM   - 时间序列大模型（非线性残差修正）                                    ║
    ║    • Hybrid - 混合预测模型（线性+非线性融合）                                     ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """检查依赖是否安装"""
    print("🔍 检查依赖...")
    
    required_packages = [
        'numpy', 'pandas', 'scipy', 'sklearn',
        'statsmodels', 'torch', 'fastapi', 'uvicorn',
        'streamlit', 'plotly', 'apscheduler'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  缺少依赖包: {', '.join(missing)}")
        print("📦 安装命令: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖已安装")
    return True


def run_api_server():
    """启动API服务器"""
    print("\n🚀 启动API服务器...")
    print("   地址: http://localhost:8000")
    print("   文档: http://localhost:8000/docs")
    
    api_script = PROJECT_ROOT / "backend" / "api" / "main.py"
    
    process = subprocess.Popen(
        [sys.executable, str(api_script)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return process


def run_frontend():
    """启动前端界面"""
    print("\n🚀 启动前端界面...")
    print("   地址: http://localhost:8501")
    
    frontend_script = PROJECT_ROOT / "frontend" / "app.py"
    
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(frontend_script)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return process


def run_scheduler():
    """启动定时任务调度器"""
    print("\n🚀 启动定时任务调度器...")
    
    scheduler_script = PROJECT_ROOT / "backend" / "tasks" / "scheduler.py"
    
    process = subprocess.Popen(
        [sys.executable, str(scheduler_script)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return process


def run_demo():
    """运行完整演示"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 请先安装依赖: pip install -r requirements.txt")
        return
    
    processes = []
    
    try:
        # 启动API服务器
        api_process = run_api_server()
        processes.append(("API Server", api_process))
        time.sleep(3)  # 等待API启动
        
        # 启动前端
        frontend_process = run_frontend()
        processes.append(("Frontend", frontend_process))
        time.sleep(2)
        
        # 启动定时任务（可选）
        # scheduler_process = run_scheduler()
        # processes.append(("Scheduler", scheduler_process))
        
        print("\n" + "="*80)
        print("✅ 所有服务已启动!")
        print("="*80)
        print("\n📱 访问地址:")
        print("   • 前端界面: http://localhost:8501")
        print("   • API文档: http://localhost:8000/docs")
        print("   • API健康检查: http://localhost:8000/api/v1/health")
        print("\n⚠️  按 Ctrl+C 停止所有服务")
        print("="*80 + "\n")
        
        # 等待所有进程
        while True:
            for name, process in processes:
                retcode = process.poll()
                if retcode is not None:
                    print(f"❌ {name} 已退出 (code: {retcode})")
                    return
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 正在停止所有服务...")
        for name, process in processes:
            print(f"   停止 {name}...")
            process.terminate()
            process.wait()
        print("✅ 所有服务已停止")


def run_quick_test():
    """快速测试"""
    print_banner()
    print("\n🧪 运行快速测试...\n")
    
    # 测试数据生成
    print("1️⃣  测试数据生成...")
    from data.data_generator import MacroDataGenerator
    generator = MacroDataGenerator()
    data = generator.generate_all_data()
    print("   ✅ 数据生成成功")
    
    # 测试数据预处理
    print("\n2️⃣  测试数据预处理...")
    from data.data_processor import DataProcessor
    processor = DataProcessor()
    processed = processor.preprocess_pipeline(data)
    print("   ✅ 数据预处理成功")
    
    # 测试MIDAS模型
    print("\n3️⃣  测试MIDAS模型...")
    from models.midas.midas_model import MIDASModel
    
    gdp = processed['gdp']['gdp_value_clean']
    electricity = processed['monthly']['electricity']
    
    gdp_aligned, elec_aligned = processor.align_frequencies(
        pd.DataFrame(gdp), pd.DataFrame(electricity), 'Q', 'M'
    )
    
    midas = MIDASModel(n_lags=12)
    midas.fit(gdp_aligned['gdp_value_clean'], elec_aligned['electricity'])
    print("   ✅ MIDAS模型训练成功")
    
    # 测试DFM模型
    print("\n4️⃣  测试DFM模型...")
    from models.dfm.dfm_model import DFMModel
    
    monthly_numeric = processed['monthly'].select_dtypes(include=[np.number])
    dfm = DFMModel(n_factors=3)
    dfm.fit(monthly_numeric)
    print("   ✅ DFM模型训练成功")
    
    # 测试TSLM适配器
    print("\n5️⃣  测试TSLM适配器...")
    from models.tslm.tslm_adapter import TSLMAdapter, TSLMConfig
    
    config = TSLMConfig(model_name="timesfm", context_length=128, prediction_length=12)
    tslm = TSLMAdapter(config)
    tslm.initialize(use_mock=True)
    print("   ✅ TSLM适配器初始化成功")
    
    # 测试混合模型
    print("\n6️⃣  测试混合模型...")
    from models.hybrid.hybrid_model import HybridPredictor, HybridModelConfig
    
    hybrid_config = HybridModelConfig(linear_weight=0.6, nonlinear_weight=0.4)
    hybrid = HybridPredictor(hybrid_config)
    hybrid.set_models(midas, tslm)
    hybrid.fit(gdp_aligned['gdp_value_clean'])
    prediction = hybrid.predict(steps=1)
    print("   ✅ 混合模型预测成功")
    print(f"      预测值: {prediction['final_prediction'][0]:.2f}")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过!")
    print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="宏观经济预测系统 - 主运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py demo      # 启动完整演示
  python run.py test      # 运行快速测试
  python run.py api       # 仅启动API服务器
  python run.py frontend  # 仅启动前端界面
        """
    )
    
    parser.add_argument(
        'command',
        choices=['demo', 'test', 'api', 'frontend', 'scheduler'],
        help='运行命令'
    )
    
    args = parser.parse_args()
    """
run.py编辑配置的另一种途径，可选，但只有一种功能限制，可以改变default变量达到目的
parser = argparse.ArgumentParser(...)
parser.add_argument('command', nargs='?', default='api',
                   choices=['demo', 'test', 'api', 'frontend', 'scheduler'])
args = parser.parse_args()
    """
    if args.command == 'demo':
        run_demo()
    elif args.command == 'test':
        run_quick_test()
    elif args.command == 'api':
        print_banner()
        if check_dependencies():
            process = run_api_server()
            try:
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
    elif args.command == 'frontend':
        print_banner()
        if check_dependencies():
            process = run_frontend()
            try:
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
    elif args.command == 'scheduler':
        print_banner()
        if check_dependencies():
            process = run_scheduler()
            try:
                process.wait()
            except KeyboardInterrupt:
                process.terminate()


if __name__ == "__main__":
    main()
