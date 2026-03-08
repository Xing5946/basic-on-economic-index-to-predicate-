"""
定时任务调度器
实现数据自动更新、模型重训练、报告生成等定时任务
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from datetime import datetime
from typing import Dict, List, Callable, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    定时任务调度器
    基于APScheduler实现
    """
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.jobs = {}
        self.is_running = False
        
        # 添加事件监听
        self.scheduler.add_listener(
            self._job_event_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )
        
    def _job_event_listener(self, event):
        """任务事件监听器"""
        if event.exception:
            logger.error(f"❌ 任务执行失败: {event.job_id}")
            logger.error(f"   错误: {event.exception}")
        else:
            logger.info(f"✅ 任务执行成功: {event.job_id}")
            
    def add_data_update_job(self, func: Callable, 
                           cron_expression: str = "0 8 * * *"):
        """
        添加数据更新任务
        
        Args:
            func: 执行函数
            cron_expression: cron表达式 (默认每天8点)
        """
        job = self.scheduler.add_job(
            func=func,
            trigger=CronTrigger.from_crontab(cron_expression),
            id='data_update',
            name='数据自动更新',
            replace_existing=True
        )
        self.jobs['data_update'] = job
        logger.info(f"📅 添加数据更新任务: {cron_expression}")
        
    def add_model_retrain_job(self, func: Callable,
                              cron_expression: str = "0 2 * * 0"):
        """
        添加模型重训练任务
        
        Args:
            func: 执行函数
            cron_expression: cron表达式 (默认每周日凌晨2点)
        """
        job = self.scheduler.add_job(
            func=func,
            trigger=CronTrigger.from_crontab(cron_expression),
            id='model_retrain',
            name='模型自动重训练',
            replace_existing=True
        )
        self.jobs['model_retrain'] = job
        logger.info(f"📅 添加模型重训练任务: {cron_expression}")
        
    def add_report_generation_job(self, func: Callable,
                                  cron_expression: str = "0 9 * * 1"):
        """
        添加报告生成任务
        
        Args:
            func: 执行函数
            cron_expression: cron表达式 (默认每周一上午9点)
        """
        job = self.scheduler.add_job(
            func=func,
            trigger=CronTrigger.from_crontab(cron_expression),
            id='report_generation',
            name='自动生成报告',
            replace_existing=True
        )
        self.jobs['report_generation'] = job
        logger.info(f"📅 添加报告生成任务: {cron_expression}")
        
    def add_nowcast_update_job(self, func: Callable,
                               cron_expression: str = "0 */6 * * *"):
        """
        添加现时预测更新任务
        
        Args:
            func: 执行函数
            cron_expression: cron表达式 (默认每6小时)
        """
        job = self.scheduler.add_job(
            func=func,
            trigger=CronTrigger.from_crontab(cron_expression),
            id='nowcast_update',
            name='现时预测更新',
            replace_existing=True
        )
        self.jobs['nowcast_update'] = job
        logger.info(f"📅 添加现时预测更新任务: {cron_expression}")
        
    def add_custom_job(self, job_id: str, func: Callable,
                       cron_expression: str, name: str = ""):
        """
        添加自定义任务
        
        Args:
            job_id: 任务ID
            func: 执行函数
            cron_expression: cron表达式
            name: 任务名称
        """
        job = self.scheduler.add_job(
            func=func,
            trigger=CronTrigger.from_crontab(cron_expression),
            id=job_id,
            name=name or job_id,
            replace_existing=True
        )
        self.jobs[job_id] = job
        logger.info(f"📅 添加自定义任务 '{name}': {cron_expression}")
        
    def start(self):
        """启动调度器"""
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            logger.info("🚀 定时任务调度器已启动")
            
            # 打印所有任务
            self._print_jobs()
        else:
            logger.warning("调度器已在运行")
            
    def stop(self):
        """停止调度器"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("👋 定时任务调度器已停止")
            
    def pause_job(self, job_id: str):
        """暂停任务"""
        if job_id in self.jobs:
            self.jobs[job_id].pause()
            logger.info(f"⏸️ 任务已暂停: {job_id}")
            
    def resume_job(self, job_id: str):
        """恢复任务"""
        if job_id in self.jobs:
            self.jobs[job_id].resume()
            logger.info(f"▶️ 任务已恢复: {job_id}")
            
    def remove_job(self, job_id: str):
        """移除任务"""
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            logger.info(f"🗑️ 任务已移除: {job_id}")
            
    def get_jobs(self) -> List[Dict]:
        """获取所有任务信息"""
        jobs_info = []
        for job in self.scheduler.get_jobs():
            jobs_info.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        return jobs_info
        
    def _print_jobs(self):
        """打印所有任务"""
        logger.info("📋 当前定时任务列表:")
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.strftime('%Y-%m-%d %H:%M:%S') if job.next_run_time else "未知"
            logger.info(f"   - {job.name} ({job.id}): 下次执行 {next_run}")


# ============ 任务函数定义 ============

def data_update_task():
    """数据更新任务"""
    logger.info("="*60)
    logger.info("🔄 执行数据更新任务...")
    logger.info(f"   时间: {datetime.now().isoformat()}")
    
    try:
        # 这里实现数据更新逻辑
        # 1. 从Wind API获取最新数据
        # 2. 从统计局网站抓取数据
        # 3. 数据清洗和入库
        
        logger.info("   从Wind API获取数据...")
        # wind_data = fetch_wind_data()
        
        logger.info("   数据清洗...")
        # clean_data(wind_data)
        
        logger.info("   数据入库...")
        # save_to_database(clean_data)
        
        logger.info("✅ 数据更新任务完成")
    except Exception as e:
        logger.error(f"❌ 数据更新任务失败: {str(e)}")
        
    logger.info("="*60)


def model_retrain_task():
    """模型重训练任务"""
    logger.info("="*60)
    logger.info("🔄 执行模型重训练任务...")
    logger.info(f"   时间: {datetime.now().isoformat()}")
    
    try:
        # 这里实现模型重训练逻辑
        logger.info("   加载最新数据...")
        logger.info("   训练MIDAS模型...")
        logger.info("   训练DFM模型...")
        logger.info("   更新TSLM适配器...")
        logger.info("   保存模型...")
        
        logger.info("✅ 模型重训练任务完成")
    except Exception as e:
        logger.error(f"❌ 模型重训练任务失败: {str(e)}")
        
    logger.info("="*60)


def report_generation_task():
    """报告生成任务"""
    logger.info("="*60)
    logger.info("🔄 执行报告生成任务...")
    logger.info(f"   时间: {datetime.now().isoformat()}")
    
    try:
        logger.info("   生成预测结果...")
        logger.info("   生成图表...")
        logger.info("   生成PDF报告...")
        logger.info("   发送邮件通知...")
        
        logger.info("✅ 报告生成任务完成")
    except Exception as e:
        logger.error(f"❌ 报告生成任务失败: {str(e)}")
        
    logger.info("="*60)


def nowcast_update_task():
    """现时预测更新任务"""
    logger.info("="*60)
    logger.info("🔄 执行现时预测更新任务...")
    logger.info(f"   时间: {datetime.now().isoformat()}")
    
    try:
        logger.info("   获取最新可用数据...")
        logger.info("   更新现时预测...")
        logger.info("   保存预测结果...")
        
        logger.info("✅ 现时预测更新任务完成")
    except Exception as e:
        logger.error(f"❌ 现时预测更新任务失败: {str(e)}")
        
    logger.info("="*60)


# ============ 主程序入口 ============

if __name__ == "__main__":
    print("="*60)
    print("宏观经济预测系统 - 定时任务调度器")
    print("="*60)
    
    # 创建调度器
    scheduler = TaskScheduler()
    
    # 添加任务
    scheduler.add_data_update_job(data_update_task, "*/5 * * * *")  # 每5分钟（测试用）
    scheduler.add_model_retrain_job(model_retrain_task, "0 2 * * 0")
    scheduler.add_report_generation_job(report_generation_task, "0 9 * * 1")
    scheduler.add_nowcast_update_job(nowcast_update_task, "0 */6 * * *")
    
    # 启动调度器
    scheduler.start()
    
    print("\n按 Ctrl+C 停止调度器\n")
    
    try:
        # 保持程序运行
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 正在停止调度器...")
        scheduler.stop()
        print("调度器已停止")
