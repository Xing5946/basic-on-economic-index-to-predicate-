# 宏观经济预测系统 - 部署指南

本指南详细说明如何将宏观经济预测系统部署到生产环境。

---

## 📋 目录

1. [环境要求](#环境要求)
2. [本地部署](#本地部署)
3. [Docker部署](#docker部署)
4. [云服务器部署](#云服务器部署)
5. [生产环境优化](#生产环境优化)
6. [监控与维护](#监控与维护)

---

## 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4核 | 8核+ |
| 内存 | 16GB | 32GB+ |
| 存储 | 50GB SSD | 100GB SSD |
| GPU | 可选 | NVIDIA RTX 3090 / A100 |

### 软件要求

- Python 3.9+
- CUDA 11.8+ (GPU推理)
- Docker 20.10+ (容器化部署)
- Nginx 1.18+ (反向代理)

---

## 本地部署

### 1. 克隆项目

```bash
git clone <repository-url>
cd macro_econ_forecast
```

### 2. 创建虚拟环境

```bash
# 使用conda
conda create -n macro_forecast python=3.10
conda activate macro_forecast

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# GPU支持（可选）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. 配置环境变量

```bash
# 创建.env文件
cat > .env << EOF
# API配置
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# 数据库配置
DATABASE_URL=sqlite:///./data/macro_econ.db

# Redis配置（可选）
REDIS_URL=redis://localhost:6379/0

# Wind API配置（真实数据源）
WIND_API_KEY=your_wind_api_key

# 日志级别
LOG_LEVEL=INFO

# GPU配置
CUDA_VISIBLE_DEVICES=0
EOF
```

### 5. 初始化数据

```bash
# 生成模拟数据
python -c "
from data.data_generator import MacroDataGenerator
generator = MacroDataGenerator()
data = generator.generate_all_data()
generator.save_data(data, './data/raw')
"
```

### 6. 启动服务

```bash
# 方式一：一键启动所有服务
python run.py demo

# 方式二：分别启动
# 终端1 - API服务
python run.py api

# 终端2 - 前端服务
python run.py frontend

# 终端3 - 定时任务
python run.py scheduler
```

### 7. 访问系统

- 前端界面: http://localhost:8501
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/v1/health

---

## Docker部署

### 1. 创建Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p data/raw data/processed

# 暴露端口
EXPOSE 8000 8501

# 启动命令
CMD ["python", "run.py", "demo"]
```

### 2. 创建docker-compose.yml

```yaml
version: '3.8'

services:
  # API服务
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - DATABASE_URL=sqlite:///./data/macro_econ.db
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    command: python run.py api
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # 前端服务
  frontend:
    build: .
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000
    depends_on:
      - api
    command: python run.py frontend
    restart: unless-stopped

  # 定时任务
  scheduler:
    build: .
    volumes:
      - ./data:/app/data
    depends_on:
      - api
    command: python run.py scheduler
    restart: unless-stopped

  # Redis（可选，用于缓存和任务队列）
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Nginx反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
      - frontend
    restart: unless-stopped

volumes:
  redis_data:
```

### 3. 构建和启动

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## 云服务器部署

### AWS部署

```bash
# 1. 创建EC2实例（推荐t3.xlarge或更高）

# 2. 连接实例
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# 4. 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 5. 克隆项目并部署
git clone <repository-url>
cd macro_econ_forecast
docker-compose up -d
```

### 阿里云部署

```bash
# 1. 创建ECS实例

# 2. 配置安全组（开放8000, 8501端口）

# 3. 连接实例并部署
ssh root@your-ecs-ip

# 安装Docker和Docker Compose
# ... 同上

# 部署
docker-compose up -d
```

---

## 生产环境优化

### 1. Gunicorn配置

```python
# gunicorn.conf.py
import multiprocessing

# 工作进程数
workers = multiprocessing.cpu_count() * 2 + 1

# 工作模式
worker_class = "uvicorn.workers.UvicornWorker"

# 绑定地址
bind = "0.0.0.0:8000"

# 超时设置
timeout = 120
keepalive = 5

# 日志配置
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"

# 进程名称
proc_name = "macro_forecast_api"

# 预加载应用
preload_app = True
```

### 2. Nginx配置

```nginx
# nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    # Gzip压缩
gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript;

    # 上游服务器
    upstream api_server {
        server api:8000;
    }

    upstream frontend_server {
        server frontend:8501;
    }

    # API服务
    server {
        listen 80;
        server_name api.yourdomain.com;

        location / {
            proxy_pass http://api_server;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 超时设置
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
    }

    # 前端服务
    server {
        listen 80;
        server_name yourdomain.com;

        location / {
            proxy_pass http://frontend_server;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket支持
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

### 3. 数据库优化

```python
# 使用连接池
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

### 4. 缓存配置

```python
# 使用Redis缓存
import redis
from functools import wraps

redis_client = redis.Redis.from_url(REDIS_URL)

def cache_result(expire=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # 尝试从缓存获取
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            redis_client.setex(
                cache_key,
                expire,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator
```

---

## 监控与维护

### 1. 日志监控

```bash
# 查看API日志
docker-compose logs -f api

# 查看前端日志
docker-compose logs -f frontend

# 查看定时任务日志
docker-compose logs -f scheduler
```

### 2. 性能监控

```python
# 添加Prometheus监控
from prometheus_client import Counter, Histogram, generate_latest

# 定义指标
prediction_count = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

# 使用装饰器监控
@prediction_duration.time()
def make_prediction():
    prediction_count.inc()
    # 预测逻辑
```

### 3. 健康检查

```bash
# API健康检查
curl http://localhost:8000/api/v1/health

# 系统状态检查
curl http://localhost:8000/api/v1/model/status
```

### 4. 备份策略

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/macro_forecast"
DATE=$(date +%Y%m%d_%H%M%S)

# 备份数据库
cp data/macro_econ.db $BACKUP_DIR/db_$DATE.db

# 备份模型
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# 备份配置
cp -r config $BACKUP_DIR/config_$DATE

# 清理旧备份（保留7天）
find $BACKUP_DIR -name "*.db" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

### 5. 自动更新

```bash
#!/bin/bash
# update.sh

# 拉取最新代码
git pull origin main

# 重建镜像
docker-compose build

# 滚动更新
docker-compose up -d --no-deps api
docker-compose up -d --no-deps frontend

# 清理旧镜像
docker image prune -f

echo "Update completed"
```

---

## 故障排除

### 服务无法启动

```bash
# 检查端口占用
netstat -tulpn | grep 8000

# 检查日志
docker-compose logs api

# 重启服务
docker-compose restart
```

### 内存不足

```bash
# 查看内存使用
free -h

# 限制容器内存
docker-compose up -d --memory=8g

# 清理缓存
echo 3 > /proc/sys/vm/drop_caches
```

### 数据库连接失败

```bash
# 检查数据库文件
ls -lh data/macro_econ.db

# 修复权限
chmod 666 data/macro_econ.db

# 重新初始化
rm data/macro_econ.db
python -c "from backend.core.data_manager import DataManager; DataManager().initialize()"
```

---

## 安全建议

1. **使用HTTPS**: 配置SSL证书
2. **限制访问**: 配置防火墙规则
3. **定期更新**: 及时更新依赖包
4. **日志审计**: 定期检查访问日志
5. **数据加密**: 敏感数据加密存储

---

## 参考资源

- [FastAPI部署文档](https://fastapi.tiangolo.com/deployment/)
- [Docker官方文档](https://docs.docker.com/)
- [Nginx官方文档](https://nginx.org/en/docs/)
