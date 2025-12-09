# 使用阿里云推荐的 Python 推理基础镜像（带 cuda/cudnn）
FROM pai-eas/python-inference:py39-ubuntu2004

# 设置工作目录
WORKDIR /workspace/app

# 将当前目录（EAS_service）全部复制进镜像
COPY . /workspace/app

# 设置环境变量，让你的 dihuman_core 能读取数据
ENV DIHUMAN_DATA_PATH=/workspace/app/stream_data

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 需要暴露给 EAS 的推理端口
EXPOSE 8000

# 入口命令（启动 FastAPI）
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
