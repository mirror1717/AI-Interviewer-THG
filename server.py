# server.py
# -*- coding: utf-8 -*-
import base64
import os
from typing import Optional, Dict

import numpy as np
import cv2
from fastapi import FastAPI
from pydantic import BaseModel

from dihuman_core import DiHumanProcessor

# ---------------------------
# 1. 创建 FastAPI 应用
# ---------------------------
app = FastAPI()

# ---------------------------
# 2. 会话池：session_id -> DiHumanProcessor
#    每个 session 一套流式状态（缓冲、索引等）
# ---------------------------
processors: Dict[str, DiHumanProcessor] = {}

# 3. 通过环境变量控制数据路径
#    在 Docker / EAS 里，我们会设置：
#    DIHUMAN_DATA_PATH=/workspace/app/stream_data
DATA_PATH = os.environ.get("DIHUMAN_DATA_PATH", "./stream_data")


def get_processor(session_id: str) -> DiHumanProcessor:
    """
    为每个 session_id 维护一个独立的 DiHumanProcessor 实例。
    - 第一次遇到这个 session_id 时创建一个新的 Processor；
    - 之后同一个 session 会复用上次的状态（音频缓冲、index 等）。
    """
    if session_id not in processors:
        processors[session_id] = DiHumanProcessor(DATA_PATH)
    return processors[session_id]


# ---------------------------
# 4. 请求 / 响应数据结构
# ---------------------------
class StreamStepRequest(BaseModel):
    session_id: str                # 会话 id，前端 / 工作流负责维护
    audio_chunk: str               # base64 编码的 int16 PCM（建议 10ms = 160 采样）
    reset: Optional[bool] = False  # 是否重新开始（清空内部状态）


class StreamStepResponse(BaseModel):
    frame: Optional[str]           # base64(jpeg)，可能为 None
    audio: str                     # base64(int16 PCM 10ms)
    check_img: int                 # 1 表示本次有新图像帧


# ---------------------------
# 5. 探活接口（给 EAS / 你自己调试用）
# ---------------------------
@app.get("/health")
def health():
    """
    简单健康检查：
    - 部署后你可以直接 GET /health 看是否返回 {"status": "ok"}
    """
    return {"status": "ok"}


@app.get("/")
def root():
    """
    根路径简单说明一下服务情况。
    """
    return {"message": "Digital Human streaming service is running.",
            "data_path": DATA_PATH}


# ---------------------------
# 6. 主推理接口：流式一步
# ---------------------------
@app.post("/stream_step", response_model=StreamStepResponse)
def stream_step(req: StreamStepRequest):
    # 1. 找到 / 创建对应 session 的处理器
    proc = get_processor(req.session_id)

    # 2. reset（可选），相当于本地调用 proc.reset()
    if req.reset:
        proc.reset()

    # 3. 解码 audio_chunk：base64 -> bytes -> np.int16
    pcm_bytes = base64.b64decode(req.audio_chunk)
    audio_frame = np.frombuffer(pcm_bytes, dtype=np.int16)

    # 4. 调用你原本的流式逻辑
    img, playing_audio, check_img = proc.process(audio_frame)

    # 5. 编码音频（一定有）
    audio_b64 = base64.b64encode(playing_audio.tobytes()).decode("ascii")

    # 6. 编码图像（可能没有）
    frame_b64 = None
    if check_img and img is not None:
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return StreamStepResponse(
        frame=frame_b64,
        audio=audio_b64,
        check_img=int(check_img),
    )
