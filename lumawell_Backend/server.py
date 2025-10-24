# server.py  
from __future__ import annotations

import os
import traceback
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.requests import Request

#与 main.py 使用相同的 LangGraph 构建函数
from main import build_graph  # build_graph() 里已加载 .env、设置 checkpointer
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState  # 与 main.py 保持一致

app = FastAPI(title="LumaWell API", version="1.0.0")

# ------- 全局异常：统一返回 detail，前端可见 -------
@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )

# ------- CORS（开发期全开，生产按域名收紧） -------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------- I/O 模型 -------
class ChatIn(BaseModel):
    thread_id: str
    message: str
    city: Optional[str] = None    
    realtime: bool = True        

class ChatOut(BaseModel):
    reply: str
    route: Optional[str] = None
    sources: Optional[List[str]] = None
    tools: Optional[List[dict]] = None

# ------- 只构建一次，与 CLI 复用同一个图 -------
graph = build_graph()

@app.get("/")
def index():
    return {"ok": True, "service": "LumaWell API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ------- 对话入口：与 main.py 相同的调用方式 -------
@app.post("/chat", response_model=ChatOut)
def chat(inf: ChatIn):
    """
    前端 POST /chat
    Body: { thread_id, message, city?, realtime? }
    """
    try:
        #组装输入 —— 与 main.py 完全相同
        inputs = MessagesState(messages=[HumanMessage(content=inf.message)])

        # 调 LangGraph，同样传 thread_id
        result = graph.invoke(
            inputs,
            config={"configurable": {"thread_id": inf.thread_id}},
        )

        #取最后一条消息的文本 —— 与 main.py 完全相同
        last = result["messages"][-1]
        reply = getattr(last, "content", last.get("content") if isinstance(last, dict) else str(last))

        # 可选：附带路由、来源、工具（若你的图里有）
        route = result.get("route") or None

        sources = None
        if result.get("retrieved"):
            # 兼容你之前的来源结构：[{path:..., cite:...}, ...]
            try:
                sources = [
                    f"[{d.get('cite')}] {os.path.basename(d.get('path',''))}"
                    for d in result["retrieved"]
                ]
            except Exception:
                # 安全兜底
                sources = [str(d) for d in result["retrieved"]]

        tools = result.get("tool_outputs") or None

        return ChatOut(reply=reply, route=route, sources=sources, tools=tools)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
