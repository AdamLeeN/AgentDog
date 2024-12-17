import os
import time
# import torch
import uvicorn
import json

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
# from sse_starlette.sse import EventSourceResponse



# #Set up limit request time
# EventSourceResponse.DEFAULT_PING_INTERVAL = 1500
# #定义异步上下文管理器: yield分割应用程序启动or结束前的操作
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     yield
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()

app = FastAPI(debug=True)
#CORS配置允许前端访问所有api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    param: str






@app.post("/turn_left")
async def TURN_LEFT_API(item: Item):
    return {"response": "TurnLeft()"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)








