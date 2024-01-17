from concurrent import futures
from logging import log
import os
import time
import paddle
from fastapi import FastAPI
import uvicorn

import RequestDto
from cfg_utils import argsparser
from my_utils import getConfigFilesList
from pipeline_fight import main

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/healthcheck")
def healthcheck():
    return {"server status": "ok"}

@app.post("/api/fight")
def start(request_dto: RequestDto.RequestDto):
    
    paddle.enable_static()
    # log.info("算法模块启动请求：{}", request_dto)
    parser = argsparser()
    input_type = request_dto.input
    camera_id = request_dto.camera_id
    algorithm = request_dto.algorithm
    FLAGS = parser.parse_args()
    config_dir_root = "deploy/pipeline/config/examples/"
    # config_dir_root = "config/examples/"
    config_files = os.listdir(config_dir_root)
    config_dict = getConfigFilesList(config_files)
    
    for k, v in config_dict.items():
        if algorithm == k:
            FLAGS.config = config_dir_root + v  
    # FLAGS.config = "./config/infer_cfg_pphuman.yml"
    FLAGS.rtsp = request_dto.stream_url
    FLAGS.device = request_dto.device
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU'
                            ], "device should be CPU, GPU, XPU or NPU"
    # TODO 异步响应，后台执行
    executor = futures.ThreadPoolExecutor(1)
    future = executor.submit(main, FLAGS)
    # 等待线程启动，两秒之后判断线程是否正在运行
    time.sleep(2)
    if future.running():
        return {200: "启动成功"}
    else:
        return {500: "启动失败"}
    # main(FLAGS)
    
if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8000)