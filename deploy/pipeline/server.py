import paddle
# import pipeline.cfg_utils as cfg_utils
from fastapi import FastAPI

import RequestDto
from cfg_utils import argsparser
from pipeline_bak import main

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/api/detection")
def start(request_dto: RequestDto.RequestDto):
    
    paddle.enable_static()
    parser = argsparser()
    input_type = request_dto.input
    camera_id = request_dto.camera_id
    FLAGS = parser.parse_args()
    FLAGS.config = "config/infer_cfg_pphuman.yml"
    FLAGS.rtsp = request_dto.stream_url
    FLAGS.device = request_dto.device
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU'
                            ], "device should be CPU, GPU, XPU or NPU"
    # main(FLAGS=FLAGS)
    
    return {200: "启动成功"}
