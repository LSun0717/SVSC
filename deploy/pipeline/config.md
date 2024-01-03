## FLAGS中参数

camera_id=-1, 
config='deploy/pipeline/config/infer_cfg_pphuman.yml', 
cpu_threads=1, 
device='gpu', 
do_break_in_counting=False, 
do_entrance_counting=False, 
draw_center_traj=False, 
enable_mkldnn=False, 
illegal_parking_time=-1, 
image_dir=None, 
image_file=None, 
opt={}, 
output_dir='output', 
pushurl='', 
region_polygon=[], 
region_type='horizontal', 
rtsp=['rtsp://admin:my550025@192.168.1.64:554/h264/ch1/main/av_stream'], run_mode='paddle', 
secs_interval=2, 
trt_calib_mode=False, 
trt_max_shape=1280, 
trt_min_shape=1, 
trt_opt_shape=640, 
video_dir=None, 
video_file=None

{
  "input": "stream",
  "stream_url": "rtsp://admin:my550025@192.168.1.64:554/h264/ch1/main/av_stream",
  "algorithm": "fight",
  "device": "gpu"
}


{'opt': {}, 'config': None, 'image_file': None, 'image_dir': None, 'video_file': None, 'video_dir': None, 'rtsp': None, 'camera_id': -1, 'output_dir': 'output', 'pushurl': '', 'run_mode': 'paddle', 'device': 'cpu', 'enable_mkldnn': False, 'cpu_threads': 1, 'trt_min_shape': 1, 'trt_max_shape': 1280, 'trt_opt_shape': 640, 'trt_calib_mode': False, 'do_entrance_counting': False, 'do_break_in_counting': False, 'illegal_parking_time': -1, 'region_type': 'horizontal', 'region_polygon': [], 'secs_interval': 2, 'draw_center_traj': False}