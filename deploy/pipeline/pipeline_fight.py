import datetime
import os
import cv2
import paddle
import sys
import threading
import queue
import time
from datacollector import DataCollector, Result

# add deploy path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from cfg_utils import argsparser, print_arguments, merge_cfg
from pipe_utils import PipeTimer
from pipe_utils import get_test_images
from pipe_utils import PushStream

from python.infer import Detector
from python.preprocess import ShortSizeScale
from python.visualize import visualize_action


from pphuman.video_action_infer import VideoActionRecognizer


from download import auto_download_model


class Pipeline(object):
    """
    Pipeline

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
    """

    def __init__(self, args, cfg):
        self.multi_camera = False
        reid_cfg = cfg.get('REID', False)
        self.enable_mtmct = reid_cfg['enable'] if reid_cfg else False
        self.is_video = False
        self.output_dir = args.output_dir
        self.vis_result = cfg['visual']
        self.input = self._parse_input(args.image_file, args.image_dir,
                                       args.video_file, args.video_dir,
                                       args.camera_id, args.rtsp)
        if self.multi_camera:
            self.predictor = []
            for name in self.input:
                predictor_item = PipePredictor(
                    args, cfg, is_video=True, multi_camera=True)
                predictor_item.set_file_name(name)
                self.predictor.append(predictor_item)

        else:
            self.predictor = PipePredictor(args, cfg, self.is_video)
            if self.is_video:
                self.predictor.set_file_name(self.input)

    # 解析数据源
    def _parse_input(self, image_file, image_dir, video_file, video_dir,
                     camera_id, rtsp):
        if image_file is not None or image_dir is not None:
            input = get_test_images(image_dir, image_file)
            self.is_video = False
            self.multi_camera = False

        elif video_file is not None:
            assert os.path.exists(
                video_file
            ) or 'rtsp' in video_file, "video_file not exists and not an rtsp site."
            self.multi_camera = False
            input = video_file
            self.is_video = True

        elif video_dir is not None:
            videof = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
            if len(videof) > 1:
                self.multi_camera = True
                videof.sort()
                input = videof
            else:
                input = videof[0]
            self.is_video = True

        elif rtsp is not None:
            if len(rtsp) > 1:
                rtsp = [rtsp_item for rtsp_item in rtsp if 'rtsp' in rtsp_item]
                self.multi_camera = True
                input = rtsp
            else:
                self.multi_camera = False
                input = rtsp[0]
            self.is_video = True

        elif camera_id != -1:
            self.multi_camera = False
            input = camera_id
            self.is_video = True

        else:
            raise ValueError(
                "Illegal Input, please set one of ['video_file', 'camera_id', 'image_file', 'image_dir']"
            )

        return input

    def run_multithreads(self):
        if self.multi_camera:
            multi_res = []
            threads = []
            for idx, (predictor,
                      input) in enumerate(zip(self.predictor, self.input)):
                thread = threading.Thread(
                    name=str(idx).zfill(3),
                    target=predictor.run,
                    args=(input, idx))
                threads.append(thread)

            for thread in threads:
                thread.start()

            for predictor, thread in zip(self.predictor, threads):
                thread.join()
                collector_data = predictor.get_result()
                multi_res.append(collector_data)

        else:
            self.predictor.run(self.input)

    def run(self):
        if self.multi_camera:
            multi_res = []
            for predictor, input in zip(self.predictor, self.input):
                predictor.run(input)
                collector_data = predictor.get_result()
                multi_res.append(collector_data)
        else:
            self.predictor.run(self.input)


def get_model_dir(cfg):
    """ 
        Auto download inference model if the model_path is a url link. 
        Otherwise it will use the model_path directly.
    """
    for key in cfg.keys():
        if type(cfg[key]) ==  dict and \
            ("enable" in cfg[key].keys() and cfg[key]['enable']
                or "enable" not in cfg[key].keys()):

            if "model_dir" in cfg[key].keys():
                model_dir = cfg[key]["model_dir"]
                downloaded_model_dir = auto_download_model(model_dir)
                if downloaded_model_dir:
                    model_dir = downloaded_model_dir
                    cfg[key]["model_dir"] = model_dir
                print(key, " model dir: ", model_dir)
            elif key == "VEHICLE_PLATE":
                det_model_dir = cfg[key]["det_model_dir"]
                downloaded_det_model_dir = auto_download_model(det_model_dir)
                if downloaded_det_model_dir:
                    det_model_dir = downloaded_det_model_dir
                    cfg[key]["det_model_dir"] = det_model_dir
                print("det_model_dir model dir: ", det_model_dir)

                rec_model_dir = cfg[key]["rec_model_dir"]
                downloaded_rec_model_dir = auto_download_model(rec_model_dir)
                if downloaded_rec_model_dir:
                    rec_model_dir = downloaded_rec_model_dir
                    cfg[key]["rec_model_dir"] = rec_model_dir
                print("rec_model_dir model dir: ", rec_model_dir)

        elif key == "MOT":  # for idbased and skeletonbased actions
            model_dir = cfg[key]["model_dir"]
            downloaded_model_dir = auto_download_model(model_dir)
            if downloaded_model_dir:
                model_dir = downloaded_model_dir
                cfg[key]["model_dir"] = model_dir
            print("mot_model_dir model_dir: ", model_dir)


class PipePredictor(object):
    """
    Predictor in single camera
    
    The pipeline for image input: 

        1. Detection
        2. Detection -> Attribute

    The pipeline for video input: 

        1. Tracking
        2. Tracking -> Attribute
        3. Tracking -> KeyPoint -> SkeletonAction Recognition
        4. VideoAction Recognition

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
        is_video (bool): whether the input is video, default as False
        multi_camera (bool): whether to use multi camera in pipeline, 
            default as False
    """

    def __init__(self, args, cfg, is_video=True, multi_camera=False):

        # only for pphuman
        self.with_video_action = cfg.get(
            'VIDEO_ACTION', False)['enable'] if cfg.get('VIDEO_ACTION',
                                                        False) else False
        if self.with_video_action:
            print('VideoAction Recognition enabled')

        self.modebase = {
            "framebased": False,
            "videobased": False,
            "idbased": False,
            "skeletonbased": False
        }

        self.basemode = {
            "MOT": "idbased",
            "ATTR": "idbased",
            "VIDEO_ACTION": "videobased",
            "SKELETON_ACTION": "skeletonbased",
            "ID_BASED_DETACTION": "idbased",
            "ID_BASED_CLSACTION": "idbased",
            "REID": "idbased",
            "VEHICLE_PLATE": "idbased",
            "VEHICLE_ATTR": "idbased",
            "VEHICLE_PRESSING": "idbased",
            "VEHICLE_RETROGRADE": "idbased",
        }

        self.is_video = is_video
        self.multi_camera = multi_camera
        self.cfg = cfg

        self.output_dir = args.output_dir

        self.warmup_frame = self.cfg['warmup_frame']
        self.pipeline_res = Result()
        self.pipe_timer = PipeTimer()
        self.file_name = None
        self.collector = DataCollector()

        self.pushurl = args.pushurl

        # auto download inference model
        get_model_dir(self.cfg)

        if not is_video:

            det_cfg = self.cfg['DET']
            model_dir = det_cfg['model_dir']
            batch_size = det_cfg['batch_size']
            self.det_predictor = Detector(
                model_dir, args.device, args.run_mode, batch_size,
                args.trt_min_shape, args.trt_max_shape, args.trt_opt_shape,
                args.trt_calib_mode, args.cpu_threads, args.enable_mkldnn)
        else:
            if self.with_video_action:
                video_action_cfg = self.cfg['VIDEO_ACTION']
                basemode = self.basemode['VIDEO_ACTION']
                self.modebase[basemode] = True
                self.video_action_predictor = VideoActionRecognizer.init_with_cfg(
                    args, video_action_cfg)

    def set_file_name(self, path):
        if type(path) == int:
            self.file_name = path
        elif path is not None:
            self.file_name = os.path.split(path)[-1]
            if "." in self.file_name:
                self.file_name = self.file_name.split(".")[-2]
        else:
            # use camera id
            self.file_name = None

    def get_result(self):
        return self.collector.get_res()

    # 单线程执行（单rtsp）
    def run(self, input, thread_idx=0):
        self.predict_video(input, thread_idx=thread_idx)
        self.pipe_timer.info()

    # 存储图像帧至队列
    def capturevideo(self, capture, queue):
        frame_id = 0
        while (1):
            if queue.full():
                time.sleep(0.1)
            else:
                ret, frame = capture.read()
                if not ret:
                    return
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                queue.put(frame_rgb)

    def predict_video(self, video_file, thread_idx=0):
        capture = cv2.VideoCapture(video_file)

        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("video fps: %d, frame_count: %d" % (fps, frame_count))
        
        # for save video file
        algorithm = "fight"
        root_out_dir = "/home/lab102/project/PaddleDetection/output"
        now = datetime.datetime.now()
        day_folder = str(now.date())
        time_tag = str(now.time())
        parent_dir = os.path.join(root_out_dir, algorithm, day_folder)
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        out_path = os.path.join(parent_dir, time_tag + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        
        # 记录是否是事件的开始
        global begin_record
        begin_record = False
        # 全局视频文件句柄
        global writer
        writer = cv2.VideoWriter(out_path, fourcc, 30, (width, height))
            
        frame_id = 0
        video_fps = fps
        
        video_action_imgs = []

        if self.with_video_action:
            short_size = self.cfg["VIDEO_ACTION"]["short_size"]
            scale = ShortSizeScale(short_size)
        # 图像帧队列
        framequeue = queue.Queue(10)

        thread = threading.Thread(
            target=self.capturevideo, args=(capture, framequeue))
        thread.start()
        time.sleep(1)
        
        global valid_frame_count
        valid_frame_count = 10

        # while (not framequeue.empty()):
        while (True):
            if frame_id % 10 == 0:
                print('Thread: {}; frame id: {}'.format(thread_idx, frame_id))

            frame_rgb = framequeue.get()
            if frame_id > self.warmup_frame:
                self.pipe_timer.total_time.start()

            if self.with_video_action:
                # get the params
                frame_len = self.cfg["VIDEO_ACTION"]["frame_len"]
                sample_freq = self.cfg["VIDEO_ACTION"]["sample_freq"]

                # if sample_freq * frame_len > frame_count:  # video is too short
                #     sample_freq = int(frame_count / frame_len)

                # filter the warmup frames
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['video_action'].start()

                # collect frames
                if frame_id % sample_freq == 0:
                    # Scale image
                    scaled_img = scale(frame_rgb)
                    video_action_imgs.append(scaled_img)

                if len(video_action_imgs) == frame_len:
                    classes, scores = self.video_action_predictor.predict(
                        video_action_imgs)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['video_action'].end()

                    video_action_res = {"class": classes[0], "score": scores[0]}
                    self.pipeline_res.update(video_action_res, 'video_action')

                    print("video_action_res:", video_action_res)

                    video_action_imgs.clear()  # next clip


            self.collector.append(frame_id, self.pipeline_res)

            if frame_id > self.warmup_frame:
                self.pipe_timer.img_num += 1
                self.pipe_timer.total_time.end()
            frame_id += 1

            if self.cfg['visual']:
                _, _, fps = self.pipe_timer.get_total_time()

                im = self.visualize_video(frame_rgb, self.pipeline_res,
                                          self.collector, frame_id, fps,
                                          None, None, None,
                                          False,
                                          None)  # visualize
                video_action_res = self.pipeline_res.res_dict.get('video_action')
                
                # 保存视频片段
                if video_action_res.get("class") == 1 or valid_frame_count > 0:
                    if video_action_res.get("class") == 1:
                        writer.write(im)
                        begin_record = True
                        valid_frame_count = 10
                    elif begin_record: 
                        writer.write(im)
                        valid_frame_count -= 1
                elif valid_frame_count == 0:
                    print('save result to {}'.format(out_path))
                    
                    # 重新初始化相关变量
                    writer.release()
                    valid_frame_count = 10
                    begin_record = False
                    
                    # 创建以时间戳分割的文件夹
                    algorithm = "fight"
                    now = datetime.datetime.now()
                    day_folder = str(now.date())
                    time_tag = str(now.time())
                    parent_dir = os.path.join(root_out_dir, algorithm, day_folder)
                    if not os.path.exists(parent_dir):
                        os.mkdir(parent_dir)
                    out_path = os.path.join(parent_dir, time_tag + ".mp4")
                    
                    writer = cv2.VideoWriter(out_path, fourcc, 30, (width, height))
                else:
                    continue

    # TODO 增加可视化fps
    def visualize_video(self,
                        image_rgb,
                        result,
                        collector,
                        frame_id,
                        fps,
                        entrance=None,
                        records=None,
                        center_traj=None,
                        do_illegal_parking_recognition=False,
                        illegal_parking_dict=None):
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        video_action_res = result.get('video_action')
        if video_action_res is not None:
            video_action_score = None
            if video_action_res and video_action_res["class"] == 1:
                video_action_score = video_action_res["score"]
            image = visualize_action(
                image,
                None,
                action_visual_collector=None,
                action_text="SkeletonAction",
                video_action_score=video_action_score,
                video_action_text="Fight")

        visual_helper_for_display = []
        action_to_display = []

        if len(visual_helper_for_display) > 0:
            image = visualize_action(image, None,
                                     visual_helper_for_display,
                                     action_to_display)

        return image

def main(FLAGS):
    cfg = merge_cfg(FLAGS)  # use command params to update config
    print_arguments(cfg)

    pipeline = Pipeline(FLAGS, cfg)
    # pipeline.run()
    pipeline.run_multithreads()


if __name__ == '__main__':
    paddle.enable_static()

    # parse params from command
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU'
                            ], "device should be CPU, GPU, XPU or NPU"

    main()
