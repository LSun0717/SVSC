source /usr/local/miniconda3/bin/activate sadc-paddle

# nohup python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_fight_recognition.yml \
#                                                    --rtsp rtsp://admin:my550025@192.168.1.64:554/h264/ch1/main/av_stream \
#                                                    --pushurl rtmp://192.168.1.106:1935/out_sl01 \
#                                                    --device=gpu &
python pipeline_slim.py --config config/infer_cfg_pphuman.yml \
                                                   --rtsp rtsp://admin:my550025@192.168.1.64:554/h264/ch1/main/av_stream \
                                                   --pushurl rtmp://192.168.1.106:1935/out_sl01 \
                                                   --device=gpu 