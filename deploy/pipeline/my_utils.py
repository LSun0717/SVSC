from collections import OrderedDict
import datetime
import os

import cv2  
  
# file_names = ['infer_cfg_illegal_parking.yml', 'infer_cfg_smoking.yml', 'infer_cfg_fight_recognition.yml',  
#                'infer_cfg_vehicle_attr.yml', 'infer_cfg_calling.yml', 'infer_cfg_fall_down.yml',  
#                'infer_cfg_vehicle_plate.yml', 'infer_cfg_reid.yml', 'infer_cfg_vehicle_violation.yml',  
#                'infer_cfg_human_mot.yml', 'infer_cfg_human_attr.yml']
    
def getConfigFilesList(file_names):
    file_dict = OrderedDict(zip(range(len(file_names)), file_names))  
    
    new_file_dict = {}  
    for key, value in file_dict.items():  
        new_key = value[len("infer_cfg_"):-4]  
        new_file_dict[new_key] = value  
    return new_file_dict

if __name__ == "__main__":
    
    # capture = cv2.VideoCapture("rtsp://admin:my550025@192.168.1.64:554/h264/ch1/main/av_stream")
    
    
    root_out_dir = "/home/lab102/project/PaddleDetection/output"
    algorithm = "fight"
    now = datetime.datetime.now()
    day_folder = str(now.date())
    time_tag = str(now.time())
    out_path = os.path.join(root_out_dir, algorithm, day_folder, time_tag + ".mp4")

    # out_path = root_out_dir + "/fight.mp4"
    # Open a video capture object 
    cap = cv2.VideoCapture("rtsp://admin:my550025@192.168.1.64:554/h264/ch1/main/av_stream") 

    # Define the codec and create a VideoWriter object 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (640, 480)) 
    print(out_path)
    frame_count = 100
    # Capture video frames and write them to the file 
    while cap.isOpened(): 
        ret, frame = cap.read() 
        if ret: 
            # Flip the frame horizontally 
            frame = cv2.flip(frame, 1) 
            frame_count -= 1
            # Write the frame to the output file 
            out.write(frame) 
            if (frame_count < 0):
                break
        
    # Release the resources 
    cap.release() 
    out.release() 