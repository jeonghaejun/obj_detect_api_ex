import cv2
from video import Video
from objdetect import ObjDetectApi
import time

PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

api = ObjDetectApi(MODEL_NAME, PATH_TO_LABELS)


def detect(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_dict = api.inference_image(frame_rgb)
    labeled_image = api.visualize(frame_rgb, output_dict)
    labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', labeled_image)
    key = cv2.waitKey(1)
    if key == 27:
        return False
    else:
        return True
with Video(file="vtest.avi") as v:
# with Video(device=0) as v:
    for image in v:
        start_time=time.time()
        if not detect(image):break
        end_time = time.time()
        fps = 1 / (end_time-start_time)
        print(f'fps: {fps}')
            