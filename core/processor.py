from ultralytics import YOLO
import cv2

from .state import State
from .analysis import draw_box, resize_high_quality

def run_pipeline(model_path, image_path, conf):
    state = State()
    state.model = YOLO(model_path)
    state.min_conf = conf
    state.image_list = [image_path]

    for path in state.image_list:
        frame = cv2.imread(path)
        frame = resize_high_quality(frame)

        processed = draw_box(state, frame)

        cv2.imshow("Result", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
