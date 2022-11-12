from models.yolo_v3.model import YoloV3
import os
yolov3 = YoloV3()


def run_prediction(input_img):
    yolov3.load_image_pixels(input_img)
    yolov3.make_prediction()
    yolov3.draw_boxes()

if __name__ == "__main__":
    input_img = f'{os.getcwd()}/inputs/Zebra.jpg'
    run_prediction(input_img)
