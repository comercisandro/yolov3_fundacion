from keras.models import load_model
import keras_preprocessing.image as keras_image
import os
import numpy as np
from models.yolo_v3.boundbox import BoundBox
from matplotlib import pyplot
from matplotlib.patches import Rectangle


class YoloV3:
    def __init__(self, class_threshold=0.6, nms_thresh=0.5):
        self.model = self.load_model()
        self.image = None
        self.image_path = None
        self.image_w = None
        self.image_h = None
        self.class_threshold = class_threshold
        self.nms_thresh = nms_thresh
        self.anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
        self.boxes = list()
        self.net_w = 416
        self.net_h = 416
        self.labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                       "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                       "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                       "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                       "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                       "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                       "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                       "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                       "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                       "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        self.v_boxes = list()
        self.v_labels = list()
        self.v_scores = list()


    def load_model(self):
        path = f'{os.getcwd()}/models/yolo_v3/yolov3_model.h5'
        return load_model(f'{path}')

    def load_image_pixels(self, path):
        # Obtiene el tamaño de la img
        self.image_path = path
        original_image = keras_image.load_img(path)
        # Carga de la imagen con el tamaño requerido por la red
        self.image_w, self.image_h = original_image.size
        image = keras_image.load_img(path, target_size=(self.net_w, self.net_h))
        # transformo en array
        image = keras_image.img_to_array(image)
        # normalizado de los pixels
        image = image.astype('float32')
        image /= 255.0
        # agregado de dimension para obtener una imagen
        self.image = np.expand_dims(image, 0)

    def make_prediction(self):
        self.output(self.model.predict(self.image))
        self.get_boxes()

    def output(self, prediction):
        for i in range(len(prediction)):
            self.boxes += self.decode_netout(netout=prediction[i][0],
                                             anchors=self.anchors[i])

        self.correct_yolo_boxes()
        self.do_nms()

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def decode_netout(self, netout, anchors):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5
        boxes = []
        netout[..., :2] = self.sigmoid(netout[..., :2])
        netout[..., 4:] = self.sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > self.class_threshold

        for i in range(grid_h * grid_w):
            row = i / grid_w
            col = i % grid_w
            for b in range(nb_box):

                # cuarto elemento es la probabilidad
                objectness = netout[int(row)][int(col)][b][4]
                if (objectness.all() <= self.class_threshold): continue

                # primeros 4 elmentos son x,y,w y h
                x, y, w, h = netout[int(row)][int(col)][b][:4]
                # centrados
                x = (col + x) / grid_w
                y = (row + y) / grid_h
                w = anchors[2 * b + 0] * np.exp(w) / self.net_w
                h = anchors[2 * b + 1] * np.exp(h) / self.net_h
                # el ultimo elemento es la etiqueta
                classes = netout[int(row)][col][b][5:]
                box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
                boxes.append(box)
        return boxes


    def correct_yolo_boxes(self):
        new_w, new_h = self.net_w, self.net_h
        for i in range(len(self.boxes)):
            x_offset, x_scale = (self.net_w - new_w) / 2. / self.net_w, float(new_w) / self.net_w
            y_offset, y_scale = (self.net_h - new_h) / 2. / self.net_h, float(new_h) / self.net_h
            self.boxes[i].xmin = int((self.boxes[i].xmin - x_offset) / x_scale * self.image_w)
            self.boxes[i].xmax = int((self.boxes[i].xmax - x_offset) / x_scale * self.image_w)
            self.boxes[i].ymin = int((self.boxes[i].ymin - y_offset) / y_scale * self.image_h)
            self.boxes[i].ymax = int((self.boxes[i].ymax - y_offset) / y_scale * self.image_h)

    @staticmethod
    def interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def bbox_iou(self, box1, box2):
        intersect_w = self.interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self.interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
        union = w1 * h1 + w2 * h2 - intersect
        return float(intersect) / union


    def do_nms(self):
        if len(self.boxes) > 0:
            nb_class = len(self.boxes[0].classes)
        else:
            return
        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in self.boxes])
            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                if self.boxes[index_i].classes[c] == 0: continue
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if self.bbox_iou(self.boxes[index_i], self.boxes[index_j]) >= self.nms_thresh:
                        self.boxes[index_j].classes[c] = 0

    # obtener todos los resultados por encima de cierto umbral
    def get_boxes(self):
        # enumerado de los bbox
        for box in self.boxes:
            # enumerado de todas las etiquetas
            for i in range(len(self.labels)):
                # se chequea si se supera el umbral
                if box.classes[i] > self.class_threshold:
                    self.v_boxes.append(box)
                    self.v_labels.append(self.labels[i])
                    self.v_scores.append(box.classes[i] * 100)

    def draw_boxes(self):
        # carga de la imagen
        data = pyplot.imread(self.image_path)
        # plot de la imagen
        pyplot.imshow(data)
        ax = pyplot.gca()
        # plot de cada box
        for i in range(len(self.v_boxes)):
            box = self.v_boxes[i]
            # obtener coordenadas
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            # calculo de ancho y alto del box
            width, height = x2 - x1, y2 - y1
            # se crea el rectangulo
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # se dibuja el box
            ax.add_patch(rect)
            # escritura de etiqueta y score
            label = "%s (%.3f)" % (self.v_labels[i], self.v_scores[i])
            pyplot.text(x1, y1, label, color='red')
        # mostrar imagen
        pyplot.show()
