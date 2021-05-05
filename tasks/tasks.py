import glob
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
import tensorflow as tf
from tf_bodypix.api import BodyPixModelPaths, download_model, load_model

from tasks.base import BaseTask, ObjectInImage
from tasks.utils.crop_window import *
from tasks.utils.detect_face import *
from tasks.utils.make_group_img import *

HEIGHT = 800
WIDTH = 1200


class CropWindowTask(BaseTask):
    def __init__(
        self,
        file_path: str,
        data_dir: str = "tmp",
    ) -> None:
        super().__init__(file_path, data_dir)

    def run(self):
        img = cv2.imread(self.file_path)
        H, W, _ = img.shape

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]

        ver_partitions = detect_vertical_partitions(gray, 200)
        hor_partitions = detect_vertical_partitions(gray.T, 200)

        ver_lines, y = detect_lines(ver_partitions, 5)
        hor_lines, x = detect_lines(hor_partitions, 5)

        ver_edges = detect_edges(ver_lines, y, H, 5)
        hor_edges = detect_edges(hor_lines, x, W, 5)

        len_hor_edge = np.median(hor_lines).astype(int)
        is_shift = check_is_shift(gray, len_hor_edge, ver_edges, hor_edges)
        rects = get_rects(len_hor_edge, ver_edges, hor_edges, is_shift, 5)

        for i, (x1, y1, x2, y2) in enumerate(rects):
            rect = img[y1: y2, x1: x2]
            cv2.imwrite(os.path.join(self.data_dir, f"rect{i}.png"), rect)


class DetectFaseTask(BaseTask):
    def __init__(
        self,
        file_path: str,
        data_dir: str = "tmp",
        threshold: float = 0.5
    ) -> None:
        super().__init__(file_path, data_dir)
        self.threshold = threshold

    def run(self):
        model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        ))
        file_list = glob.glob(os.path.join(self.data_dir, "rect*"))
        face_dict = defaultdict(dict)
        for path in file_list:
            img = tf.keras.preprocessing.image.load_img(path)
            input_array = tf.keras.preprocessing.image.img_to_array(img)
            h, w, _ = input_array.shape
            result = model.predict_single(input_array)
            mask = result.get_mask(threshold=self.threshold)
            colored_mask = result.get_colored_part_mask(mask)
            face_mask = get_face_mask(colored_mask)
            if len(face_mask[0]) > h * w * 0.01:
                gray = np.zeros((h, w), dtype=np.uint8)
                gray[face_mask] += 255
                cnt, _ = cv2.findContours(gray, 1, 2)
                max_cnt = max(cnt, key=lambda x: cv2.contourArea(x))
                ellipse = cv2.fitEllipse(max_cnt)
                face_mask = cv2.ellipse(np.zeros((h, w), dtype=np.uint8), ellipse, 1, thickness=-1)
                face_mask = np.where(face_mask > 0)
                face_dict[path] = {
                    "original_height": h,
                    "original_width": w,
                    "face_y": face_mask[0].tolist(),
                    "face_x": face_mask[1].tolist(),
                    "face_rect": [min(face_mask[1]), min(face_mask[0]),
                                  max(face_mask[1]), max(face_mask[0])]
                }
        with open(os.path.join(self.data_dir, "faces.pkl"), "wb") as f:
            pickle.dump(face_dict, f)


class MakeGroupImgTask(BaseTask):
    def __init__(
        self,
        file_path: str,
        out_path: str,
        body_path: str,
        background_path: str,
        data_dir: str = "tmp",
    ) -> None:
        super().__init__(file_path, data_dir)
        self.out_path = out_path
        self.body_path = body_path
        self.background_path = background_path

    def run(self):
        body = cv2.imread(self.body_path)
        idx_y, idx_x, _ = np.where(body != 0)
        body = ObjectInImage(body, get_mask(body, idx_x, idx_y))
        with open(os.path.join(self.data_dir, "faces.pkl"), "rb") as f:
            face_dict = pickle.load(f)
        humans = []
        for path, face_dict in face_dict.items():
            img = cv2.imread(path)

            face_x = face_dict["face_x"]
            face_y = face_dict["face_y"]
            x1, y1, x2, y2 = face_dict["face_rect"]

            mask = get_mask(img, face_x, face_y)
            face = ObjectInImage(img, mask)
            face = face.get_cropped_obj(x1, y1, x2, y2)
            out = paste_face(body, face)
            humans.append(out)
        n = len(humans)
        lines = get_lines(n)
        w = min(200, 1000 // lines[0])

        step = 0
        line_imgs = []
        for n_people in lines:
            for i in range(n_people):
                if i == 0:
                    line_img = humans[step].get_resized_obj(w)
                else:
                    line_img = line_img.left_join(humans[step], w)
                step += 1
            line_imgs.append(line_img)
        group_img = assemble_lines(line_imgs, WIDTH, HEIGHT)

        background = cv2.imread(self.background_path)
        cv2.imwrite(self.out_path,
                    add_background(group_img.img, group_img.mask, background))
