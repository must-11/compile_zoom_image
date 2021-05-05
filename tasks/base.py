from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from typing import List

import cv2
import numpy as np


class BaseTask(metaclass=ABCMeta):
    """
    ベースとなるクラス
    """
    def __init__(self, file_path: str, data_dir: str = "tmp") -> None:
        self.data_dir = data_dir
        self.file_path = file_path

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError()


class Compose:
    def __init__(self, tasks: List[BaseTask]) -> None:
        self.tasks = tasks

    def run(self) -> None:
        for t in self.tasks:
            t.run()


class ObjectInImage:
    def __init__(
        self,
        img: np.ndarray,
        mask: np.ndarray
    ) -> None:
        self.img = img
        self.mask = mask
        self.shape = img.shape
        self.height = self.shape[0]
        self.width = self.shape[1]
        index = np.where(mask > 0)
        self.index_x = index[1]
        self.index_y = index[0]

    def get_resized_obj(self, w: int, h: int = None) -> ObjectInImage:
        if h is None:
            h = int(self.height * (w / self.width))
        resized_img = cv2.resize(self.img, (w, h))
        resized_mask = cv2.resize(self.mask, (w, h))
        return ObjectInImage(resized_img, resized_mask)

    def get_cropped_obj(self, x1: int, y1: int, x2: int, y2: int) -> ObjectInImage:
        cropped_img = self.img[y1: y2, x1: x2]
        cropped_mask = self.mask[y1: y2, x1: x2]
        return ObjectInImage(cropped_img, cropped_mask)

    def left_join(self, right: ObjectInImage, w: int) -> ObjectInImage:
        right = right.get_resized_obj(w)
        h = max(self.height, right.height)
        w = w + self.width
        joined_img = np.zeros((h, w, 3), dtype=np.uint8)
        joined_mask = np.zeros((h, w))

        joined_img[-self.height:, :self.width] = self.img
        joined_img[-right.height:, self.width:] = right.img
        joined_mask[-self.height:, :self.width] = self.mask
        joined_mask[-right.height:, self.width:] = right.mask
        return ObjectInImage(joined_img, joined_mask)
