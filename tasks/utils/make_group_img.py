from typing import List, Union

import cv2
import numpy as np

from ..base import ObjectInImage


def get_mask(
    img: np.ndarray,
    index_x: Union[List[int], np.ndarray],
    index_y: Union[List[int], np.ndarray]
) -> np.ndarray:
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask[index_y, index_x] = 1.
    return mask


def add_background(
    img: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray = None
) -> np.ndarray:
    if background is None:
        background = np.zeros_like(img)
    elif background.shape != img.shape:
        background = cv2.resize(background, (img.shape[1], img.shape[0]))
    idx = np.where(mask > 0)
    background[idx] = img[idx]
    return background


def paste_face(
    body: ObjectInImage,
    face: ObjectInImage,
    out_w: int = 510,
    out_h: int = 700
) -> "ObjectInImage":
    h, w, _ = body.shape
    h_, w_, _ = face.shape

    w_face = int(w / 2)
    h_face = int(h_ * (w_face / w_))
    add_h = int(h_face * (3 / 4))
    face = face.get_resized_obj(w_face, h_face)

    pasted_img = np.zeros((h + add_h, w, 3), dtype=np.uint8)
    pasted_mask = np.zeros((h + add_h, w))
    pasted_img[add_h:] = body.img
    pasted_mask[add_h:] = body.mask
    index_x = face.index_x + int(w / 2 - w_face / 2)
    pasted_img[face.index_y, index_x] = face.img[face.index_y, face.index_x]
    pasted_mask[face.index_y, index_x] = face.mask[face.index_y, face.index_x]
    return ObjectInImage(pasted_img, pasted_mask)


def compute_lines(lines: List[int], n: int) -> List[int]:
    k = len(lines)
    if (2 * ((k+1)**2) + sum(range(1, k+1))) == n:
        return [2 * (k+1) + i for i in reversed(range(k+1))]
    for i in reversed(range(1, k)):
        if lines[i] + 1 < lines[i-1]:
            lines[i] += 1
            return lines
    lines[0] += 1
    return lines


def get_lines(n: int) -> List[int]:
    lines = [1]
    for i in range(n-1):
        lines = compute_lines(lines, i+2)
    return lines


def assemble_lines(line_imgs: List[ObjectInImage], w: int, h: int) -> "ObjectInImage":
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background_mask = np.zeros((h, w))
    k = len(line_imgs)
    bottom_h = line_imgs[0].height // 2
    for i, line_img in enumerate(reversed(line_imgs)):
        x1 = int(w / 2 - line_img.width / 2)
        y1 = h - line_img.height - (bottom_h * (k - i - 1))
        index_x = line_img.index_x + x1
        index_y = line_img.index_y + y1
        background[index_y, index_x] = line_img.img[line_img.index_y, line_img.index_x]
        background_mask[index_y, index_x] = line_img.mask[line_img.index_y, line_img.index_x]
    return ObjectInImage(background, background_mask)
