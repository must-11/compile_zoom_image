import sys
from typing import List, Tuple, Union

import numpy as np


def detect_vertical_partitions(gray: np.ndarray, threshold: float = 200) -> np.ndarray:
    y = []
    for row in gray:
        y.append(np.mean(row))
    return np.where(np.array(y) > threshold)[0]


def detect_lines(
    partitions: np.ndarray,
    threshold: float = 5
) -> Union[List[int], List[List[int]]]:
    last = -1
    len_list = []
    coordinates_list = []
    for i in partitions:
        if last == -1:
            last = i
        elif (i - last) > threshold:
            len_list.append(i - last - 1)
            coordinates_list.append([last + 1, i - 1])
        last = i
    return len_list, coordinates_list


def detect_edges(
    len_list: List[int],
    coordinates_list: List[List[int]],
    len_img: int,
    w_partitions: int = 5
) -> List[List[int]]:
    if len(len_list) != len(coordinates_list):
        sys.stderr.write("ERROR: The length of arrays do not match!")
        sys.exit()
    len_edge = np.median(len_list).astype(int)
    edges = []
    half_partitions = w_partitions // 2 + 1
    z_max = 0
    for length, (z1, z2) in zip(len_list, coordinates_list):
        if (abs(length - len_edge) / len_edge) < 0.05:
            if (z1 - z_max + half_partitions) > len_edge:
                edges.append([z1 - len_edge + half_partitions, z1 - half_partitions])
            edges.append([z1, z2])
            z_max = z2 + half_partitions

    if (len_img - z_max + half_partitions) > len_edge:
        edges.append([z_max + half_partitions, z_max + len_edge + half_partitions])
    return edges


def check_is_shift(
    gray: np.ndarray,
    len_edge: int,
    ver_edges: List[List[int]],
    hor_edges: List[List[int]]
) -> bool:
    y1, y2 = ver_edges[-1]
    on_shift = []
    off_shift = []
    for i, (x1, x2) in enumerate(hor_edges):
        off_shift.append(gray[y1: y2, x2].mean())
        on_shift.append(gray[y1: y2, x1 + len_edge//2 + 3].mean())
        if i != len(hor_edges) - 1:
            on_shift.append(gray[y1: y2, x2 + len_edge//2 + 3].mean())
    return np.mean(off_shift) < np.mean(on_shift)


def get_rects(
    len_edge: int,
    ver_edges: List[List[int]],
    hor_edges: List[List[int]],
    is_shift: bool = False,
    w_partitions: int = 5
) -> List[Tuple[int, int, int, int]]:
    half_partitions = w_partitions // 2 + 1
    rects = []
    for j, (y1, y2) in enumerate(ver_edges):
        for i, (x1, x2) in enumerate(hor_edges):
            if (j == len(ver_edges) - 1) & (is_shift):
                if i == len(hor_edges) - 1:
                    break
                else:
                    x1 += len_edge//2 + half_partitions
                    x2 += len_edge//2 + half_partitions
            rects.append((x1, y1, x2, y2))
    return rects
