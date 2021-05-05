from typing import Tuple

import numpy as np


def get_face_mask(colored_mask: np.ndarray) -> Tuple[np.ndarray]:
    return np.where(
        ((colored_mask[:, :, 0] == 143) &
         (colored_mask[:, :, 1] == 61) &
         (colored_mask[:, :, 2] == 178)) |
        ((colored_mask[:, :, 0] == 110) &
         (colored_mask[:, :, 1] == 64) &
         (colored_mask[:, :, 2] == 170))
    )
