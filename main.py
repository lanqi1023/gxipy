from ccd import CCD

import cv2
import logging
import numpy as np
from numpy.typing import NDArray

X_BEG, Y_BEG = 0, 0
N_PIXEL = 1600
N_ARRAY = 32

def my_process(bayer: NDArray[np.uint8]) -> NDArray[np.uint8]:
    image = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RGGB2GRAY)
    image = image[Y_BEG : Y_BEG+N_PIXEL, X_BEG : X_BEG+N_PIXEL]
    array = image.reshape(N_ARRAY, N_PIXEL // N_ARRAY, N_ARRAY, N_PIXEL // N_ARRAY).mean(axis=(1, 3))
    ...
    return array.astype(np.uint8)

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, format = '[%(levelname).1s] %(message)s')
    my_ccd = CCD()
    if my_ccd.open_camera():
        my_ccd.process = my_process
        my_ccd.capture_loop(window = None, log_fps = False)
        my_ccd.close_camera()
