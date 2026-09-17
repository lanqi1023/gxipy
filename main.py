import sys
sys.path.append(r'D:\software\GalaxySDK\Development\Samples\Python')

import cv2
import logging
import time
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional

import gxipy

class CCD:
    def __init__(self):
        self.process:   Callable[[NDArray], NDArray]   = lambda bayer: cv2.cvtColor(bayer, cv2.COLOR_BayerRGGB2BGR)
        self.__manager: Optional[gxipy.DeviceManager]  = None
        self.__camera:  Optional[gxipy.Device]         = None
        self.__feature: Optional[gxipy.FeatureControl] = None
        self.log = logging.getLogger(__name__)

    def open_camera(self) -> bool:
        self.__manager = gxipy.DeviceManager()
        device_num, device_list = self.__manager.update_all_device_list()
        if device_num > 0:
            self.__camera  = self.__manager.open_device_by_index(1)
            self.__feature = self.__camera.get_remote_device_feature_control()
            self.__feature.get_enum_feature('UserSetSelector').set('Default')
            self.__feature.get_command_feature('UserSetLoad').send_command()
            self.log.info(f"open {device_list[0]['model_name']} @ {device_list[0]['ip']}, "
                          f"size: {self.size}, format: {self.format}, exposure: {self.exposure}")
            return True
        else:
            self.log.error('no camera found')
            return False

    def close_camera(self) -> None:
        if self.__camera is None:
            self.log.warning('camera already closed')
        else:
            try:
                self.__camera.close_device()
                self.log.info(f'close camera')
            except Exception as e:
                self.log.error(f'close camera failed: {e}')

    @property
    def size(self) -> Optional[tuple[int, int]]:
        '''
        height in [16, 4096], width in [16, 3072], must be a multiple of 16
        '''
        try:
            return self.__feature.get_int_feature("Width").get(), self.__feature.get_int_feature("Height").get()
        except Exception as e:
            self.log.error(f'get size failed: {e}')
            return None

    @size.setter
    def size(self, size: tuple[int, int]) -> None:
        try:
            self.__feature.get_int_feature("Width").set(size[0])
            self.__feature.get_int_feature("Height").set(size[1])
        except Exception as e:
            self.log.error(f'set size falied: {e}')

    @property
    def format(self) -> Optional[str]:
        '''
        'BayerRG8' or 'BayerRG12'
        '''
        try:
            return self.__feature.get_enum_feature("PixelFormat").get()[1]
        except Exception as e:
            self.log.error(f'get format failed: {e}')
            return None

    @format.setter
    def format(self, format: str) -> None:
        try:
            self.__feature.get_enum_feature("PixelFormat").set(format)
        except Exception as e:
            self.log.error(f'set format falied: {e}')

    @property
    def exposure(self) -> Optional[float]:
        '''
        from 37.0 us to 1000000.0 us
        '''
        try:
            return self.__feature.get_float_feature("ExposureTime").get()
        except Exception as e:
            self.log.error(f'get exposure time failed: {e}')
            return None

    @exposure.setter
    def exposure(self, exposure: float) -> None:
        try:
            self.__feature.get_float_feature("ExposureTime").set(exposure)
        except Exception as e:
            self.log.error(f'set exposure time falied: {e}')

    def __grab(self) -> Optional[NDArray[np.uint8 | np.uint16]]:
        buf = None
        try:
            buf = self.__camera.data_stream[0].dq_buf(timeout = 1000)
            image = buf.get_numpy_array().copy()
        except Exception as e:
            image = None
            self.log.error(f'get buffer failed: {e}')
        finally:
            if buf is not None:
                self.__camera.data_stream[0].q_buf(buf)
        return image

    def capture(self, window = None) -> Optional[NDArray[np.uint8 | np.uint16]]:
        self.__camera.stream_on()
        image = self.__grab()
        if image is not None:
            image = self.process(image)
            if window is not None:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.imshow(window, image)
                cv2.waitKey(0)
        self.__camera.stream_off()
        return image

    def capture_loop(self, window = 'ccd') -> None:
        if window is not None:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        self.__camera.stream_on()
        count = 0
        begin = time.perf_counter()
        try:
            while True:
                image = self.__grab()
                if image is not None:
                    image = self.process(image)
                    count += 1
                    now = time.perf_counter()
                    if now - begin >= 0.1:
                        print(f'[I] get frame {image.shape}, fps: {count / (now - begin):.2f}', end = '\r', flush = True)
                        count = 0
                        begin = time.perf_counter()
                    if window is not None:
                        cv2.imshow(window, image)
                        cv2.waitKey(1)
                        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                            window = None
                            break
        except KeyboardInterrupt:
            pass
        finally:
            print()
            self.log.info('stop capture loop')
            if window is not None:
                cv2.destroyWindow(window)
            self.__camera.stream_off()

if __name__ == '__main__':
    logging.basicConfig(
        level  = logging.INFO,
        format = '[%(levelname).1s] %(message)s'
    )

    ccd = CCD()
    if ccd.open_camera():
        # ccd.format   = 'BayerRG12'
        # ccd.size     = (1600, 1600)
        # ccd.exposure = 1000
        # ccd.process  = lambda bayer: cv2.cvtColor((bayer >> 4).astype(np.uint8), cv2.COLOR_BayerRGGB2BGR)
        ccd.capture_loop()
        ccd.close_camera()
