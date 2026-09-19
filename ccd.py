import sys
sys.path.append(r'D:\software\GalaxySDK\Development\Samples\Python')
import gxipy

import cv2
import logging
import numpy as np
import time
from numpy.typing import NDArray
from threading import Thread
from typing import Callable, Optional

class CCD:
    '''
    Daheng CCD class, based on gxipy SDK
    ---
    0. config logging
    ```
    logging.basicConfig(level = logging.INFO, format = '[%(levelname).1s] %(message)s')
    ```

    1. open camera and capture an image
    ```
    ccd = CCD()
    if ccd.open_camera():
        image = ccd.capture(window = 'window name')
        ...
        ccd.close_camera()
    ```

    2. run capture loop
    ```
    ccd = CCD()
    if ccd.open_camera():
        ccd.capture_loop(window = 'window name', log_fps = True)
        ccd.close_camera()
    ```

    3. get / set parameters (should open_camera first)
    ```
    ccd = CCD()
    if ccd.open_camera():
        ccd.list_feature()
        ccd.size     = (1600, 1600)
        ccd.exposure = 1000 # us
        ccd.format   = 'BayerRG12'
        ccd.process  = CCD.DEFAULT_PROCESS_RG12
    ```
    '''
    DEFAULT_PROCESS_RG8  = lambda bayer: cv2.cvtColor(bayer, cv2.COLOR_BAYER_RGGB2BGR)
    DEFAULT_PROCESS_RG12 = lambda bayer: cv2.cvtColor((bayer >> 4).astype(np.uint8), cv2.COLOR_BAYER_RGGB2BGR)

    def __init__(self):
        self.process:   Callable[[NDArray], NDArray]   = CCD.DEFAULT_PROCESS_RG8
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
        from 37.0 us to 1000000.0 us, continuously
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

    def capture(self, window = None) -> Optional[NDArray[np.uint8 | np.uint16]]:
        self.__camera.stream_on()
        image = self.__camera.data_stream[0].get_image()
        if image is not None:
            image = self.process(image)
            if window is not None:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.imshow(window, image)
                cv2.waitKey(0)
        self.__camera.stream_off()
        return image

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

    def capture_loop(self, window = 'untitled', log_fps: bool = True) -> None:
        if window is not None:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        self.__camera.stream_on()
        if log_fps:
            count = 0
            begin = time.perf_counter()
        try:
            while True:
                image = self.__grab()
                if image is not None:
                    image = self.process(image)
                    if log_fps:
                        count += 1
                        duration = time.perf_counter() - begin
                        if duration >= 1:
                            print(f'[I] get frame {image.shape}, fps: {count / duration:.2f}', end = '\r', flush = True)
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
            if window is not None:
                cv2.destroyWindow(window)
            self.__camera.stream_off()

    def list_feature(self) -> None:
        for name, feature in vars(self.__camera).items():
            if isinstance(feature, gxipy.Feature):
                if feature.is_implemented():
                    if isinstance(feature, gxipy.IntFeature):
                        print(name, '= '
                            f"{feature.get() if feature.is_readable() else '?'} "
                            f"{feature.get_range().get('unit, ', '')}"
                            f"in [{feature.get_range().get('min')}, "
                            f"{feature.get_range().get('max')}]")
                    if isinstance(feature, gxipy.FloatFeature):
                        print(name, '= '
                            f"{feature.get() if feature.is_readable() else '?'} "
                            f"{feature.get_range().get('unit, ', '')}"
                            f"in [{feature.get_range().get('min')}, "
                            f"{feature.get_range().get('max')}]")
                    elif isinstance(feature, gxipy.EnumFeature):
                        print(name, '= '
                            f"{feature.get()[1] if feature.is_readable() else '?'} "
                            f"in {list(feature.get_range())}")
                    elif isinstance(feature, gxipy.BoolFeature):
                        print(name, '= '
                            f"{feature.get() if feature.is_readable() else '?'}")
                    elif isinstance(feature, gxipy.StringFeature):
                        print(name, '= '
                            f"'{feature.get() if feature.is_readable() else '?'}'")
                    elif isinstance(feature, gxipy.BufferFeature):
                        print('(buffer)', name)

if __name__ == '__main__':
    logging.basicConfig(
        level  = logging.INFO,
        format = '[%(levelname).1s] %(message)s'
    )

    ccd = CCD()
    if ccd.open_camera():
        ccd.list_feature()
        # ccd.size     = (1600, 1600)
        # ccd.exposure = 1000
        # ccd.format   = 'BayerRG12'
        # ccd.process  = CCD.DEFAULT_PROCESS_RG12
        ccd.capture_loop()
        ccd.close_camera()
