"""
帧抓取器
专门负责从播放器抓取帧并发送给YOLO推理引擎
"""

import time
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
import cv2
import numpy as np

class FrameGrabberThread(QThread):
    frame_processed = Signal(QImage)
    processing_complete = Signal(dict)
    error_occurred = Signal(str)
    status_update = Signal(str)
    finished = Signal()

    def __init__(self, player):
        super().__init__()
        self.player = player
        self.yolo = None
        self._every = 5
        self._stop = False
        self._n = self._det = 0
        self._t_infer = 0.0

    # -------------- 外部原接口 --------------
    def set_yolo_processor(self, yolo): self.yolo = yolo
    def start_grabbing(self, grab_interval=5):
        self._every, self._stop = max(1, grab_interval), False
        self._n = self._det = 0; self._t_infer = 0.0
        if not self.isRunning(): self.start()
    def stop_grabbing(self):
        self._stop = True; self.wait(500); self.finished.emit()
    def get_statistics_summary(self):
        return {"total_frames_processed": self._n, "total_detections": self._det,
                "avg_inference_time_per_frame": (self._t_infer/self._n*1000 if self._n else 0)}

    # -------------- 主循环 ----------------
    def run(self):
        while not self._stop:
            frame = self.player.get_current_frame()
            if frame is None: self.msleep(10); continue
            self._n += 1
            if self._n % self._every: continue
            ok, infer, det, out = self._proc(frame)
            if ok:
                self._t_infer += infer; self._det += det
                self.frame_processed.emit(self._to_qimage(out))
                self.processing_complete.emit({"detection_count": det,
                                               "inference_time": infer*1000,
                                               "fps": 1.0/(infer+1e-3)})
        self.finished.emit()

    # -------------- 内部辅助 --------------
    def _proc(self, frame):
        if self.yolo is None: return True, 0.0, 0, frame
        try:
            t0 = time.time()
            r = self.yolo.process_frame(frame)
            infer = time.time() - t0
            return True, infer, r.get("detection_count", 0), r.get("image", frame)
        except Exception as e:
            self.error_occurred.emit(str(e)); return False, 0.0, 0, frame
    @staticmethod
    def _to_qimage(arr):
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape; bpl = ch * w
        return QImage(rgb.data, w, h, bpl, QImage.Format_RGB888).copy()