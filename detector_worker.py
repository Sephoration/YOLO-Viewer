"""
独立推理工作器
使用Qt信号槽与播放器通信、处理YOLO推理任务
"""

import time
import traceback
from queue import Queue
from typing import Optional, Dict, Any

from PySide6.QtCore import QThread, Signal, Slot, QMutex, QWaitCondition
from PySide6.QtGui import QImage
import numpy as np


class DetectorWorker(QThread):
    """
    独立推理工作器
    从播放器接收帧，进行YOLO推理，返回结果
    """
    
    # 信号定义
    frame_processed = Signal(QImage)  # 处理后的帧 → UI显示
    detection_stats = Signal(dict)    # 统计信息 → UI统计面板
    status_updated = Signal(str)      # 状态更新 → 日志/状态栏
    error_occurred = Signal(str)      # 错误信息
    processing_complete = Signal()    # 处理完成
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # YOLO处理器
        self.yolo_processor = None
        
        # 处理控制
        self.is_processing = False
        self.is_paused = False
        self.stop_requested = False
        
        # 帧队列
        self.frame_queue = Queue(maxsize=5)  # 限制队列大小，避免内存爆炸
        
        # 处理间隔
        self.process_interval = 1  # 每N帧处理一次
        self.frame_counter = 0
        
        # 性能统计
        self.total_frames_processed = 0
        self.total_inference_time = 0.0
        self.start_time = None
        
        # 线程同步
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # 当前处理的帧（用于UI显示）
        self.current_original_frame = None
        self.current_processed_frame = None
    
    def set_yolo_processor(self, processor):
        """设置YOLO处理器"""
        self.mutex.lock()
        try:
            self.yolo_processor = processor
            if processor:
                self.status_updated.emit(f"YOLO处理器已设置: {type(processor).__name__}")
        finally:
            self.mutex.unlock()
    
    def set_process_interval(self, interval: int):
        """设置处理间隔（每N帧处理一次）"""
        self.mutex.lock()
        try:
            self.process_interval = max(1, interval)
        finally:
            self.mutex.unlock()
    
    def update_parameters(self, **params):
        """更新推理参数"""
        self.mutex.lock()
        try:
            if self.yolo_processor and hasattr(self.yolo_processor, 'update_params'):
                # 处理参数名映射
                yolo_params = {}
                if 'iou_threshold' in params:
                    yolo_params['iou'] = params['iou_threshold']
                if 'confidence_threshold' in params:
                    yolo_params['conf'] = params['confidence_threshold']  # 改为conf
                if 'line_width' in params:
                    yolo_params['line_width'] = params['line_width']
                
                # 如果有其他参数，直接添加（向后兼容）
                for key, value in params.items():
                    if key not in ['iou_threshold', 'confidence_threshold']:
                        yolo_params[key] = value
                
                success = self.yolo_processor.update_params(**yolo_params)
                if success:
                    self.status_updated.emit(f"推理参数已更新: {yolo_params}")
                else:
                    self.status_updated.emit("参数更新失败")
        except Exception as e:
            self.error_occurred.emit(f"更新参数失败: {str(e)}")
        finally:
            self.mutex.unlock()
    
    @Slot(np.ndarray)
    def on_frame_received(self, frame: np.ndarray):
        """
        接收来自播放器的原始帧
        使用@Slot装饰器确保正确的Qt信号连接
        """
        if not self.is_processing or self.stop_requested:
            return
        
        try:
            # 确保帧是有效的numpy数组
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                return
            
            # 复制帧数据，避免引用问题
            frame_copy = frame.copy() if hasattr(frame, 'copy') else frame
            
            # 更新当前原始帧（用于可能的实时显示）
            self.current_original_frame = frame_copy
            
            # 如果队列未满，加入队列
            if not self.frame_queue.full():
                self.frame_queue.put(frame_copy)
            elif hasattr(self, 'skip_frames_if_busy') and self.skip_frames_if_busy:
                # 队列满且允许跳过帧，丢弃最老的帧
                try:
                    self.frame_queue.get_nowait()  # 丢弃一个
                    self.frame_queue.put(frame_copy)  # 加入新的
                except:
                    pass
            
            # 唤醒等待的线程
            self.condition.wakeAll()
            
        except Exception as e:
            print(f"接收帧失败: {e}")
    
    def start_processing(self):
        """开始处理"""
        self.mutex.lock()
        try:
            if self.is_processing:
                self.status_updated.emit("已经在处理中")
                return
            
            # 检查YOLO处理器
            if self.yolo_processor is None:
                self.error_occurred.emit("请先加载YOLO模型")
                return False
            
            # 重置状态
            self.is_processing = True
            self.stop_requested = False
            self.is_paused = False
            
            # 清空队列
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break
            
            # 重置统计
            self.total_frames_processed = 0
            self.total_inference_time = 0.0
            self.start_time = time.time()
            
            # 启动线程
            if not self.isRunning():
                self.start()
            else:
                self.condition.wakeAll()
            
            self.status_updated.emit("开始推理处理")
            return True
            
        finally:
            self.mutex.unlock()
    
    def stop_processing(self):
        """停止处理"""
        self.mutex.lock()
        try:
            if not self.is_processing:
                return
            
            self.stop_requested = True
            self.is_processing = False
            self.condition.wakeAll()
            
            # 清空队列
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break
            
            self.status_updated.emit("停止推理处理")
            
        finally:
            self.mutex.unlock()
    
    def pause_processing(self):
        """暂停处理"""
        self.mutex.lock()
        try:
            if self.is_processing and not self.is_paused:
                self.is_paused = True
                self.status_updated.emit("推理处理已暂停")
        finally:
            self.mutex.unlock()
    
    def resume_processing(self):
        """恢复处理"""
        self.mutex.lock()
        try:
            if self.is_processing and self.is_paused:
                self.is_paused = False
                self.condition.wakeAll()
                self.status_updated.emit("恢复推理处理")
        finally:
            self.mutex.unlock()
    
    def run(self):
        """线程主循环"""
        self.status_updated.emit("推理工作器线程启动")
        
        while not self.stop_requested:
            # 检查暂停状态
            self.mutex.lock()
            if self.is_paused:
                self.condition.wait(self.mutex)
                self.mutex.unlock()
                if self.stop_requested:
                    break
                continue
            self.mutex.unlock()
            
            try:
                # 从队列获取帧（阻塞等待，最多1秒）
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except:
                    # 队列为空，继续等待
                    continue
                
                # 检查帧有效性
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    continue
                
                # 检查是否应该处理此帧（根据间隔）
                self.frame_counter += 1
                if self.frame_counter % self.process_interval != 0:
                    # 跳过此帧，但可以显示原始帧
                    self._emit_original_frame(frame)
                    continue
                
                # 执行推理
                inference_start = time.time()
                
                if self.yolo_processor is None:
                    # 没有处理器，显示原始帧
                    self._emit_original_frame(frame)
                    continue
                
                # 处理帧
                result = self.yolo_processor.process_frame(frame)
                
                inference_time = time.time() - inference_start
                self.total_inference_time += inference_time
                self.total_frames_processed += 1
                
                # 处理结果
                if isinstance(result, dict):
                    # 提取处理后的图像和统计信息
                    processed_image = result.get('image', frame)
                    stats_data = result.get('stats', {})
                    
                    # 添加推理时间到统计
                    stats_data['inference_time'] = inference_time * 1000  # 转换为毫秒
                    stats_data['fps'] = 1.0 / inference_time if inference_time > 0 else 0
                    
                    # 发送统计信息
                    self.detection_stats.emit(stats_data)
                    
                    # 转换为QImage并发送
                    qimg = self._numpy_to_qimage(processed_image)
                    self.frame_processed.emit(qimg)
                    
                    # 保存当前处理后的帧
                    self.current_processed_frame = processed_image.copy() if hasattr(processed_image, 'copy') else processed_image
                    
                else:
                    # 结果不是字典格式，发送原始帧
                    self._emit_original_frame(frame)
                
                # 定期报告性能
                if self.total_frames_processed % 30 == 0:
                    avg_inference_time = self.total_inference_time / self.total_frames_processed
                    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
                    self.status_updated.emit(f"推理性能: {avg_inference_time*1000:.1f}ms/帧, {fps:.1f} FPS")
                
            except Exception as e:
                error_msg = f"推理处理失败: {str(e)}"
                self.error_occurred.emit(error_msg)
                traceback.print_exc()
        
        # 处理完成
        self._emit_final_stats()
        self.processing_complete.emit()
        self.status_updated.emit("推理工作器线程结束")
    
    def _emit_original_frame(self, frame: np.ndarray):
        """发送原始帧"""
        try:
            qimg = self._numpy_to_qimage(frame)
            self.frame_processed.emit(qimg)
        except Exception as e:
            print(f"发送原始帧失败: {e}")
    
    def _numpy_to_qimage(self, frame: np.ndarray) -> QImage:
        """将numpy数组转换为QImage"""
        try:
            import cv2
            
            # 确保是有效的numpy数组
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                height, width = 480, 640
                empty_image = np.zeros((height, width, 3), dtype=np.uint8)
                frame = empty_image
            
            # 确保是3通道图像
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.shape[2] != 3:
                # 如果不是3通道，转换为BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 确保内存连续
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            
            return QImage(
                frame.data, width, height, bytes_per_line,
                QImage.Format_BGR888
            ).copy()
            
        except Exception as e:
            print(f"转换numpy到QImage失败: {e}")
            # 返回一个空的QImage
            return QImage(640, 480, QImage.Format_RGB888)
    
    def _emit_final_stats(self):
        """发送最终统计信息"""
        try:
            if self.total_frames_processed > 0:
                avg_inference_time = self.total_inference_time / self.total_frames_processed
                avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
                
                final_stats = {
                    'total_frames': self.total_frames_processed,
                    'avg_inference_time': avg_inference_time * 1000,  # 毫秒
                    'avg_fps': avg_fps,
                    'total_time': time.time() - self.start_time if self.start_time else 0
                }
                
                self.detection_stats.emit(final_stats)
        except Exception as e:
            print(f"发送最终统计失败: {e}")
    
    def get_current_frame(self, processed: bool = False):
        """获取当前帧（原始或处理后的）"""
        self.mutex.lock()
        try:
            if processed and self.current_processed_frame is not None:
                return self.current_processed_frame.copy() if hasattr(self.current_processed_frame, 'copy') else self.current_processed_frame
            elif self.current_original_frame is not None:
                return self.current_original_frame.copy() if hasattr(self.current_original_frame, 'copy') else self.current_original_frame
            return None
        finally:
            self.mutex.unlock()