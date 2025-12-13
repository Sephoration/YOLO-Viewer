"""
逻辑协调控制器
负责连接UI、播放器、抓取器和YOLO推理引擎
使用QThread管理视频播放线程
"""

import os
import sys
import time
import traceback
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from PySide6.QtCore import QObject, Signal, Qt, QThread, QMutex, QWaitCondition, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QMessageBox, QFileDialog

from window_ui import YOLOMainWindowUI
from baseDetect import baseDetect

# 导入配置文件
from config import AppConfig

# 添加QMutexLocker辅助类
class QMutexLocker:
    """用于简化QMutex的使用，类似于C++的QMutexLocker"""
    def __init__(self, mutex):
        self.mutex = mutex
        self.mutex.lock()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mutex.unlock()


class VideoPlayerThread(QThread):
    """
    使用Qt线程机制、提供更可靠的线程管理
    """
    
    frame_ready = Signal(QImage)  # 帧就绪信号（用于显示）
    status_update = Signal(str)   # 状态更新
    progress_updated = Signal(int, int, float)  # 进度更新信号 (当前帧, 总帧数, 当前时间)
    finished = Signal()           # 播放完成
    
    def __init__(self):
        super().__init__()
        self.playing = False
        self.paused = False
        self.cap = None
        self.current_frame = None  # 当前帧（numpy array）
        self.frame_mutex = QMutex()  # 用于帧数据访问的互斥锁
        self.cap_mutex = QMutex()    # 专门用于保护cap对象操作的互斥锁
        self.wait_condition = QWaitCondition()  # 用于暂停控制
        
        # 视频信息
        self.total_frames = 0
        self.current_frame_num = 0
        self.fps = AppConfig.VIDEO_SETTINGS['min_fps']  # 使用配置中的默认FPS
        self.duration = 0.0
        
        # 播放源
        self.video_path = None
        self.camera_id = AppConfig.CAMERA_SETTINGS['default_camera_id']  # 使用配置中的默认摄像头ID
        self.play_mode = None  # 'video' or 'camera'
        
        # 优化开关
        self._use_grab = AppConfig.PLAYER_SETTINGS['use_grab_method']  # 使用配置中的优化设置
        
        # 禁用OpenCV的多线程功能以避免FFmpeg线程冲突
        self._disable_cv2_multithreading()
        
    def _disable_cv2_multithreading(self):
        """禁用OpenCV的多线程功能以避免与FFmpeg的线程冲突"""
        try:
            import cv2
            cv2.setNumThreads(0)
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    
    def play_video(self, video_path: str):
        """播放视频文件"""
        self.stop()
        
        with QMutexLocker(self.frame_mutex):
            self.video_path = video_path
            self.camera_id = None
            self.play_mode = 'video'
            self.playing = True
            self.paused = False
        
        # 如果线程未启动或已结束，重新启动
        if not self.isRunning():
            self.start(QThread.NormalPriority)
    
    def play_camera(self, camera_id: int = 0):
        """播放摄像头"""
        self.stop()
        
        with QMutexLocker(self.frame_mutex):
            self.video_path = None
            self.camera_id = camera_id
            self.play_mode = 'camera'
            self.playing = True
            self.paused = False
        
        # 如果线程未启动或已结束，重新启动
        if not self.isRunning():
            self.start(QThread.NormalPriority)
    
    def stop(self):
        """停止播放"""
        # 安全地设置停止标志
        with QMutexLocker(self.frame_mutex):
            self.playing = False
            self.paused = False
            self.wait_condition.wakeAll()  # 唤醒可能在等待的线程
        
        # 等待线程结束
        if self.isRunning():
            self.wait(3000)  # 增加等待时间到3秒
        
        # 安全地释放资源
        with QMutexLocker(self.cap_mutex):
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    traceback.print_exc()
                finally:
                    self.cap = None
        
        self.finished.emit()
    
    def pause(self):
        """暂停播放"""
        with QMutexLocker(self.frame_mutex):
            if not self.playing:
                return
            self.paused = True
    
    def resume(self):
        """继续播放"""
        with QMutexLocker(self.frame_mutex):
            if not self.playing:
                return
            self.paused = False
            self.wait_condition.wakeAll()  # 唤醒等待的线程
    
    def get_current_frame(self):
        """获取当前帧（用于抓取）"""
        with QMutexLocker(self.frame_mutex):
            if self.current_frame is not None and hasattr(self.current_frame, 'copy'):
                return self.current_frame.copy()
            return None
    
    def seek_frame(self, target_frame: int):
        """跳转到指定帧"""
        try:
            import cv2
            
            # 首先检查cap对象是否有效
            with QMutexLocker(self.cap_mutex):
                if not self.cap or not self.cap.isOpened() or self.play_mode != 'video':
                    return
                
                # 跳转到目标帧
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(target_frame)))
                except Exception as e:
                    self.status_update.emit(f"跳转失败: {str(e)}")
                    return
                
                # 读取帧
                if self._use_grab:
                    try:
                        self.cap.grab()
                        ret, frame = self.cap.retrieve()
                    except Exception:
                        ret = False
                else:
                    try:
                        ret, frame = self.cap.read()
                    except Exception:
                        ret = False
                
                if not ret or frame is None:
                    return
                
                # 更新帧计数
                try:
                    pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if pos >= 0:
                        self.current_frame_num = pos
                    else:
                        self.current_frame_num = max(0, int(target_frame))
                except:
                    self.current_frame_num = max(0, int(target_frame))
                
                # 更新当前帧数据
                with QMutexLocker(self.frame_mutex):
                    self.current_frame = frame.copy()
            
            current_time = self.current_frame_num / max(1.0, self.fps)
            self.progress_updated.emit(self.current_frame_num, self.total_frames, current_time)
            
            if not self.current_frame.flags['C_CONTIGUOUS']:
                frame_copy = self.current_frame.copy()
            else:
                frame_copy = self.current_frame
            height, width, channel = frame_copy.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame_copy.data, width, height, bytes_per_line, QImage.Format_BGR888).copy()
            self.frame_ready.emit(q_img)
        except Exception:
            traceback.print_exc()
    
    def run(self):
        """线程运行函数"""
        try:
            if self.play_mode == 'video' and self.video_path:
                self._video_playback()
            elif self.play_mode == 'camera' and self.camera_id is not None:
                self._camera_playback()
        except Exception as e:
            self.status_update.emit(f"播放错误: {str(e)}")
            traceback.print_exc()
        finally:
            self.playing = False
            self.paused = False
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            self.finished.emit()
    
    def _video_playback(self):
        """视频播放实现"""
        import time
        time.sleep(0.05)

        try:
            import cv2
            import numpy as np

            # 初始化视频捕获对象
            with QMutexLocker(self.cap_mutex):
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    self.status_update.emit(f"无法打开视频文件: {self.video_path}")
                    return

                # 尝试设置较小的内部缓冲
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, AppConfig.VIDEO_SETTINGS['default_buffer_size'])  # 使用配置中的缓冲大小
                except Exception:
                    pass

                # 获取视频信息
                try:
                    self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                except:
                    self.total_frames = 1000  # 默认帧数
                
                try:
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                except:
                    self.fps = AppConfig.VIDEO_SETTINGS['min_fps']
                
                if self.fps <= 0:
                    self.fps = AppConfig.VIDEO_SETTINGS['min_fps']

                if self.total_frames > 0:
                    self.duration = self.total_frames / self.fps
                else:
                    self.total_frames = 1000

                try:
                    self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                except:
                    self.current_frame_num = 0

                # 预先读取几帧以确保初始化正常
                if self._use_grab:
                    # 从配置中获取预加载帧数或使用默认值
                    preload_frames = getattr(AppConfig.VIDEO_SETTINGS, 'frame_preload_count', 2)
                    for _ in range(preload_frames):
                        try:
                            self.cap.grab()
                        except Exception:
                            break

            self.status_update.emit(f"开始播放视频: {os.path.basename(self.video_path)}")
            self.status_update.emit(f"总帧数: {self.total_frames}, FPS: {self.fps:.2f}")

            frame_interval = 1.0 / self.fps if self.fps > 0 else 0.033

            while self.playing:
                # 检查暂停状态
                with QMutexLocker(self.frame_mutex):
                    if self.paused:
                        self.wait_condition.wait(self.frame_mutex)
                        if not self.playing:
                            break
                        continue

                loop_start = time.time()

                # 读取帧，所有cap操作都在锁保护下进行
                frame = None
                frame_read_ok = False
                
                with QMutexLocker(self.cap_mutex):
                    if not self.cap or not self.cap.isOpened():
                        break
                    
                    try:
                        if self._use_grab:
                            ok = self.cap.grab()
                            if ok:
                                ret, frame = self.cap.retrieve()
                                frame_read_ok = ret and frame is not None
                        else:
                            ret, frame = self.cap.read()
                            frame_read_ok = ret and frame is not None
                            
                        # 帧读取失败时的处理
                        if not frame_read_ok:
                            try:
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                with QMutexLocker(self.frame_mutex):
                                    self.current_frame_num = 0
                                continue
                            except Exception:
                                break
                        
                        # 更新帧计数
                        try:
                            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                            with QMutexLocker(self.frame_mutex):
                                if pos >= 0:
                                    self.current_frame_num = pos
                                else:
                                    self.current_frame_num += 1
                        except Exception:
                            with QMutexLocker(self.frame_mutex):
                                self.current_frame_num += 1
                    except Exception as e:
                        self.status_update.emit(f"读取帧错误: {str(e)}")
                        break

                # 更新当前帧数据
                if frame_read_ok and frame is not None:
                    with QMutexLocker(self.frame_mutex):
                        self.current_frame = frame.copy()

                    # 发送进度更新
                    with QMutexLocker(self.frame_mutex):
                        current_time = self.current_frame_num / self.fps if self.fps > 0 else 0.0
                    self.progress_updated.emit(self.current_frame_num, self.total_frames, current_time)

                    # 转换并发送帧
                    try:
                        if not frame.flags['C_CONTIGUOUS']:
                            frame = np.ascontiguousarray(frame)
                        height, width, channel = frame.shape
                        bytes_per_line = 3 * width

                        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888).copy()
                        self.frame_ready.emit(q_img)
                    except Exception:
                        traceback.print_exc()

                # 控制帧率
                elapsed = time.time() - loop_start
                wait_time = max(0.0, frame_interval - elapsed) * 1000  # 转换为毫秒
                
                # 短暂睡眠，使用更简单的方式避免复杂的计时器操作
                if wait_time > 0:
                    QThread.msleep(int(wait_time))
                
                # 再次检查暂停状态
                with QMutexLocker(self.frame_mutex):
                    if self.paused:
                        self.wait_condition.wait(self.frame_mutex)
                        if not self.playing:
                            break

        except Exception:
            self.status_update.emit(f"视频播放错误")
            traceback.print_exc()
    
    def _camera_playback(self): 
        """摄像头播放实现"""
        try:
            import cv2
            import numpy as np
            import time
            import traceback
            from PySide6.QtCore import QThread, QMutexLocker
            
            # 初始化摄像头捕获对象
            with QMutexLocker(self.cap_mutex):
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    self.status_update.emit(f"无法打开摄像头: {self.camera_id}")
                    return
                
                # ✅ 优化摄像头参数设置
                # 使用配置中的分辨率以提高帧率
                width, height = AppConfig.CAMERA_SETTINGS['resolution']
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)   
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # 设置MJPG编码（如果有的话）
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                except:
                    pass
                
                # 减少缓冲区
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, AppConfig.CAMERA_SETTINGS['buffer_size'])
                except:
                    pass
                
                # 设置帧率（如果摄像头支持）
                try:
                    self.cap.set(cv2.CAP_PROP_FPS, AppConfig.CAMERA_SETTINGS['target_fps'])
                except:
                    pass
            
            # 预热摄像头
            for _ in range(AppConfig.CAMERA_SETTINGS['warmup_frames']):
                try:
                    ret_warm, _ = self.cap.read()
                    if not ret_warm:
                        break
                except:
                    break
            
            # 使用配置中的分辨率显示状态信息
            width, height = AppConfig.CAMERA_SETTINGS['resolution']
            self.status_update.emit(f"开始摄像头实时显示 ({width}x{height})")
            
            frame_count = 0
            start_time = time.time()
            
            while self.playing:
                # 检查暂停状态
                with QMutexLocker(self.frame_mutex):
                    if self.paused:
                        self.wait_condition.wait(self.frame_mutex)
                        if not self.playing:
                            break
                        continue
                
                # ✅ 更简单的帧读取逻辑
                frame = None
                frame_read_ok = False
                
                with QMutexLocker(self.cap_mutex):
                    if not self.cap or not self.cap.isOpened():
                        break
                    
                    try:
                        ret, frame = self.cap.read()
                        frame_read_ok = ret and frame is not None
                    except Exception:
                        frame_read_ok = False
                
                if not frame_read_ok or frame is None:
                    self.status_update.emit("无法读取摄像头画面，重试...")
                    QThread.msleep(10)
                    continue
                
                # ✅ 限制FPS，避免过快处理
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    self.status_update.emit(f"摄像头FPS: {fps:.1f}")
                    frame_count = 0
                    start_time = time.time()
                
                # 更新当前帧
                with QMutexLocker(self.frame_mutex):
                    self.current_frame = frame.copy()
                
                # 转换并发送帧
                try:
                    # ✅ 确保内存连续
                    if not frame.flags['C_CONTIGUOUS']:
                        frame = np.ascontiguousarray(frame)
                    
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width
                    
                    # ✅ 创建QImage，立即复制数据
                    q_img = QImage(
                        frame.data, width, height, bytes_per_line,
                        QImage.Format_BGR888
                    ).copy()  # 重要：复制数据以避免引用问题
                    
                    self.frame_ready.emit(q_img)
                except Exception:
                    traceback.print_exc()
                
                # ✅ 简单的延时控制，目标15-20FPS
                QThread.msleep(50)  # 50ms = 20FPS
                
        except Exception:
            self.status_update.emit(f"摄像头播放错误")
            traceback.print_exc()
        finally:
            with QMutexLocker(self.cap_mutex):
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None


class YOLOMainController(QObject):
    """主逻辑协调控制器"""
    
    def __init__(self, ui_window: YOLOMainWindowUI):
        super().__init__()
        self.ui = ui_window
        
        # 核心组件
        self.video_player = VideoPlayerThread()  # 使用基于QThread的播放器
        # 延迟导入FrameGrabberThread以避免循环导入问题
        try:
            from frame_grabber import FrameGrabberThread
            self.frame_grabber = FrameGrabberThread(self.video_player)  # 使用新的QThread版本抓取器
            print("FrameGrabberThread导入成功")
        except ImportError as e:
            print(f"导入FrameGrabberThread失败: {e}")
            self.frame_grabber = None
        self.yolo_processor = None                   # YOLO推理引擎
        
        # 状态变量
        self.model_loaded = False
        self.model_path = None
        self.model_mode = None  # 'detect', 'classify', 'pose'
        
        # 处理状态
        self.is_processing = False      # 是否正在YOLO处理
        self.is_playing = False         # 是否正在播放
        self.current_file = None
        self.current_mode = None        # 'image', 'video', 'camera'
        
        # 默认参数 - 从配置文件读取
        self.default_params = {
            'iou_threshold': AppConfig.YOLO_SETTINGS['default_iou'],
            'confidence_threshold': AppConfig.YOLO_SETTINGS['default_confidence'],
            'delay_ms': 10,  # 这个值可以考虑添加到配置中
            'line_width': AppConfig.YOLO_SETTINGS['default_line_width']
        }
        
        # 获取UI组件引用
        self.left_panel = self.ui.get_left_panel()
        self.right_panel = self.ui.get_right_panel()
        
        # 初始化UI状态
        self._init_ui_state()
        # 连接信号
        self._connect_signals()
        
        print("YOLO逻辑控制器初始化完成")
    

    
    def _init_ui_state(self):
        """初始化UI状态"""
        # 设置默认参数
        self.right_panel.set_parameters(**self.default_params)
        
        # 显示初始状态
        self.left_panel.clear_display()
        self.right_panel.update_model_info()
        
        # 设置控制按钮状态
        self.right_panel.set_control_state(False)
    
    def _connect_signals(self):
        """连接UI暴露的信号，将错误处理放在控制器中"""
        # ===== 连接视频播放器信号 =====
        self.video_player.frame_ready.connect(self._on_player_frame)
        self.video_player.status_update.connect(self._on_status_update)
        self.video_player.progress_updated.connect(self._on_progress_updated)
        self.video_player.finished.connect(self._on_player_finished)
        
        # ===== 连接抓取器信号 =====
        self.frame_grabber.frame_processed.connect(self._on_frame_processed)
        self.frame_grabber.processing_complete.connect(self._on_processing_complete)
        self.frame_grabber.status_update.connect(self._on_status_update)
        self.frame_grabber.error_occurred.connect(self._on_grabber_error)
        self.frame_grabber.finished.connect(self._on_grabber_finished)
        
        # ===== 文件菜单信号 =====
        self.ui.file_menu_init.connect(self._on_file_init)
        self.ui.file_menu_exit.connect(self._on_file_exit)
        
        # ===== 帮助菜单信号 =====
        self.ui.help_menu_about.connect(self._on_help_about)
        self.ui.help_menu_manual.connect(self._on_help_manual)
        
        # ===== 主要功能信号 =====
        self.ui.model_load.connect(self.handle_load_model)
        self.ui.image_open.connect(self.handle_open_image)
        self.ui.video_open.connect(self.handle_open_video)
        self.ui.camera_open.connect(self.handle_open_camera)
        
        # ===== 控制按钮信号 =====
        self.right_panel.start_inference.connect(self.handle_start_inference)
        self.right_panel.stop_inference.connect(self.handle_stop_inference)
        self.right_panel.save_screenshot.connect(self.handle_save_screenshot)
        
        # ===== 左侧面板播放/暂停信号 =====
        self.ui.left_panel_play_pause.connect(self.handle_play_pause)
        
        # ===== 左侧面板进度条信号 =====
        self.left_panel.progress_changed.connect(self.handle_progress_change)
        
        # ===== 参数信号 =====
        self.right_panel.iou_changed.connect(self.handle_iou_change)
        self.right_panel.confidence_changed.connect(self.handle_confidence_change)
        self.right_panel.delay_changed.connect(self.handle_delay_change)
        self.right_panel.line_width_changed.connect(self.handle_line_width_change)
    
    # ============================================================================
    # 信号处理方法
    # ============================================================================
    
    def _on_player_frame(self, q_image: QImage):
        """接收到播放器的原始帧 - 直接显示（无YOLO处理时）"""
        try:
            if not self.is_processing:
                pixmap = QPixmap.fromImage(q_image)
                self.left_panel.set_display_image(pixmap)
        except Exception as e:
            print(f"显示原始帧失败: {e}")
    
    def _on_frame_processed(self, q_image: QImage):
        """接收到处理后的帧 - 显示YOLO检测结果"""
        try:
            pixmap = QPixmap.fromImage(q_image)
            self.left_panel.set_display_image(pixmap)
        except Exception as e:
            print(f"显示处理帧失败: {e}")
    
    def _on_player_finished(self):
        """播放器完成"""
        self.is_playing = False
        self.left_panel.set_play_state(False)
        print("播放器停止")
    
    def _on_processing_complete(self, stats: dict):
        """处理完成 - 更新统计信息"""
        try:
            self.right_panel.update_statistics(
                detection_count=stats.get('detection_count', 0),
                confidence=stats.get('avg_confidence', 0.0),
                inference_time=stats.get('inference_time', 0),
                fps=stats.get('fps', 0.0)
            )
        except Exception as e:
            print(f"更新统计信息失败: {e}")
    
    def _on_grabber_error(self, error_msg: str):
        """抓取器错误"""
        print(f"抓取器错误: {error_msg}")
    
    def _on_grabber_finished(self):
        """抓取器完成"""
        self.is_processing = False
        self.right_panel.set_control_state(False)
        print("帧抓取停止")
    
    def _on_status_update(self, status: str):
        """状态更新"""
        print(f"状态: {status}")
    
    def _on_progress_updated(self, current_frame, total_frames, current_time):
        """视频进度更新"""
        try:
            if self.current_mode == 'video':
                self.left_panel.set_progress_range(0, 1000)
                
                if total_frames > 0:
                    progress_value = int((current_frame / total_frames) * 1000)
                    self.left_panel.set_progress_value(progress_value)
                
                current_time_str = self._format_time(current_time)
                total_time_str = self._format_time(total_frames / self.video_player.fps) if self.video_player.fps > 0 else "--:--"
                self.left_panel.set_time_display(current_time_str, total_time_str)
                
        except Exception as e:
            print(f"更新进度失败: {e}")
    
    def handle_progress_change(self, value):
        """用户拖动进度条"""
        if self.current_mode == 'video' and hasattr(self.video_player, 'cap') and self.video_player.cap:
            try:
                total_frames = self.video_player.total_frames
                if total_frames > 0:
                    target_frame = int((value / 1000.0) * total_frames)
                    self.video_player.seek_frame(target_frame)
                    
                    print(f"跳转到进度: {value}/1000, 帧号: {target_frame}/{total_frames}")
            except Exception as e:
                print(f"跳转进度失败: {e}")
    
    def handle_play_pause(self):
        """播放/暂停按钮点击"""
        try:
            if self.current_mode == 'video':
                if self.video_player.playing:
                    self.video_player.pause()
                    self.left_panel.set_play_state(False)
                else:
                    self.video_player.resume()
                    self.left_panel.set_play_state(True)
            elif self.current_mode == 'camera':
                if self.video_player.playing:
                    self.video_player.pause()
                    self.left_panel.set_play_state(False)
                else:
                    self.video_player.resume()
                    self.left_panel.set_play_state(True)
        except Exception as e:
            print(f"播放/暂停失败: {e}")
    
    # ============================================================================
    # 文件菜单处理方法
    # ============================================================================
    
    def _on_file_init(self):
        """初始化"""
        reply = QMessageBox.question(
            self.ui, "确认初始化",
            "是否要初始化所有设置？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._stop_all()
            self._init_ui_state()
            QMessageBox.information(self.ui, "初始化完成", "所有设置已重置")
    
    def _on_file_exit(self):
        """退出"""
        reply = QMessageBox.question(
            self.ui, "确认退出",
            "是否要退出YOLO检测系统？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._stop_all()
            self.ui.close()
    
    # ============================================================================
    # 帮助菜单处理方法
    # ============================================================================
    
    def _on_help_about(self):
        """显示关于对话框"""
        about_text = f"""
        <h3>YOLO多功能检测系统</h3>
        
        <b>版本:</b> 1.0.0<br>
        <b>作者:</b> Sephoration<br><br>
        
        <b>功能特点:</b><br>
        • 目标检测与跟踪<br>
        • 关键点/姿态检测<br>
        • 图像分类<br>
        • 支持图片、视频、摄像头<br>
        • 实时统计与可视化<br><br>
        
        <b>技术支持:</b><br>
        • PySide6 (Qt for Python)<br>
        • Ultralytics YOLO<br>
        • OpenCV<br><br>
        
        <b>© 2024 版权所有</b>
        """
        
        QMessageBox.about(self.ui, "关于", about_text)
    
    def _on_help_manual(self):
        """显示使用说明"""
        manual_text = """
        <h3>YOLO多功能检测系统 - 使用说明</h3>
        
        <b>1. 加载模型</b><br>
        • 点击"打开模型"按钮选择YOLO模型文件 (.pt)<br>
        • 选择对应的模块类型：分析器(目标检测)、分类器、关键点检测<br><br>
        
        <b>2. 打开媒体文件</b><br>
        • <b>图片</b>: 点击"打开图片"，选择图片文件<br>
        • <b>视频</b>: 点击"打开视频"，选择视频文件<br>
        • <b>摄像头</b>: 点击"打开摄像头"，使用默认摄像头<br><br>
        
        <b>3. 参数设置</b><br>
        • <b>IOU阈值</b>: 控制检测框重叠度 (0.0-1.0)<br>
        • <b>置信度</b>: 过滤低置信度检测结果 (0.0-1.0)<br>
        • <b>延迟(ms)</b>: 控制处理间隔，影响实时性<br>
        • <b>线宽</b>: 调整检测框和关键点的绘制线宽<br><br>
        
        <b>4. 开始检测</b><br>
        • 点击"开始"按钮开始推理处理<br>
        • 实时统计面板显示处理结果<br>
        • 点击"停止"按钮结束处理<br><br>
        
        <b>5. 视频控制</b><br>
        • <b>播放/暂停</b>: 控制视频播放<br>
        • <b>进度条</b>: 拖动跳转到指定位置<br>
        • <b>时间显示</b>: 显示当前/总时长<br><br>
        
        <b>6. 其他功能</b><br>
        • <b>保存截图</b>: 保存当前显示画面<br>
        • <b>初始化</b>: 重置所有设置<br>
        • <b>退出</b>: 关闭应用程序<br><br>
        
        <b>提示:</b><br>
        • 确保已安装必要的Python库<br>
        • 使用合适的YOLO模型文件<br>
        • 调整参数以获得最佳检测效果
        """
        
        QMessageBox.information(self.ui, "使用说明", manual_text)
    
    # ============================================================================
    # 主要功能处理方法
    # ============================================================================
    
    def handle_load_model(self):
        """加载模型"""
        try:
            # 使用配置文件中的文件过滤器
            model_filter = AppConfig.FILE_SETTINGS['file_filters']['model']
            model_path, _ = QFileDialog.getOpenFileName(
                self.ui, "选择YOLO模型文件",
                "", model_filter
            )
            
            if model_path:
                # 清空之前的模型信息
                self.model_path = None
                self.model_mode = None
                self.yolo_processor = None
                self.model_loaded = False
                
                print(f"开始分析模型: {model_path}")
                
                try:
                    # 使用UnifiedYOLO分析模型（不加载完整模型）
                    from yolo_analyzer import UnifiedYOLO
                    
                    # 分析模型信息（轻量级分析，不真正加载模型）
                    model_info = UnifiedYOLO.analyze_model_info(model_path)
                    
                    if model_info:
                        # 获取模型信息
                        model_name = os.path.basename(model_path)
                        task_type = model_info.get('task_type', 'detection')
                        input_size = model_info.get('input_size', '640x640')
                        class_count = model_info.get('class_count', '未知')
                        
                        # 任务类型到显示名称映射
                        # 使用配置文件中的任务显示映射
                        task_display_map = AppConfig.TASK_CONFIG['task_display_map']
                        
                        display_name = task_display_map.get(task_type, task_type)
                        self.model_mode = task_type
                        
                        # 更新UI显示模型信息
                        self.right_panel.update_model_info(
                            model_path=model_path,
                            task_type=display_name,
                            input_size=input_size,
                            class_count=class_count
                        )
                        
                        self.model_path = model_path
                        
                        # 显示成功信息
                        QMessageBox.information(
                            self.ui, "模型分析成功",
                            f"✅ 已自动识别模型类型\n\n"
                            f"📦 模型名称: {model_name}\n"
                            f"🎯 任务类型: {display_name}\n"
                            f"📏 输入尺寸: {input_size}\n"
                            f"🔢 类别数量: {class_count}\n\n"
                            f"模型将在点击'开始'时正式加载。"
                        )
                        
                        print(f"模型分析完成，类型: {task_type}")
                    else:
                        raise Exception("无法识别模型类型")
                    
                except Exception as e:
                    print(f"模型分析失败: {e}")
                    # 分析失败，让用户选择模型类型
                    self._show_model_type_dialog(model_path)
                    
        except Exception as e:
            self._show_error("选择模型失败", str(e))
    
    def _show_model_type_dialog(self, model_path):
        """显示模型类型选择对话框"""
        try:
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel
            from PySide6.QtCore import Qt
            
            dialog = QDialog(self.ui)
            dialog.setWindowTitle("选择模型类型")
            dialog.setFixedSize(300, 220)
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)
            
            model_name = os.path.basename(model_path)
            info_label = QLabel(f"已选择模型:\n{model_name}")
            info_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(info_label)
            
            tip_label = QLabel("请选择处理模式:")
            tip_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(tip_label)
            
            btn_detect = QPushButton("目标检测")
            btn_classify = QPushButton("图像分类")
            btn_pose = QPushButton("关键点检测")
            
            button_style = """
                QPushButton {
                    background-color: #f0f0f0;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 10px;
                    font-weight: normal;
                    min-height: 40px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                    border-color: #aaaaaa;
                }
            """
            
            for btn in [btn_detect, btn_classify, btn_pose]:
                btn.setStyleSheet(button_style)
            
            btn_detect.clicked.connect(lambda: self._select_model_type('detection', model_path, dialog))
            btn_classify.clicked.connect(lambda: self._select_model_type('classification', model_path, dialog))
            btn_pose.clicked.connect(lambda: self._select_model_type('pose', model_path, dialog))
            
            layout.addWidget(btn_detect)
            layout.addWidget(btn_classify)
            layout.addWidget(btn_pose)
            
            dialog.exec()
            
        except Exception as e:
            self._show_error("选择模型类型失败", str(e))
    
    def _select_model_type(self, model_type: str, model_path: str, dialog):
        """选择模型类型"""
        try:
            self.model_mode = model_type
            self.model_path = model_path
            
            # 任务类型到显示名称映射
            # 使用配置文件中的任务显示映射
            task_display_map = AppConfig.TASK_CONFIG['task_display_map']
            
            display_name = task_display_map.get(model_type, model_type)
            
            # 更新UI显示模型信息
            self.right_panel.update_model_info(
                model_path=model_path,
                task_type=display_name,
                input_size="640x640",  # 默认尺寸
                class_count="待检测"
            )
            
            # 显示成功信息
            QMessageBox.information(
                self.ui, "模型选择成功",
                f"✅ 已选择{display_name}模式\n\n"
                f"📦 模型: {os.path.basename(model_path)}\n"
                f"🎯 任务: {display_name}\n\n"
                f"模型将在点击'开始'时正式加载。"
            )
            
            print(f"已选择{display_name}模式，模型将在点击'开始'时加载")
            
            dialog.close()
            
        except Exception as e:
            self._show_error("选择模块失败", str(e))
    
    def handle_open_image(self):
        """打开图片"""
        try:
            self._stop_all()
            
            # 使用配置文件中的文件过滤器
            image_filter = AppConfig.FILE_SETTINGS['file_filters']['image']
            image_path, _ = QFileDialog.getOpenFileName(
                self.ui, "选择图片文件",
                "", image_filter
            )
            
            if image_path:
                self.current_file = image_path
                self.current_mode = 'image'
                
                self.left_panel.update_info(os.path.basename(image_path), 'image')
                
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    self.left_panel.set_display_image(pixmap)
                    print(f"已加载图片: {os.path.basename(image_path)}")
                else:
                    QMessageBox.warning(self.ui, "警告", "无法加载图片文件")
                
        except Exception as e:
            self._show_error("打开图片失败", str(e))
    
    def handle_open_video(self):
        """打开视频"""
        try:
            self._stop_all()

            # 使用配置文件中的文件过滤器
            video_filter = AppConfig.FILE_SETTINGS['file_filters']['video']
            video_path, _ = QFileDialog.getOpenFileName(
                self.ui, "选择视频文件",
                "", video_filter
            )

            if video_path:
                self.current_file = video_path
                self.current_mode = 'video'
                self.is_playing = True

                self.left_panel.update_info(os.path.basename(video_path), 'video')

                # 使用VideoPlayerThread (QThread版本)
                self.video_player.play_video(video_path)
                self.left_panel.set_play_state(True)  # 设置播放状态

            print(f"开始播放视频: {os.path.basename(video_path)}")
                
        except Exception as e:
            self._show_error("打开视频失败", str(e))

    def handle_open_camera(self):
        """打开摄像头"""
        try:
            self._stop_all()

            camera_id = 0  # 默认摄像头

            self.current_file = f"摄像头 {camera_id}"
            self.current_mode = 'camera'
            self.is_playing = True

            self.left_panel.update_info(f"摄像头 {camera_id}", 'camera')

            # 使用VideoPlayerThread (QThread版本)
            self.video_player.play_camera(camera_id)
            self.left_panel.set_play_state(True)  # 设置播放状态

            print(f"开始摄像头实时显示")
                
        except Exception as e:
            self._show_error("打开摄像头失败", str(e))
        
    # ============================================================================
    # 控制按钮处理方法
    # ============================================================================
    
    def handle_start_inference(self):
        """开始推理"""
        try:
            # 检查必要条件
            if not self.current_file:
                QMessageBox.warning(self.ui, "警告", "请先选择媒体文件！")
                return
            
            if not self.model_path or not self.model_mode:
                QMessageBox.warning(self.ui, "警告", "请先选择模型！")
                return
            
            # 加载模型（此时才真正加载）
            if not self._load_yolo_processor():
                return
            
            # 检查当前模式
            if self.current_mode == 'image':
                self._process_image()
            elif self.current_mode in ['video', 'camera']:
                self._process_video_camera()
            else:
                QMessageBox.warning(self.ui, "警告", "请先选择媒体文件！")
                
        except Exception as e:
            self._show_error("开始处理失败", str(e))
    
    def _process_image(self):
        """处理图片"""
        try:
            if not self._load_yolo_processor():
                return
            
            print(f"开始处理图片: {self.current_file}")
            
            # 加载图片
            import cv2
            from PySide6.QtGui import QImage, QPixmap
            
            image = cv2.imread(self.current_file)
            if image is None:
                QMessageBox.warning(self.ui, "警告", "无法读取图片文件")
                return
            
            # 调用YOLO处理器处理图片
            result_dict = self.yolo_processor.process_frame(image)
            
            # 提取处理后的图像和统计信息
            if isinstance(result_dict, dict):
                processed_image = result_dict.get('image', image)
                stats_data = result_dict.get('stats', {})
            else:
                processed_image = image
                stats_data = {}
                print("警告: YOLO处理器返回的不是字典格式")
            
            # 显示处理后的图像
            if isinstance(processed_image, type(image)):  # 检查是否是numpy数组
                # 确保图像是RGB格式
                if len(processed_image.shape) == 2:  # 灰度图
                    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                elif processed_image.shape[2] == 4:  # RGBA
                    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2RGB)
                else:  # BGR
                    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                height, width, channel = processed_rgb.shape
                bytes_per_line = 3 * width
                
                q_img = QImage(processed_rgb.data, width, height, 
                              bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.left_panel.set_display_image(pixmap)
            else:
                print(f"警告: 处理后的图像不是numpy数组，类型: {type(processed_image)}")
            
            # 更新统计信息
            self.right_panel.update_statistics(
                detection_count=stats_data.get('detection_count', 0),
                confidence=stats_data.get('avg_confidence', 0.0),
                inference_time=stats_data.get('inference_time', 0),
                fps=stats_data.get('fps', 0.0)
            )
            
            # 更新UI状态
            self.right_panel.set_control_state(True)
            self.is_processing = True
            
            print(f"图片处理完成: {self.current_file}")
            
        except Exception as e:
            self._show_error("图片处理失败", str(e))
    
    def _process_video_camera(self):
        """处理视频/摄像头"""
        try:
            if not self._load_yolo_processor():
                return
            
            # 设置YOLO处理器到抓取器
            self.frame_grabber.set_yolo_processor(self.yolo_processor)
            
            # 获取抓取间隔参数
            delay_ms = self.right_panel.get_parameters().get('delay_ms', 10)
            grab_interval = max(1, delay_ms // 10)  # 根据延迟计算间隔
            
            # 开始抓取帧
            self.frame_grabber.start_grabbing(grab_interval)
            
            # 更新UI状态
            self.right_panel.set_control_state(True)
            self.is_processing = True
            
            print(f"开始处理{self.current_mode}: {self.current_file}")
            print(f"抓取间隔: 每{grab_interval}帧抓取一次")
            
        except Exception as e:
            self._show_error("开始处理失败", str(e))
    
    def _load_yolo_processor(self) -> bool:
        """加载YOLO处理器（在点击"开始"时调用）"""
        try:
            if not self.model_path or not self.model_mode:
                QMessageBox.warning(self.ui, "警告", "请先选择模型！")
                return False
            
            if self.yolo_processor is not None:
                print("YOLO处理器已加载，跳过重复加载")
                return True
            
            # 获取参数
            params = self.right_panel.get_parameters()
            
            print(f"正在正式加载YOLO模型: {self.model_path}")
            print(f"模式: {self.model_mode}")
            print(f"参数: IOU={params['iou_threshold']}, 置信度={params['confidence_threshold']}")
            
            # 创建YOLO处理器实例
            from yolo_analyzer import UnifiedYOLO
            self.yolo_processor = UnifiedYOLO(
                model_path=self.model_path,
                mode=self.model_mode,
                conf_threshold=params['confidence_threshold'],
                iou_threshold=params['iou_threshold']
            )
            
            self.model_loaded = True
            
            # 获取模型信息
            model_info = self.yolo_processor.get_model_info()
            
            # 更新UI显示详细模型信息
            # 使用配置文件中的任务显示映射
            task_display_map = AppConfig.TASK_CONFIG['task_display_map']
            
            display_name = task_display_map.get(self.model_mode, self.model_mode)
            input_size_str = f"{model_info.get('input_size', 640)}"
            # 同时尝试class_count和num_classes键，确保能获取到类别数量
            class_count = model_info.get('class_count', model_info.get('num_classes', '未知'))
            
            self.right_panel.update_model_info(
                model_path=self.model_path,
                task_type=display_name,
                input_size=input_size_str,
                class_count=str(class_count)
            )
            
            print(f"✅ YOLO处理器加载成功: {self.model_mode}")
            print(f"   - 输入尺寸: {input_size_str}")
            print(f"   - 类别数量: {class_count}")
            return True
                
        except Exception as e:
            self._show_error("加载YOLO处理器失败", str(e))
            self.model_loaded = False
            return False
    
    def handle_stop_inference(self):
        """停止推理"""
        self._stop_processing()
    
    def handle_save_screenshot(self):
        """保存截图"""
        try:
            pixmap = self.left_panel.display_label.pixmap()
            if pixmap and not pixmap.isNull():
                # 使用配置文件中的文件过滤器
                file_filter = AppConfig.FILE_SETTINGS['file_filters']['screenshot']
                
                if self.current_file:
                    base_name = os.path.splitext(os.path.basename(self.current_file))[0]
                else:
                    base_name = "screenshot"
                
                default_name = f"{base_name}.png"
                
                save_path, _ = QFileDialog.getSaveFileName(
                    self.ui, "保存截图",
                    default_name,
                    file_filter
                )
                
                if save_path:
                    if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        save_path += '.png'
                    
                    success = pixmap.save(save_path)
                    if success:
                        QMessageBox.information(self.ui, "保存成功", f"截图已保存到:\n{save_path}")
                        print(f"截图保存到: {save_path}")
                    else:
                        QMessageBox.warning(self.ui, "保存失败", "无法保存截图")
            else:
                QMessageBox.warning(self.ui, "警告", "没有可保存的图像")
                
        except Exception as e:
            self._show_error("保存截图失败", str(e))
    
    # ============================================================================
    # 辅助方法
    # ============================================================================
    
    def _stop_all(self):
        """停止所有处理"""
        self._stop_processing()
        self.video_player.stop()
        self.is_playing = False
        self.left_panel.set_controls_enabled(False)
    
    def _stop_processing(self):
        """停止处理"""
        self.frame_grabber.stop_grabbing()
        self.is_processing = False
        self.right_panel.set_control_state(False)
        print("处理已停止")
    
    def _format_time(self, seconds):
        """格式化时间为 MM:SS 格式"""
        try:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        except:
            return "--:--"
    
    def _show_error(self, title: str, message: str):
        """显示错误"""
        QMessageBox.critical(
            self.ui, title,
            f"{message}\n\n详细信息请查看控制台输出。"
        )
        print(f"错误 [{title}]: {message}")
        traceback.print_exc()
    
    def handle_iou_change(self, value):
        """处理IOU阈值变化"""
        try:
            self.default_params['iou_threshold'] = value
            print(f"IOU阈值已更新: {value}")
            # 如果模型已加载，更新模型参数
            if hasattr(self, 'yolo_processor') and self.yolo_processor is not None:
                success = self.yolo_processor.update_params(iou_threshold=value)
                if success:
                    print("IOU参数更新成功")
                else:
                    print("IOU参数更新失败")
        except Exception as e:
            self._show_error("更新IOU阈值失败", str(e))
    
    def handle_confidence_change(self, value):
        """处理置信度阈值变化"""
        try:
            self.default_params['confidence_threshold'] = value
            print(f"置信度阈值已更新: {value}")
            # 如果模型已加载，更新模型参数
            if hasattr(self, 'yolo_processor') and self.yolo_processor is not None:
                success = self.yolo_processor.update_params(conf_threshold=value)
                if success:
                    print("置信度参数更新成功")
                else:
                    print("置信度参数更新失败")
        except Exception as e:
            self._show_error("更新置信度阈值失败", str(e))
    
    def handle_delay_change(self, value):
        """处理延迟时间变化"""
        try:
            self.default_params['delay_ms'] = value
            print(f"延迟时间已更新: {value}ms")
        except Exception as e:
            self._show_error("更新延迟时间失败", str(e))
    
    def handle_line_width_change(self, value):
        """处理线宽变化"""
        try:
            self.default_params['line_width'] = value
            print(f"线宽已更新: {value}")
            # 如果模型已加载，更新模型参数
            if self.yolo_processor:
                self.yolo_processor.update_params(line_width=value)
        except Exception as e:
            self._show_error("更新线宽失败", str(e))