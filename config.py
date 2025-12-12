# config.py 或 settings.py
"""
系统配置集中管理
所有硬编码参数都在这里统一管理
"""

class AppConfig:
    """应用程序配置"""
    
    # ==================== 应用信息 ====================
    APP_INFO = {
        'version': '1.0.0',
        'author': 'Sephoration',
        'copyright': '© 2024 版权所有',
    }
    
    # ==================== 视频/摄像头设置 ====================
    CAMERA_SETTINGS = {
        'default_camera_id': 0,
        'resolution': (640, 480),  # 改为640x480，更合理
        'target_fps': 30,
        'buffer_size': 1,
        'warmup_frames': 5,  # 摄像头预热帧数
        'retry_delay': 10,  # 重试延迟(毫秒)
        'msleep_duration': 50,  # 睡眠时长(毫秒)
    }
    
    VIDEO_SETTINGS = {
        'default_buffer_size': 1,
        'min_fps': 10,
        'max_fps': 60,
        'seek_accuracy': True,  # 是否精确跳帧
    }
    
    # ==================== 播放器设置 ====================
    PLAYER_SETTINGS = {
        'frame_interval_ms': 33,  # 约30FPS
        'use_grab_method': True,  # 是否使用grab+retrieve
        'pause_check_interval': 50,  # 暂停检查间隔(ms)
        'progress_update_threshold': 10,  # 进度更新阈值(帧)
    }
    
    # ==================== 线程设置 ====================
    THREAD_SETTINGS = {
        'player_priority': 'NormalPriority',
        'grabber_priority': 'LowPriority',
        'timeout_ms': 3000,  # 线程等待超时
        'use_opencv_threads': False,  # 是否启用OpenCV多线程
        'opencv_num_threads': 0,  # OpenCV线程数
    }
    
    # ==================== YOLO处理设置 ====================
    YOLO_SETTINGS = {
        'default_confidence': 0.5,
        'default_iou': 0.45,
        'default_line_width': 2,
        'inference_batch_size': 1,  # 批处理大小
        'warmup_iterations': 10,    # 模型预热次数
    }
    
    # ==================== UI设置 ====================
    UI_SETTINGS = {
        'display_ratio': (16, 9),      # 显示区域比例
        'panel_ratio': (4, 3),         # 面板比例
        'progress_range': 1000,        # 进度条范围
        'time_format': 'MM:SS',        # 时间格式
        'status_update_delay': 1000,   # 状态更新延迟(ms)
    }
    
    # ==================== 性能优化设置 ====================
    PERFORMANCE_SETTINGS = {
        'enable_frame_buffer': True,     # 启用帧缓冲
        'buffer_size': 3,                # 帧缓冲大小
        'skip_duplicate_frames': True,   # 跳过重复帧
        'downsample_large_frames': True, # 下采样大帧
        'max_frame_size': (1920, 1080),  # 最大帧尺寸
    }
    
    # ==================== 文件设置 ====================
    FILE_SETTINGS = {
        'supported_images': ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif'],
        'supported_videos': ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv'],
        'supported_models': ['*.pt', '*.pth', '*.onnx'],
        'default_save_format': 'png',
        'max_recent_files': 10,
        # 文件过滤器字符串
        'file_filters': {
            'model': "模型文件 (*.pt *.pth *.onnx);;所有文件 (*.*)",
            'image': "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*.*)",
            'video': "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)",
            'screenshot': "PNG图片 (*.png);;JPEG图片 (*.jpg *.jpeg);;所有文件 (*.*)"
        }
    }
    
    # ==================== 任务类型配置 ====================
    TASK_CONFIG = {
        'task_display_map': {
            'detection': '目标检测',
            'classification': '图像分类',
            'pose': '关键点检测',
            'segmentation': '分割检测'
        },
        'default_input_size': '640x640',  # 默认输入尺寸
    }
    
    # ==================== 图像格式设置 ====================
    IMAGE_PROCESSING_CONFIG = {
        'default_format': 'BGR888',  # 默认图像格式
    }
    
    # ==================== 调试设置 ====================
    DEBUG_SETTINGS = {
        'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
        'log_to_file': False,
        'log_file': 'app.log',
        'profile_performance': False,  # 性能分析
        'show_fps': True,              # 显示FPS
    }



    # format_spec.py
    UNIFIED_FORMAT_SPEC = {
        "required_fields": ["success", "mode", "image", "objects", "stats"],
        "mode_values": ["detection", "classification", "pose"],
        "object_types": ["bbox", "classification", "keypoint"]
}