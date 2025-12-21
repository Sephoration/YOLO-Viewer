"""
YOLO统一分析器
整合三个模式：检测、分类、关键点
直接调用模型对象，禁止使用.predict()方法
处理好的数据直接叠加在展示窗口上
数据格式统一化
加入了预热功能
加入了类别过滤
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Tuple
from ultralytics import YOLO

from baseDetect import baseDetect

# YOLO统一分析器实现 - 将预热功能集成到下面的UnifiedYOLO类中


class UnifiedYOLO(baseDetect):
    """
    统一YOLO处理器 - 整合三个模式
    遵循老师要求的代码风格：直接调用模型对象
    """
    
    def __init__(self, model_path: str, mode: str = 'auto',
                 conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                 warmup: bool = True, config_path: str = None):
        """
        初始化统一YOLO处理器
        
        Args:
            model_path: 模型文件路径
            mode: 模式 ('auto', 'detection', 'classification', 'pose')
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            warmup: 是否在加载模型时执行预热
            config_path: 配置文件路径，默认使用与模型同名的.json文件
        """
        super().__init__()
        
        self.model_path = model_path
        self.mode = self._detect_mode(model_path) if mode == 'auto' else mode
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.warmup = warmup  # 预热开关
        self.config_path = config_path  # 配置文件路径
        
        # 设备选择
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 模型对象（延迟加载）
        self.model = None
        self.model_info = {}
        self.warmed_up = False  # 预热状态标志
        
        # 可视化配置
        self._setup_visualization_params()
        
        # 加载配置文件
        self._load_config()
        
        # 模式特定参数
        self._setup_mode_params()
        
        # 可视化控制参数
        self.show_kpt_names = True  # 是否显示关键点名称
        self.show_skeleton = True   # 是否显示骨架连接
        self.show_bbox = True       # 是否显示边界框
        
        print(f"YOLO统一处理器初始化 | 模式: {self.mode} | 设备: {self.device}")
    
    def _detect_mode(self, model_path: str) -> str:
        """
        自动检测模型类型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            str: 模型模式 ('detection', 'classification', 'pose', 'segmentation')
        """
        filename = Path(model_path).name.lower()
        
        # 根据文件名判断
        if 'cls' in filename or 'classify' in filename:
            return 'classification'
        elif 'pose' in filename or 'keypoint' in filename:
            return 'pose'
        elif 'seg' in filename:
            return 'segmentation'
        elif 'det' in filename or 'obj' in filename:
            return 'detection'
        else:
            # 默认检测模式
            return 'detection'
    
    def _setup_mode_params(self):
        """根据模式设置参数"""
        if self.mode == 'pose':
            self.conf = 0.3
            self.iou = 0.6
            self.img_size = 640
        elif self.mode == 'classification':
            self.conf = 0.25
            self.iou = 0.45
            self.img_size = 224  # 分类模型通常使用224
        elif self.mode == 'segmentation':
            self.conf = 0.25
            self.iou = 0.7
            self.img_size = 640
        else:  # detection
            self.conf = 0.25
            self.iou = 0.7
            self.img_size = 640
        
        # 覆盖用户传入的参数
        self.conf = self.conf_threshold if self.conf_threshold else self.conf
        self.iou = self.iou_threshold if self.iou_threshold else self.iou
    
    def _setup_visualization_params(self):
        """设置可视化配置参数"""
        # 基础框配置
        self.bbox_color = (150, 0, 0)            # 框的 BGR 颜色
        self.bbox_thickness = 2                   # 框的线宽
        self.bbox_labelstr = {
            'font_size': 0.5,         # 字体大小
            'font_thickness': 1,   # 字体粗细
            'offset_x': 0,          # X 方向，文字偏移距离，向右为正
            'offset_y': -10,        # Y 方向，文字偏移距离，向下为正
        }
        
        # 关键点默认配置 - 支持人体17关键点和自定义关键点
        self.kpt_color_map = {
            # 人体17关键点默认配置
            0: {'name': 'nose', 'color': [255, 0, 0], 'radius': 3},          # 鼻子
            1: {'name': 'left_eye', 'color': [0, 255, 0], 'radius': 3},      # 左眼
            2: {'name': 'right_eye', 'color': [0, 0, 255], 'radius': 3},     # 右眼
            3: {'name': 'left_ear', 'color': [255, 255, 0], 'radius': 3},    # 左耳
            4: {'name': 'right_ear', 'color': [255, 0, 255], 'radius': 3},   # 右耳
            5: {'name': 'left_shoulder', 'color': [0, 255, 255], 'radius': 4}, # 左肩
            6: {'name': 'right_shoulder', 'color': [128, 0, 0], 'radius': 4}, # 右肩
            7: {'name': 'left_elbow', 'color': [0, 128, 0], 'radius': 4},    # 左肘
            8: {'name': 'right_elbow', 'color': [0, 0, 128], 'radius': 4},   # 右肘
            9: {'name': 'left_wrist', 'color': [128, 128, 0], 'radius': 4},  # 左手腕
            10: {'name': 'right_wrist', 'color': [128, 0, 128], 'radius': 4}, # 右手腕
            11: {'name': 'left_hip', 'color': [0, 128, 128], 'radius': 4},   # 左髋
            12: {'name': 'right_hip', 'color': [64, 0, 0], 'radius': 4},     # 右髋
            13: {'name': 'left_knee', 'color': [0, 64, 0], 'radius': 4},     # 左膝
            14: {'name': 'right_knee', 'color': [0, 0, 64], 'radius': 4},    # 右膝
            15: {'name': 'left_ankle', 'color': [64, 64, 0], 'radius': 4},   # 左脚踝
            16: {'name': 'right_ankle', 'color': [64, 0, 64], 'radius': 4},  # 右脚踝
        }
        
        # 关键点类别文字配置
        self.kpt_labelstr = {
            'font_size': 0.4,             # 字体大小
            'font_thickness': 1,       # 字体粗细
            'offset_x': 5,             # X 方向，文字偏移距离，向右为正
            'offset_y': 5,            # Y 方向，文字偏移距离，向下为正
        }
        
        # 骨架连接 BGR 配色方案
        self.skeleton_map = [
            # 人体17关键点骨架连接
            {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [196, 75, 255], 'thickness': 2},  # 鼻子-左眼
            {'srt_kpt_id': 0, 'dst_kpt_id': 2, 'color': [196, 75, 255], 'thickness': 2},  # 鼻子-右眼
            {'srt_kpt_id': 1, 'dst_kpt_id': 3, 'color': [196, 75, 255], 'thickness': 2},  # 左眼-左耳
            {'srt_kpt_id': 2, 'dst_kpt_id': 4, 'color': [196, 75, 255], 'thickness': 2},  # 右眼-右耳
            {'srt_kpt_id': 0, 'dst_kpt_id': 5, 'color': [196, 75, 255], 'thickness': 2},  # 鼻子-左肩
            {'srt_kpt_id': 0, 'dst_kpt_id': 6, 'color': [196, 75, 255], 'thickness': 2},  # 鼻子-右肩
            {'srt_kpt_id': 5, 'dst_kpt_id': 6, 'color': [196, 75, 255], 'thickness': 2},  # 左肩-右肩
            {'srt_kpt_id': 5, 'dst_kpt_id': 7, 'color': [196, 75, 255], 'thickness': 2},  # 左肩-左肘
            {'srt_kpt_id': 6, 'dst_kpt_id': 8, 'color': [196, 75, 255], 'thickness': 2},  # 右肩-右肘
            {'srt_kpt_id': 7, 'dst_kpt_id': 9, 'color': [196, 75, 255], 'thickness': 2},  # 左肘-左手腕
            {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'color': [196, 75, 255], 'thickness': 2},  # 右肘-右手腕
            {'srt_kpt_id': 5, 'dst_kpt_id': 11, 'color': [196, 75, 255], 'thickness': 2},  # 左肩-左髋
            {'srt_kpt_id': 6, 'dst_kpt_id': 12, 'color': [196, 75, 255], 'thickness': 2},  # 右肩-右髋
            {'srt_kpt_id': 11, 'dst_kpt_id': 12, 'color': [196, 75, 255], 'thickness': 2},  # 左髋-右髋
            {'srt_kpt_id': 11, 'dst_kpt_id': 13, 'color': [196, 75, 255], 'thickness': 2},  # 左髋-左膝
            {'srt_kpt_id': 12, 'dst_kpt_id': 14, 'color': [196, 75, 255], 'thickness': 2},  # 右髋-右膝
            {'srt_kpt_id': 13, 'dst_kpt_id': 15, 'color': [196, 75, 255], 'thickness': 2},  # 左膝-左脚踝
            {'srt_kpt_id': 14, 'dst_kpt_id': 16, 'color': [196, 75, 255], 'thickness': 2},  # 右膝-右脚踝
        ]
        
        # 可视化控制参数
        self.show_kpt_names = True  # 是否显示关键点名称
        self.show_skeleton = True   # 是否显示骨架连接
        self.show_bbox = True       # 是否显示边界框
    
    def _load_config(self):
        """加载配置文件"""
        import json
        import os
        
        # 如果没有提供配置文件路径，尝试使用与模型同名的.json文件
        if not self.config_path:
            # 获取模型文件的目录和文件名（不带扩展名）
            model_dir = os.path.dirname(self.model_path)
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            self.config_path = os.path.join(model_dir, f"{model_name}.json")
        
        # 检查配置文件是否存在
        if not os.path.exists(self.config_path):
            print(f"配置文件不存在，使用默认配置: {self.config_path}")
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 更新配置
            self._update_config(config)
            print(f"配置文件加载成功: {self.config_path}")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    def _update_config(self, config):
        """更新配置"""
        # 更新边界框配置
        if 'bbox' in config:
            bbox_config = config['bbox']
            if 'color' in bbox_config:
                self.bbox_color = tuple(bbox_config['color'])
            if 'thickness' in bbox_config:
                self.bbox_thickness = bbox_config['thickness']
            if 'label' in bbox_config:
                self.bbox_labelstr.update(bbox_config['label'])
        
        # 更新关键点配置
        if 'keypoints' in config:
            kpt_config = config['keypoints']
            # 转换字符串键为整数
            kpt_config = {int(k): v for k, v in kpt_config.items()}
            self.kpt_color_map.update(kpt_config)
        
        # 更新关键点标签配置
        if 'keypoint_label' in config:
            self.kpt_labelstr.update(config['keypoint_label'])
        
        # 更新骨架连接配置
        if 'skeleton' in config:
            self.skeleton_map = config['skeleton']
        
        # 更新可视化控制参数
        if 'visualization' in config:
            vis_config = config['visualization']
            if 'show_keypoint_names' in vis_config:
                self.show_kpt_names = vis_config['show_keypoint_names']
            if 'show_skeleton' in vis_config:
                self.show_skeleton = vis_config['show_skeleton']
            if 'show_bbox' in vis_config:
                self.show_bbox = vis_config['show_bbox']
    
    def load_model(self):
        """加载模型（延迟加载）"""
        if self.model is not None:
            return True
        
        try:
            # 确保使用相对路径显示
            model_name = Path(self.model_path).name
            print(f"正在加载模型: {model_name}")
            
            # ✅ 老师的方式：直接创建YOLO对象，不使用.predict()
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # 收集模型信息
            self._collect_model_info()
            
            print(f"✅ 模型加载成功: {model_name}")
            
            # 执行预热（如果启用）
            if self.warmup and not self.warmed_up:
                self._perform_warmup()
                
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def _collect_model_info(self):
        """收集模型信息"""
        if self.model is None:
            return
        
        # 初始化基本信息，包含默认值
        self.model_info = {
            'mode': self.mode,
            'device': self.device,
            'input_size': self.img_size,
            'conf_threshold': self.conf,
            'iou_threshold': self.iou,
            'class_names': [],  # 默认空列表
            'class_count': '未知',  # 默认'未知'
            'task': '未知'  # 默认'未知'
        }
        
        # 尝试获取模型详细信息并覆盖默认值
        try:
            if hasattr(self.model, 'names'):
                # 确保values()返回的是列表
                self.model_info['class_names'] = list(self.model.names.values())
                self.model_info['class_count'] = len(self.model.names)
            
            if hasattr(self.model, 'task'):
                self.model_info['task'] = self.model.task
        except Exception as e:  # 捕获具体异常
            print(f"[WARNING] 收集模型详细信息时出错: {e}")
            # 即使出错，也保留默认值
            pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.model_info:
            self._collect_model_info()
        
        return self.model_info.copy()
    
    @staticmethod
    def analyze_model_info(model_path: str) -> Dict[str, Any]:
        """
        分析模型信息（轻量级，不真正加载模型）
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            Dict: 模型信息
        """
        try:
            import os
            from pathlib import Path
            
            filename = Path(model_path).name.lower()
            file_size = os.path.getsize(model_path)
            
            # 根据文件名猜测模式
            if 'cls' in filename or 'classify' in filename:
                task_type = 'classification'
                input_size = '224x224'
            elif 'pose' in filename or 'keypoint' in filename:
                task_type = 'pose'
                input_size = '640x640'
            elif 'seg' in filename:
                task_type = 'segmentation'
                input_size = '640x640'
            else:
                task_type = 'detection'
                input_size = '640x640'
            
            # 尝试加载模型获取更准确的信息
            try:
                model = YOLO(model_path)
                if hasattr(model, 'names'):
                    class_count = len(model.names)
                else:
                    class_count = '未知'
                    
                if hasattr(model, 'task'):
                    task_type = model.task
                    
                # 释放模型
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except:
                class_count = '未知'
            
            return {
                'model_name': Path(model_path).name,
                'task_type': task_type,
                'input_size': input_size,
                'class_count': class_count,
                'file_size': f"{file_size/1024/1024:.1f} MB"
            }
            
        except Exception as e:
            print(f"模型信息分析失败: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理单帧图像 - 统一接口
        
        Args:
            frame: 输入图像 (BGR格式)
            
        Returns:
            Dict: 处理结果，包含图像和统计信息
        """
        start_time = time.time()
        
        # 确保模型已加载
        if not self.load_model():
            return {
                'success': False,
                'error': '模型加载失败',
                'image': frame,
                'stats': {}
            }
        
        try:
            # 根据模式调用不同的处理方法
            if self.mode == 'classification':
                result_dict = self._process_classification(frame)
            elif self.mode == 'pose':
                result_dict = self._process_pose(frame)
            elif self.mode == 'segmentation':
                result_dict = self._process_segmentation(frame)
            else:  # detection
                result_dict = self._process_detection(frame)
            
            # 计算处理时间
            inference_time = time.time() - start_time
            
            # 添加时间信息到统计
            if 'stats' in result_dict:
                result_dict['stats']['inference_time'] = inference_time * 1000  # 转换为毫秒
                result_dict['stats']['fps'] = 1.0 / inference_time if inference_time > 0 else 0
            
            result_dict['success'] = True
            return result_dict
            
        except Exception as e:
            print(f"帧处理失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'image': frame,
                'stats': {
                    'detection_count': 0,
                    'avg_confidence': 0.0,
                    'inference_time': 0,
                    'fps': 0.0
                }
            }
    
    def _process_detection(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理目标检测
        
        Args:
            frame: 输入图像
            
        Returns:
            Dict: 检测结果
        """
        # ✅ 老师的方式：直接调用模型对象
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            verbose=False
        )
        
        result = results[0]
        
        # 提取检测结果
        if result.boxes is None:
            return {
                'image': frame,
                'stats': {
                    'detection_count': 0,
                    'avg_confidence': 0.0
                }
            }
        
        # 提取边界框
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
        
        # 提取类别名称
        class_names = []
        for cls_id in class_ids:
            if hasattr(result, 'names') and cls_id < len(result.names):
                class_names.append(result.names[cls_id])
            else:
                class_names.append(f"object_{cls_id}")
        
        # 构建检测结果列表（用于画框）
        pred_boxes = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            lbl = class_names[i] if i < len(class_names) else f"object_{class_ids[i]}"
            confidence = confidences[i] if i < len(confidences) else 0.0
            track_id = None  # 检测模式没有track_id
            
            pred_boxes.append((x1, y1, x2, y2, lbl, confidence, track_id))
        
        # 使用基类的画框方法
        processed_frame = self.draw_bboxes(frame, pred_boxes)
        
        # 计算统计信息
        detection_count = len(boxes)
        avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0.0
        
        # 类别分布
        class_distribution = {}
        for cls_name in class_names:
            class_distribution[cls_name] = class_distribution.get(cls_name, 0) + 1
        
        return {
            'image': processed_frame,
            'stats': {
                'detection_count': detection_count,
                'avg_confidence': float(avg_confidence),
                'class_distribution': class_distribution,
                'mode': 'detection'
            },
            'raw_data': {
                'boxes': boxes,
                'confidences': confidences,
                'class_ids': class_ids,
                'class_names': class_names
            }
        }
    
    def _process_classification(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理图像分类
        
        Args:
            frame: 输入图像
            
        Returns:
            Dict: 分类结果
        """
        # ✅ 老师的方式：直接调用模型对象
        results = self.model(
            frame,
            conf=self.conf,
            imgsz=self.img_size,
            verbose=False
        )
        
        result = results[0]
        
        # 提取分类结果
        if hasattr(result, 'probs') and result.probs is not None:
            # 获取概率和类别
            probs = result.probs.data.cpu().numpy()
            top_idx = np.argsort(probs)[-1]  # 最高概率的索引
            top_prob = probs[top_idx]
            
            # 获取类别名称
            if hasattr(result, 'names'):
                top_class = result.names[top_idx]
            else:
                top_class = f"class_{top_idx}"
            
            # 在图像上绘制分类结果
            processed_frame = frame.copy()
            
            # 使用PIL绘制中文（从老师代码中借鉴）
            from PIL import Image, ImageDraw, ImageFont
            import os
            
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(img_pil)
            
            # 加载字体 - 使用相对路径
            font_path = "SimHei.ttf"
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 20)
            else:
                print(f"警告: 未找到字体文件 {font_path}，使用默认字体")
                font = ImageFont.load_default()
            
            # 绘制文本
            text = f"{top_class}: {top_prob:.2%}"
            text_position = (30, 30)
            
            # 绘制边框（从老师代码中借鉴）
            border_color = (255, 255, 255)
            border_width = 2
            for dx, dy in [(-border_width, 0), (border_width, 0), (0, -border_width), (0, border_width),
                          (-border_width, -border_width), (-border_width, border_width),
                          (border_width, -border_width), (border_width, border_width)]:
                draw.text((text_position[0] + dx, text_position[1] + dy), text, font=font, fill=border_color)
            
            # 绘制正文
            draw.text(text_position, text, font=font, fill=(255, 0, 0, 1))
            
            # 转换回BGR
            processed_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            return {
                'image': processed_frame,
                'stats': {
                    'detection_count': 1,  # 分类任务固定为1
                    'avg_confidence': float(top_prob),
                    'class_name': top_class,
                    'mode': 'classification'
                },
                'raw_data': {
                    'top_class': top_class,
                    'top_confidence': float(top_prob),
                    'all_probs': probs.tolist()
                }
            }
        else:
            # 没有分类结果
            return {
                'image': frame,
                'stats': {
                    'detection_count': 0,
                    'avg_confidence': 0.0,
                    'mode': 'classification'
                }
            }
            
        print('[CLS]绘制完成', processed_frame.shape, processed_frame.dtype)
    
    def _process_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理关键点检测
        
        Args:
            frame: 输入图像
            
        Returns:
            Dict: 姿态估计结果
        """
        # ✅ 老师的方式：直接调用模型对象
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            verbose=False
        )
        
        result = results[0]
        
        # 提取关键点结果
        if result.boxes is None or result.keypoints is None:
            return {
                'image': frame,
                'stats': {
                    'detection_count': 0,
                    'avg_confidence': 0.0,
                    'mode': 'pose'
                }
            }
        
        # 提取边界框和关键点
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints.xy is not None else []
        keypoints_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else []
        
        # 可视化关键点
        processed_frame = frame.copy()
        
        for person_idx in range(len(boxes)):
            # 绘制边界框
            if self.show_bbox and person_idx < len(boxes):
                box = boxes[person_idx]
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), self.bbox_color, self.bbox_thickness)
            
            # 绘制关键点
            if person_idx < len(keypoints):
                person_keypoints = keypoints[person_idx]
                
                # 绘制骨架连接
                if self.show_skeleton:
                    for skeleton in self.skeleton_map:
                        start_idx = skeleton['srt_kpt_id']
                        end_idx = skeleton['dst_kpt_id']
                        if (start_idx < len(person_keypoints) and end_idx < len(person_keypoints)):
                            start_kp = person_keypoints[start_idx]
                            end_kp = person_keypoints[end_idx]
                            
                            # 检查关键点置信度
                            start_conf = keypoints_conf[person_idx][start_idx] if (person_idx < len(keypoints_conf) and start_idx < len(keypoints_conf[person_idx])) else 1.0
                            end_conf = keypoints_conf[person_idx][end_idx] if (person_idx < len(keypoints_conf) and end_idx < len(keypoints_conf[person_idx])) else 1.0
                            
                            if start_conf > 0.1 and end_conf > 0.1:
                                color = skeleton['color']
                                thickness = skeleton['thickness']
                                cv2.line(processed_frame, 
                                        (int(start_kp[0]), int(start_kp[1])),
                                        (int(end_kp[0]), int(end_kp[1])),
                                        color, thickness)
                
                # 绘制关键点
                for kp_idx, kp in enumerate(person_keypoints):
                    kp_conf = keypoints_conf[person_idx][kp_idx] if (person_idx < len(keypoints_conf) and kp_idx < len(keypoints_conf[person_idx])) else 1.0
                    if kp_conf > 0.1:
                        # 获取关键点配置
                        kpt_config = self.kpt_color_map.get(kp_idx, {
                            'color': [0, 255, 0],
                            'radius': 3
                        })
                        
                        color = tuple(kpt_config['color'])
                        radius = kpt_config['radius']
                        
                        # 绘制关键点圆
                        cv2.circle(processed_frame, (int(kp[0]), int(kp[1])), radius, color, -1)
                        
                        # 显示关键点名称
                        if self.show_kpt_names and 'name' in kpt_config:
                            kpt_name = kpt_config['name']
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_size = self.kpt_labelstr['font_size']
                            font_thickness = self.kpt_labelstr['font_thickness']
                            offset_x = self.kpt_labelstr['offset_x']
                            offset_y = self.kpt_labelstr['offset_y']
                            
                            text_x = int(kp[0]) + offset_x
                            text_y = int(kp[1]) + offset_y
                            
                            # 绘制文本背景
                            (text_width, text_height), baseline = cv2.getTextSize(kpt_name, font, font_size, font_thickness)
                            bg_x1 = text_x
                            bg_y1 = text_y - text_height - baseline
                            bg_x2 = text_x + text_width
                            bg_y2 = text_y + baseline
                            cv2.rectangle(processed_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                            
                            # 绘制文本
                            cv2.putText(processed_frame, kpt_name, (text_x, text_y), font, font_size, (0, 0, 0), font_thickness, cv2.LINE_AA)
        
        # 计算统计信息
        detection_count = len(boxes)
        avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0.0
        
        # 计算关键点数量
        total_keypoints = 0
        for i in range(len(keypoints)):
            if i < len(keypoints_conf):
                visible_keypoints = np.sum(keypoints_conf[i] > 0.1)
                total_keypoints += visible_keypoints
        
        return {
            'image': processed_frame,
            'stats': {
                'detection_count': detection_count,
                'avg_confidence': float(avg_confidence),
                'keypoint_count': total_keypoints,
                'mode': 'pose'
            },
            'raw_data': {
                'boxes': boxes,
                'confidences': confidences,
                'keypoints': keypoints,
                'keypoints_conf': keypoints_conf
            }
        }
    
    def _process_segmentation(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理分割检测
        
        Args:
            frame: 输入图像
            
        Returns:
            Dict: 分割结果
        """
        # ✅ 老师的方式：直接调用模型对象
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            verbose=False
        )
        
        result = results[0]
        
        # 提取分割结果
        if result.masks is None:
            return {
                'image': frame,
                'stats': {
                    'detection_count': 0,
                    'avg_confidence': 0.0,
                    'mode': 'segmentation'
                }
            }
        
        # 获取分割掩码
        masks = result.masks.data.cpu().numpy() if result.masks.data is not None else []
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
        
        # 可视化分割结果
        processed_frame = frame.copy()
        
        # 为每个掩码分配颜色
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128)
        ]
        
        for i in range(len(masks)):
            mask = masks[i]
            class_id = class_ids[i] if i < len(class_ids) else 0
            color = colors[class_id % len(colors)]
            
            # 将掩码转换为二值图像
            mask_binary = (mask > 0).astype(np.uint8) * 255
            
            # 创建彩色掩码
            mask_colored = np.zeros_like(frame)
            mask_colored[:, :, 0] = color[0] * (mask_binary / 255.0)
            mask_colored[:, :, 1] = color[1] * (mask_binary / 255.0)
            mask_colored[:, :, 2] = color[2] * (mask_binary / 255.0)
            
            # 叠加掩码到原图（半透明）
            alpha = 0.3
            processed_frame = cv2.addWeighted(processed_frame, 1, mask_colored.astype(np.uint8), alpha, 0)
            
            # 绘制边界框
            if i < len(boxes):
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                
                # 添加标签
                if hasattr(result, 'names') and class_id < len(result.names):
                    label = result.names[class_id]
                else:
                    label = f"class_{class_id}"
                
                conf = confidences[i] if i < len(confidences) else 0.0
                label_text = f"{label} {conf:.2f}"
                
                # 计算文本大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # 绘制文本背景
                cv2.rectangle(processed_frame, (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), color, -1)
                
                # 绘制文本
                cv2.putText(processed_frame, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 计算统计信息
        detection_count = len(masks)
        avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0.0
        
        # 类别分布
        class_distribution = {}
        for cls_id in class_ids:
            if hasattr(result, 'names') and cls_id < len(result.names):
                cls_name = result.names[cls_id]
            else:
                cls_name = f"class_{cls_id}"
            class_distribution[cls_name] = class_distribution.get(cls_name, 0) + 1
        
        return {
            'image': processed_frame,
            'stats': {
                'detection_count': detection_count,
                'avg_confidence': float(avg_confidence),
                'class_distribution': class_distribution,
                'mode': 'segmentation'
            },
            'raw_data': {
                'masks': masks,
                'boxes': boxes,
                'confidences': confidences,
                'class_ids': class_ids
            }
        }
    
    def _perform_warmup(self):
        """
        执行模型预热，针对不同模型类型使用合适的输入尺寸
        预热可以减少首次推理的延迟，特别是对于GPU模型
        """
        if self.model is None:
            return
            
        print(f"🔄 开始模型预热 | 模式: {self.mode} | 输入尺寸: {self.img_size}x{self.img_size}")
        start_time = time.time()
        
        try:
            # 根据模型类型使用合适的输入尺寸
            # 分类模型通常使用224，其他模型使用640
            warmup_size = self.img_size  # 已经根据模式设置好了正确的尺寸
            
            # 创建虚拟输入数据
            dummy_input = np.random.randint(0, 255, (warmup_size, warmup_size, 3), dtype=np.uint8)
            
            # 使用torch.no_grad()减少内存使用
            with torch.no_grad():
                # 进行多次预热推理（通常3-5次足够）
                for i in range(3):
                    # 执行推理，但不处理结果
                    _ = self.model(
                        dummy_input, 
                        conf=self.conf, 
                        iou=self.iou, 
                        imgsz=warmup_size, 
                        verbose=False
                    )
            
            # 清理缓存（特别是在GPU上运行时）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.warmed_up = True
            warmup_time = (time.time() - start_time) * 1000  # 转换为毫秒
            print(f"✅ 模型预热完成 | 耗时: {warmup_time:.2f}ms")
            
        except Exception as e:
            print(f"❌ 模型预热失败: {e}")
            # 预热失败不影响模型使用，只是首次推理可能较慢
    
    def update_params(self, conf_threshold=None, iou_threshold=None):
        """实时更新推理参数
        
        Args:
            conf_threshold: 新的置信度阈值（0.0-1.0）
            iou_threshold: 新的IOU阈值（0.0-1.0）
        
        Returns:
            bool: 参数更新是否成功
        """
        try:
            if conf_threshold is not None:
                # 验证置信度阈值范围
                if 0.0 <= conf_threshold <= 1.0:
                    self.conf = conf_threshold
                    self.conf_threshold = conf_threshold  # 更新原始属性以便保持一致性
                    print(f"[INFO] 置信度阈值更新为: {conf_threshold}")
                else:
                    print(f"[ERROR] 置信度阈值必须在0.0到1.0之间，提供的值: {conf_threshold}")
                    return False
            
            if iou_threshold is not None:
                # 验证IOU阈值范围
                if 0.0 <= iou_threshold <= 1.0:
                    self.iou = iou_threshold
                    self.iou_threshold = iou_threshold  # 更新原始属性以便保持一致性
                    print(f"[INFO] IOU阈值更新为: {iou_threshold}")
                else:
                    print(f"[ERROR] IOU阈值必须在0.0到1.0之间，提供的值: {iou_threshold}")
                    return False
            
            return True
        except Exception as e:
            print(f"[ERROR] 更新参数时出错: {e}")
            return False
    
    def __call__(self, frame: np.ndarray) -> Dict[str, Any]:
        """使对象可调用"""
        return self.process_frame(frame)