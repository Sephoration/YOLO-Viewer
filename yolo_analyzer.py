"""
YOLO统一分析器
整合三个模式：检测、分类、关键点
直接调用模型对象，禁止使用.predict()方法
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Tuple
from ultralytics import YOLO

from baseDetect import baseDetect


class UnifiedYOLO(baseDetect):
    """
    统一YOLO处理器 - 整合三个模式
    遵循老师要求的代码风格：直接调用模型对象
    """
    
    # ---------------------------------------------------------
    # 2. 目标追踪（Tracking）
    # ---------------------------------------------------------
    
    def __init__(self, model_path: str, mode: str = 'auto',
                 conf_threshold: float = 0.25, iou_threshold: float = 0.7):
        """
        初始化统一YOLO处理器
        
        Args:
            model_path: 模型文件路径
            mode: 模式 ('auto', 'detection', 'classification', 'pose')
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        super().__init__()
        
        self.model_path = model_path
        self.mode = self._detect_mode(model_path) if mode == 'auto' else mode
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 设备选择
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 模型对象（延迟加载）
        self.model = None
        self.model_info = {}
        
        # 模式特定参数
        self._setup_mode_params()
        
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
    
    def load_model(self):
        """加载模型（延迟加载）"""
        if self.model is not None:
            return True
        
        try:
            print(f"正在加载模型: {self.model_path}")
            
            # ✅ 老师的方式：直接创建YOLO对象，不使用.predict()
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # 收集模型信息
            self._collect_model_info()
            
            print(f"✅ 模型加载成功: {Path(self.model_path).name}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def _collect_model_info(self):
        """收集模型信息"""
        if self.model is None:
            return
        
        self.model_info = {
            'mode': self.mode,
            'device': self.device,
            'input_size': self.img_size,
            'conf_threshold': self.conf,
            'iou_threshold': self.iou,
        }
        
        # 尝试获取模型详细信息
        try:
            if hasattr(self.model, 'names'):
                self.model_info['class_names'] = list(self.model.names.values())
                self.model_info['num_classes'] = len(self.model.names)
            
            if hasattr(self.model, 'task'):
                self.model_info['task'] = self.model.task
        except:
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
            
            # 加载字体
            font_path = "SimHei.ttf"
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 20)
            else:
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
        
        # 绘制边界框和关键点
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 脸部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 躯干
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 四肢
        ]
        
        skeleton_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]
        
        for person_idx in range(len(boxes)):
            # 绘制边界框
            if person_idx < len(boxes):
                box = boxes[person_idx]
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制关键点
            if person_idx < len(keypoints):
                person_keypoints = keypoints[person_idx]
                
                # 绘制骨架连接
                for connection in skeleton_connections:
                    start_idx, end_idx = connection
                    if (start_idx < len(person_keypoints) and end_idx < len(person_keypoints)):
                        start_kp = person_keypoints[start_idx]
                        end_kp = person_keypoints[end_idx]
                        
                        # 检查关键点置信度
                        start_conf = keypoints_conf[person_idx][start_idx] if (person_idx < len(keypoints_conf) and start_idx < len(keypoints_conf[person_idx])) else 1.0
                        end_conf = keypoints_conf[person_idx][end_idx] if (person_idx < len(keypoints_conf) and end_idx < len(keypoints_conf[person_idx])) else 1.0
                        
                        if start_conf > 0.1 and end_conf > 0.1:
                            color = skeleton_colors[connection[0] % len(skeleton_colors)]
                            cv2.line(processed_frame, 
                                    (int(start_kp[0]), int(start_kp[1])),
                                    (int(end_kp[0]), int(end_kp[1])),
                                    color, 2)
                
                # 绘制关键点
                for kp_idx, kp in enumerate(person_keypoints):
                    kp_conf = keypoints_conf[person_idx][kp_idx] if (person_idx < len(keypoints_conf) and kp_idx < len(keypoints_conf[person_idx])) else 1.0
                    if kp_conf > 0.1:
                        color_intensity = int(255 * kp_conf)
                        color = (0, color_intensity, 255 - color_intensity)
                        cv2.circle(processed_frame, (int(kp[0]), int(kp[1])), 3, color, -1)
        
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
    
    def __call__(self, frame: np.ndarray) -> Dict[str, Any]:
        """使对象可调用"""
        return self.process_frame(frame)