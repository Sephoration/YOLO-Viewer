"""
软件界面_ui部分
保留单独运行能力、以方便测试
定义和布局UI组件
提供用户交互接口
显示数据和状态
触发简单事件（菜单点击、按钮点击）
错误处理应该放在控制器中
不要在UI中添加进度条组件
"""

import sys
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QSlider, QPushButton, QGroupBox, 
                               QToolBar, QSizePolicy, QMenu, QScrollArea)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QFont
from collections import OrderedDict


# ============================================================================
# 常量定义类 - 集中管理所有UI相关常量
# ============================================================================
class UIContants:
    # 尺寸常量
    MIN_DISPLAY_WIDTH = 320       # 显示标签最小宽度
    MIN_DISPLAY_HEIGHT = 180      # 显示标签最小高度
    CONTROL_HEIGHT = 40           # 视频控制部件高度
    PLAY_BUTTON_SIZE = 32         # 播放/暂停按钮尺寸
    FILE_LABEL_HEIGHT = 25        # 文件名标签高度
    TIME_LABEL_MIN_WIDTH = 85     # 时间标签最小宽度
    RIGHT_PANEL_MIN_WIDTH = 220   # 右侧面板最小宽度
    BUTTON_MIN_HEIGHT = 35        # 按钮最小高度
    
    # 布局常量
    PADDING_SMALL = 10            # 小间距/内边距
    PADDING_MEDIUM = 12           # 中等间距/内边距
    LAYOUT_SPACING = 10           # 布局间距
    FILE_LABEL_MAX_WIDTH = 300    # 文件名标签最大宽度
    FILE_LABEL_WIDTH_OFFSET = 20  # 文件名标签宽度偏移
    
    # 进度条常量
    PROGRESS_RANGE = 1000         # 进度条范围
    
    # 字体常量
    TIME_FONT_SIZE = 9            # 时间显示字体大小
    
    # 滑块常量
    SLIDER_LABEL_WIDTH = 60       # 滑块标签宽度
    SLIDER_WIDTH = 80             # 滑块宽度
    SLIDER_VALUE_LABEL_WIDTH = 35 # 滑块值标签宽度
    
    # 窗口常量
    MIN_WINDOW_WIDTH = 1000       # 窗口最小宽度
    MIN_WINDOW_HEIGHT = 650       # 窗口最小高度
    INITIAL_WINDOW_WIDTH = 1140   # 初始窗口宽度
    INITIAL_WINDOW_HEIGHT = 675   # 初始窗口高度
    
    # 缓存常量
    MAX_PIXMAP_CACHE_SIZE = 100   # QPixmap缓存池最大帧数
    PIXMAP_CACHE_SIZE = 100       # QPixmap缓存池大小（最大帧数）
    
    # 位置常量
    FILE_LABEL_X = 10             # 文件名标签X坐标
    FILE_LABEL_Y = 10             # 文件名标签Y坐标


# ============================================================================
# 自定义组件：保持16:9比例且左右贴边的显示标签
# 功能：确保图像显示区域始终保持16:9比例，随父容器宽度自适应调整
# ============================================================================
class AspectRatioDisplayLabel(QLabel):
    """保持16:9显示比例的居中图像标签，带QPixmap LRU缓存池"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(UIContants.MIN_DISPLAY_WIDTH, UIContants.MIN_DISPLAY_HEIGHT)  # 最小16:9尺寸
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 初始化QPixmap LRU缓存池
        self._pixmap_cache = OrderedDict()
        
    def setPixmap(self, pixmap, frame_id=None):
        """重写setPixmap方法以保持图像比例并使用LRU缓存
        
        Args:
            pixmap: QPixmap对象
            frame_id: 可选的帧ID，用于缓存管理
        """
        if not pixmap.isNull():
            # 如果提供了frame_id，使用缓存机制
            if frame_id is not None:
                # 检查缓存中是否已存在
                if frame_id in self._pixmap_cache:
                    # 更新访问顺序（LRU）
                    scaled_pixmap = self._pixmap_cache.pop(frame_id)
                    self._pixmap_cache[frame_id] = scaled_pixmap
                    super().setPixmap(scaled_pixmap)
                    return
                
                # 计算缩放比例以适应标签大小
                scaled_pixmap = self._scale_pixmap_to_fit(pixmap)
                
                # 添加到缓存并管理缓存大小
                self._pixmap_cache[frame_id] = scaled_pixmap
                if len(self._pixmap_cache) > UIContants.PIXMAP_CACHE_SIZE:
                    # 移除最不常用的项（OrderedDict的第一个项）
                    self._pixmap_cache.popitem(last=False)
                
                super().setPixmap(scaled_pixmap)
            else:
                # 没有frame_id时直接处理
                scaled_pixmap = self._scale_pixmap_to_fit(pixmap)
                super().setPixmap(scaled_pixmap)
        else:
            super().setPixmap(pixmap)
    
    def _scale_pixmap_to_fit(self, pixmap):
        """缩放pixmap以适应标签大小，保持宽高比"""
        return pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    
    def clear_cache(self):
        """清空QPixmap缓存池"""
        self._pixmap_cache.clear()
        
    def resizeEvent(self, event):
        """重写resize事件，保持16:9比例"""
        super().resizeEvent(event)
        
        # 获取父容器的可用空间（考虑布局内边距和底部控制栏）
        parent_widget = self.parent() if self.parent() else None
        if parent_widget:
            rect = parent_widget.contentsRect()
            parent_width = rect.width()
            parent_height = rect.height()

            # 如果父容器包含名为 VideoControlWidget 的控件，尝试将其高度从可用高度中扣除
            try:
                ctrl = parent_widget.findChild(QWidget, "VideoControlWidget")
                if ctrl and ctrl.isVisible():
                    ctrl_h = ctrl.sizeHint().height()
                    parent_layout = parent_widget.layout()
                    spacing = parent_layout.spacing() if parent_layout else 0
                    parent_height = max(0, parent_height - (ctrl_h + spacing))
            except Exception:
                pass
        else:
            parent_width = self.width()
            parent_height = self.height()
        
        # 根据可用宽度计算16:9的理想尺寸
        ideal_width = parent_width
        ideal_height = int(ideal_width * 9 / 16)
        
        # 如果根据宽度计算的高度超过了可用高度，则根据高度调整
        if ideal_height > parent_height:
            ideal_height = parent_height
            ideal_width = int(ideal_height * 16 / 9)
        
        # 设置固定尺寸以保持比例
        self.setFixedSize(ideal_width, ideal_height)


# ============================================================================
# 左侧展示区域组件
# 功能：负责图像/视频/摄像头内容展示，包含播放控制和文件名显示
# ============================================================================
# ============================================================================
# 左侧展示区域组件
# 功能：负责图像/视频/摄像头内容展示，包含播放控制和文件名显示
# ============================================================================
class LeftDisplayPanel(QWidget):
    """左侧展示面板：4:3容器，文件名直接印在黑边上，16:9区域贴边"""
    
    # 定义信号
    play_pause_clicked = Signal()   # 播放/暂停按钮点击信号
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)  # 4:3最小尺寸
        self.current_mode = None  # 模式：'image', 'video', 'camera'
        self.is_playing = False   # 视频播放状态标记
        self._init_ui()           # 初始化UI
        self._setup_style()       # 设置样式
    
    def _init_ui(self):
        """初始化UI组件"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 黑色显示容器
        self.display_container = QWidget()
        self.display_container.setObjectName("DisplayContainer")
        
        # 容器布局 - 使用弹性空间让内容居中
        container_layout = QVBoxLayout(self.display_container)
        container_layout.setContentsMargins(0, 0, 0, 0)  # 移除左右内边距，让16:9区域完全贴边
        container_layout.setSpacing(0)
        
        # 添加上方弹性空间
        container_layout.addStretch(1)
        
        # 16:9显示标签 - 居中显示
        self.display_label = AspectRatioDisplayLabel()
        self.display_label.setObjectName("DisplayLabel")
        container_layout.addWidget(self.display_label, alignment=Qt.AlignCenter)
        
        # 添加下方弹性空间
        container_layout.addStretch(1)
        
        main_layout.addWidget(self.display_container)
        
        # 创建叠加层：文件名标签和视频控制部件
        self._create_overlay_widgets()
        
        # 默认禁用视频控制（图片模式）
        self.set_controls_enabled(False)
    
    def _create_overlay_widgets(self):
        """创建叠加在显示区域上的控件（文件名和视频控制）"""
        # 创建文件名标签（叠加在黑色背景上）
        self.filename_label = QLabel(self.display_container)
        self.filename_label.setObjectName("FilenameLabel")
        self.filename_label.setFixedHeight(UIContants.FILE_LABEL_HEIGHT)
        self.filename_label.hide()  # 默认隐藏
        
        # 创建视频控制部件（叠加层）
        self.video_control_widget = self._create_video_control_widget()
        self.video_control_widget.setObjectName("VideoControlWidget")
        self.video_control_widget.setParent(self.display_container)
        self.video_control_widget.hide()  # 默认隐藏
        self.video_control_widget.raise_()  # 确保在最上层
    
    def _create_video_control_widget(self):
        """创建视频控制部件（播放/暂停按钮）- 叠加层版本"""
        control_widget = QWidget()
        control_widget.setFixedHeight(UIContants.CONTROL_HEIGHT)
        
        # 视频控制布局
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(UIContants.PADDING_MEDIUM, 0, UIContants.PADDING_MEDIUM, 0)
        control_layout.setSpacing(UIContants.LAYOUT_SPACING)
        
        # 播放/暂停按钮
        self.play_pause_button = QPushButton("▶")
        self.play_pause_button.setObjectName("PlayPauseButton")
        self.play_pause_button.setFixedSize(UIContants.PLAY_BUTTON_SIZE, UIContants.PLAY_BUTTON_SIZE)
        self.play_pause_button.clicked.connect(self._on_play_pause_clicked)
        
        # 时间显示标签
        self.time_label = QLabel("--:--")
        self.time_label.setObjectName("TimeLabel")
        self.time_label.setMinimumWidth(UIContants.TIME_LABEL_MIN_WIDTH)
        self.time_label.setAlignment(Qt.AlignCenter)
        
        # 设置时间显示字体
        font = QFont("Consolas", UIContants.TIME_FONT_SIZE)
        self.time_label.setFont(font)
        
        # 添加到布局
        control_layout.addWidget(self.play_pause_button)
        control_layout.addWidget(self.time_label, 1)       # 占1份空间
        
        return control_widget
    
    def _on_play_pause_clicked(self):
        """播放/暂停按钮点击事件处理"""
        self.is_playing = not self.is_playing  # 切换播放状态
        self._update_play_button_state()       # 更新按钮显示
        self.play_pause_clicked.emit()         # 发射点击信号
    
    def _update_play_button_state(self):
        """更新播放按钮显示状态"""
        self.play_pause_button.setText("⏸" if self.is_playing else "▶")
    
    def resizeEvent(self, event):
        """重写resize事件，调整叠加控件位置"""
        super().resizeEvent(event)
        
        # 更新文件名标签位置和宽度
        if self.filename_label.isVisible():
            self.filename_label.move(10, 10)  # 左上角偏移10px
            # 最大宽度不超过容器宽度-20，且不超过300px
            self.filename_label.setFixedWidth(min(300, self.display_container.width() - 20))
        
        # 更新视频控制部件位置（水平居中，底部固定位置）
        if self.video_control_widget.isVisible():
            control_width = min(self.display_container.width() - 2 * UIContants.PADDING_MEDIUM, 
                               UIContants.PLAY_BUTTON_SIZE + UIContants.TIME_LABEL_MIN_WIDTH + 200)  # 限制最大宽度
            
            # 计算水平居中位置
            x = (self.display_container.width() - control_width) // 2
            y = self.display_container.height() - UIContants.CONTROL_HEIGHT - UIContants.PADDING_SMALL  # 底部留白
            
            self.video_control_widget.setGeometry(x, y, control_width, UIContants.CONTROL_HEIGHT)
            self.video_control_widget.raise_()  # 确保在最上层
    
    def _setup_style(self):
        """设置样式表"""
        self.setStyleSheet("""
            QWidget {{
                background-color: #f5f5f5;
            }}
            QWidget#DisplayContainer {{
                background-color: #2a2a2a;  /* 黑色显示区域背景 */
            }}
            QLabel#DisplayLabel {{
                background-color: #2a2a2a;
                border: none;
                color: white;
                font-size: 12px;
            }}
            QLabel#FilenameLabel {{
                background-color: transparent;
                color: white;
                padding: 0;
                border: none;
                border-radius: 0;
                font-family: 'Microsoft YaHei';
                font-size: 11px;
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
            }}
            QWidget#VideoControlWidget {{
                background-color: rgba(0, 0, 0, 0.85);  /* 半透明黑色控制栏 */
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            QPushButton#PlayPauseButton {{
                background-color: rgba(0, 120, 215, 0.9);
                color: white;
                border: none;
                border-radius: 16px;
                font-size: 14px;
                font-weight: bold;
                padding: 0;
            }}
            QPushButton#PlayPauseButton:hover {{
                background-color: rgba(0, 120, 215, 1.0);
            }}
            QPushButton#PlayPauseButton:pressed {{
                background-color: rgba(0, 90, 158, 0.9);
            }}
            QPushButton#PlayPauseButton:disabled {{
                background-color: rgba(80, 80, 80, 0.5);
                color: rgba(180, 180, 180, 0.8);
            }}
            
            /* 进度条样式 */
            QSlider#ProgressSlider::groove:horizontal {{
                height: {0}px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #444444, 
                    stop:1 #666666
                );
                border-radius: {1}px;
            }}
            QSlider#ProgressSlider::handle:horizontal {{
                background: qradialgradient(
                    cx:0.5, cy:0.5, radius:0.5,
                    fx:0.5, fy:0.5,
                    stop:0.6 #0078d7,
                    stop:1.0 #005a9e
                );
                width: {2}px;
                height: {3}px;
                margin: -5px 0;
                border-radius: {4}px;
                border: 2px solid #ffffff;
            }}
            QSlider#ProgressSlider::handle:horizontal:hover {{
                background: qradialgradient(
                    cx:0.5, cy:0.5, radius:0.5,
                    fx:0.5, fy:0.5,
                    stop:0.6 #106ebe,
                    stop:1.0 #004578
                );
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QSlider#ProgressSlider::handle:horizontal:pressed {{
                background: qradialgradient(
                    cx:0.5, cy:0.5, radius:0.5,
                    fx:0.5, fy:0.5,
                    stop:0.6 #005a9e,
                    stop:1.0 #003b6a
                );
            }}
            QSlider#ProgressSlider:disabled::groove:horizontal {{
                background: #333333;
            }}
            QSlider#ProgressSlider:disabled::handle:horizontal {{
                background: #555555;
                border: 2px solid #777777;
            }}
            
            /* 时间显示样式 */
            QLabel#TimeLabel {{
                color: white;
                background-color: rgba(0, 0, 0, 0.7);
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                font-weight: bold;
                padding: 4px 10px;
                border-radius: 4px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            QLabel#TimeLabel:disabled {{
                color: #aaaaaa;
                background-color: rgba(40, 40, 40, 0.7);
                border: 1px solid rgba(100, 100, 100, 0.2);
            }}
        """.format(
            6,  # 进度条槽高度
            3,  # 进度条槽圆角
            16, # 进度条手柄宽度
            16, # 进度条手柄高度
            8   # 进度条手柄圆角
        ))
    
    # ===== 公共接口方法 =====
    
    def update_info(self, file_name="", mode="image"):
        """更新显示信息（文件名和模式）"""
        self.current_mode = mode
        
        # 更新文件名显示
        if mode == "camera":
            self.filename_label.setText("摄像头")
            self.filename_label.show()
        elif file_name:
            # 过长文件名显示省略号
            text = "..." + file_name[-27:] if len(file_name) > 30 else file_name
            self.filename_label.setText(text)
            self.filename_label.show()
            # 设置文件名标签的位置
            self.filename_label.move(12, 12)
        else:
            self.filename_label.hide()
        
        # 根据模式启用/禁用视频控制
        if mode == "video":
            self.video_control_widget.show()
            self.set_controls_enabled(True)
            self.is_playing = True  # 视频默认播放状态
            self._update_play_button_state()
        else:
            self.video_control_widget.hide()
            self.set_controls_enabled(False)
            self.is_playing = False
        
        # 更新叠加控件位置
        self.resizeEvent(None)
    
    def set_display_image(self, pixmap, frame_id=None):
        """设置显示图像（保持原比例）
        
        Args:
            pixmap: QPixmap对象
            frame_id: 可选的帧ID，用于缓存管理
        """
        if pixmap:
            # 使用带缓存的setPixmap方法
            self.display_label.setPixmap(pixmap, frame_id)
        else:
            self.display_label.clear()
    
    def clear_display(self):
        """清空显示区域"""
        self.display_label.clear()
        self.display_label.setText("等待显示图像...")
        self.filename_label.hide()
        self.video_control_widget.hide()
        self.set_controls_enabled(False)
    
    def set_controls_enabled(self, enabled):
        """启用/禁用视频控制部件
        
        Args:
            enabled: 是否启用控制部件
        """
        self.play_pause_button.setEnabled(enabled)
        self.time_label.setEnabled(enabled)
    
    def set_play_state(self, is_playing: bool):
        """设置播放状态并更新播放按钮
        
        Args:
            is_playing: 是否正在播放
        """
        self.is_playing = is_playing
        self._update_play_button_state()


# ============================================================================
# 右侧控制面板组件
# 功能：提供参数设置、模型信息展示、统计信息显示和控制按钮
# ============================================================================
class RightControlPanel(QWidget):
    """右侧控制面板：参数设置和统计信息"""
    
    # 定义信号
    iou_changed = Signal(float)          # IOU阈值改变信号
    confidence_changed = Signal(float)   # 置信度阈值改变信号
    delay_changed = Signal(int)          # 延迟时间改变信号
    line_width_changed = Signal(int)     # 线宽改变信号
    save_screenshot = Signal()           # 保存截图信号
    start_inference = Signal()           # 开始推理信号
    stop_inference = Signal()            # 停止推理信号
    
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(UIContants.RIGHT_PANEL_MIN_WIDTH)
        self._init_ui()    # 初始化UI
        self._setup_style()  # 设置样式
    
    def _init_ui(self):
        """初始化UI组件"""
        # 创建滚动区域（支持垂直滚动）
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll.setStyleSheet("QScrollArea { border: none; }")
        
        # 滚动区域内容组件
        content_widget = QWidget()
        self._content_layout = QVBoxLayout(content_widget)
        self._content_layout.setContentsMargins(UIContants.PADDING_SMALL, UIContants.PADDING_SMALL, UIContants.PADDING_SMALL, UIContants.PADDING_SMALL)
        self._content_layout.setSpacing(UIContants.LAYOUT_SPACING)
        
        # 添加各组组件
        self.model_info_group = self._create_model_info_group()
        self._content_layout.addWidget(self.model_info_group)
        
        self.params_group = self._create_params_group()
        self._content_layout.addWidget(self.params_group)
        
        self.stats_group = self._create_stats_group()
        self._content_layout.addWidget(self.stats_group)
        
        self.control_buttons = self._create_control_buttons()
        self._content_layout.addWidget(self.control_buttons)
        
        self.save_button = self._create_save_button()
        self._content_layout.addWidget(self.save_button)
        
        # 添加弹性空间（将内容顶到上方）
        self._content_layout.addStretch(1)
        
        # 设置滚动区域内容
        main_scroll.setWidget(content_widget)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(main_scroll)
    
    def _create_model_info_group(self):
        """创建模型信息显示组"""
        group = QGroupBox("模型信息")
        layout = QVBoxLayout(group)
        
        self.model_name_label = QLabel("模型: 未加载")
        layout.addWidget(self.model_name_label)
        
        self.task_type_label = QLabel("任务: 未知")
        layout.addWidget(self.task_type_label)
        
        self.input_size_label = QLabel("尺寸: 未知")
        layout.addWidget(self.input_size_label)
        
        self.class_count_label = QLabel("类别: 未知")
        layout.addWidget(self.class_count_label)
        
        return group
    
    def _create_params_group(self):
        """创建参数调节组（IOU、置信度、延迟、线宽）"""
        group = QGroupBox("推理参数")
        layout = QVBoxLayout(group)
        layout.setSpacing(5)
        
        # IOU阈值滑块
        iou_container = self._create_slider_widget(
            "IOU阈值:", 0.0, 1.0, 0.45, 100,
            self.iou_changed, lambda v: v / 100
        )
        self.iou_slider = iou_container["slider"]
        self.iou_value_label = iou_container["label"]
        layout.addWidget(iou_container["widget"])
        
        # 置信度阈值滑块
        conf_container = self._create_slider_widget(
            "置信度:", 0.0, 1.0, 0.5, 100,
            self.confidence_changed, lambda v: v / 100
        )
        self.confidence_slider = conf_container["slider"]
        self.confidence_value_label = conf_container["label"]
        layout.addWidget(conf_container["widget"])
        
        # 延迟时间滑块
        delay_container = self._create_slider_widget(
            "延迟(ms):", 0, 100, 10, 1,
            self.delay_changed
        )
        self.delay_slider = delay_container["slider"]
        self.delay_value_label = delay_container["label"]
        layout.addWidget(delay_container["widget"])
        
        # 线宽滑块
        line_width_container = self._create_slider_widget(
            "线宽:", 1, 10, 2, 1,
            self.line_width_changed
        )
        self.line_width_slider = line_width_container["slider"]
        self.line_width_value_label = line_width_container["label"]
        layout.addWidget(line_width_container["widget"])
        
        return group
    
    def _create_slider_widget(self, label_text, min_val, max_val, default_val, 
                             scale_factor=1, signal=None, transform_func=None):
        """
        创建滑块控件组
        参数:
            label_text: 标签文本
            min_val: 最小值
            max_val: 最大值
            default_val: 默认值
            scale_factor: 缩放因子（用于将浮点数转为整数处理）
            signal: 信号对象
            transform_func: 值转换函数
        返回:
            包含widget、slider、label的字典
        """
        widget = QWidget()
        widget.setMinimumHeight(30)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标签
        label = QLabel(label_text)
        label.setMinimumWidth(UIContants.SLIDER_LABEL_WIDTH)
        label.setMaximumWidth(UIContants.SLIDER_LABEL_WIDTH)
        layout.addWidget(label)
        
        # 滑块
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val * scale_factor, max_val * scale_factor)
        slider.setValue(default_val * scale_factor)
        slider.setMinimumWidth(UIContants.SLIDER_WIDTH)
        
        # 值显示标签
        value_label = QLabel(str(default_val))
        value_label.setMinimumWidth(UIContants.SLIDER_VALUE_LABEL_WIDTH)
        value_label.setMaximumWidth(UIContants.SLIDER_VALUE_LABEL_WIDTH)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # 连接信号
        if signal and transform_func:
            slider.valueChanged.connect(
                lambda v: self._on_slider_changed(v, value_label, signal, transform_func)
            )
        elif signal:
            slider.valueChanged.connect(
                lambda v: self._on_slider_changed(v, value_label, signal)
            )
        
        layout.addWidget(slider)
        layout.addWidget(value_label)
        
        return {
            "widget": widget,
            "slider": slider,
            "label": value_label
        }
    
    def _on_slider_changed(self, value, value_label, signal, transform_func=None):
        """滑块值改变时的处理函数"""
        # 转换值（如果需要）
        if transform_func:
            display_value = transform_func(value)
            actual_value = display_value
        else:
            display_value = value
            actual_value = value
        
        # 更新显示文本
        if isinstance(display_value, float):
            value_label.setText(f"{display_value:.2f}")
        else:
            value_label.setText(str(display_value))
        
        # 发射信号
        signal.emit(actual_value)
    
    def _create_stats_group(self):
        """创建实时统计组"""
        group = QGroupBox("实时统计")
        layout = QVBoxLayout(group)
        
        self.detection_count_label = QLabel("检测数: 0")
        layout.addWidget(self.detection_count_label)
        
        self.confidence_label = QLabel("置信度: 0.00")
        layout.addWidget(self.confidence_label)
        
        self.inference_time_label = QLabel("推理时间: 0ms")
        layout.addWidget(self.inference_time_label)
        
        self.fps_label = QLabel("FPS: 0.0")
        layout.addWidget(self.fps_label)
        
        return group
    
    def _create_control_buttons(self):
        """创建控制按钮组（开始/停止推理）"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        self.start_button = QPushButton("开始")
        self.start_button.setMinimumHeight(UIContants.BUTTON_MIN_HEIGHT)
        self.start_button.clicked.connect(self.start_inference.emit)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.setMinimumHeight(UIContants.BUTTON_MIN_HEIGHT)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_inference.emit)
        
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        return widget
    
    def _create_save_button(self):
        """创建保存截图按钮"""
        button = QPushButton("保存截图")
        button.setMinimumHeight(UIContants.BUTTON_MIN_HEIGHT)
        button.clicked.connect(self.save_screenshot.emit)
        return button
    
    def _setup_style(self):
        """设置样式表"""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: normal;
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 10px;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                font-weight: normal;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLabel {
                color: #333333;
                padding: 2px;
                font-size: 11px;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #d0d0d0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0078d7;
                width: 12px;
                height: 12px;
                margin: -3px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #106ebe;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
    
    # ===== 公共接口方法 =====
    
    def update_model_info(self, model_path="", task_type="", input_size="", class_count=""):
        """更新模型信息显示"""
        if model_path:
            import os
            model_name = os.path.basename(model_path)
            self.model_name_label.setText(f"模型: {model_name}")
        else:
            self.model_name_label.setText("模型: 未加载")
        
        self.task_type_label.setText(f"任务: {task_type}" if task_type else "任务: 未知")
        self.input_size_label.setText(f"尺寸: {input_size}" if input_size else "尺寸: 未知")
        self.class_count_label.setText(f"类别: {class_count}" if class_count else "类别: 未知")
    
    def update_statistics(self, detection_count=0, confidence=0.0, inference_time=0, fps=0.0):
        """更新统计信息显示"""
        self.detection_count_label.setText(f"检测数: {detection_count}")
        self.confidence_label.setText(f"置信度: {confidence:.2f}")
        self.inference_time_label.setText(f"推理时间: {inference_time:.2f}ms")
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def get_parameters(self):
        """获取当前参数值"""
        return {
            "iou_threshold": self.iou_slider.value() / 100.0,
            "confidence_threshold": self.confidence_slider.value() / 100.0,
            "delay_ms": self.delay_slider.value(),
            "line_width": self.line_width_slider.value()
        }
    
    def set_parameters(self, iou_threshold=None, confidence_threshold=None, delay_ms=None, line_width=None):
        """设置参数值"""
        if iou_threshold is not None:
            self.iou_slider.setValue(int(iou_threshold * 100))
        if confidence_threshold is not None:
            self.confidence_slider.setValue(int(confidence_threshold * 100))
        if delay_ms is not None:
            self.delay_slider.setValue(delay_ms)
        if line_width is not None:
            self.line_width_slider.setValue(line_width)
    
    def set_control_state(self, is_running):
        """设置控制按钮状态（根据推理是否运行）"""
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)


# ============================================================================
# 主窗口UI
# 功能：整合左侧展示区和右侧控制区，提供菜单栏和工具栏
# ============================================================================
class YOLOMainWindowUI(QMainWindow):
    """主窗口UI类 - 简化版，只负责UI展示"""
    
    # 只保留与菜单直接相关的信号
    file_menu_init = Signal()
    file_menu_save_as = Signal()
    file_menu_save = Signal()
    file_menu_exit = Signal()
    model_load = Signal()
    image_open = Signal()
    video_open = Signal()
    camera_open = Signal()
    detect_settings = Signal()  # 新增：检测设置信号
    help_menu_about = Signal()
    help_menu_manual = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO多功能检测系统")
        self.setGeometry(100, 100, 1140, 675)  # 初始窗口大小
        
        self._init_ui()        # 初始化主UI
        self._setup_toolbar()  # 设置工具栏
        self._setup_signals()  # 设置信号连接
    
    def _init_ui(self):
        """初始化主UI布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局（水平排列左右面板）
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 左侧展示面板（占4份空间）
        self.left_panel = LeftDisplayPanel()
        main_layout.addWidget(self.left_panel, 4)
        
        # 右侧控制面板（占1份空间）
        self.right_panel = RightControlPanel()
        main_layout.addWidget(self.right_panel, 1)
    
    def _setup_toolbar(self):
        """设置单行工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        
        # 工具栏样式
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #f0f0f0;
                border-bottom: 1px solid #e0e0e0;
                spacing: 2px;
                padding: 1px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 4px 12px;
                font-family: 'Microsoft YaHei';
                font-size: 11px;
                color: #333333;
                min-height: 26px;
            }
            QToolButton:hover {
                background-color: #e8e8e8;
            }
            QToolButton:pressed {
                background-color: #d8d8d8;
            }
        """)
        
        self.addToolBar(toolbar)
        
        # 创建工具栏按钮
        self.btn_file = QAction("文件", self)
        self.btn_file.triggered.connect(self._show_file_menu)
        toolbar.addAction(self.btn_file)
        
        self.btn_model = QAction("打开模型", self)
        self.btn_model.triggered.connect(self.model_load.emit)  # 直接触发信号
        toolbar.addAction(self.btn_model)
        
        self.btn_image = QAction("打开图片", self)
        self.btn_image.triggered.connect(self.image_open.emit)  # 直接触发信号
        toolbar.addAction(self.btn_image)
        
        self.btn_video = QAction("打开视频", self)
        self.btn_video.triggered.connect(self.video_open.emit)  # 直接触发信号
        toolbar.addAction(self.btn_video)
        
        self.btn_camera = QAction("打开摄像头", self)
        self.btn_camera.triggered.connect(self.camera_open.emit)  # 直接触发信号
        toolbar.addAction(self.btn_camera)
        
        self.btn_detect_settings = QAction("检测设置", self)
        self.btn_detect_settings.triggered.connect(self.detect_settings.emit)  # 直接触发信号
        toolbar.addAction(self.btn_detect_settings)
        
        self.btn_help = QAction("帮助", self)
        self.btn_help.triggered.connect(self._show_help_menu)
        toolbar.addAction(self.btn_help)
        
        # 添加 spacer 把按钮推到左侧
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
    
    def _setup_signals(self):
        """设置信号连接 - 简化版本，只暴露子组件信号"""
        # 子组件的信号直接暴露给外部
        # 外部控制器可以这样连接：
        # controller.ui.left_panel_play_pause.connect(handler)
        self.left_panel_play_pause = self.left_panel.play_pause_clicked
        self.iou_changed = self.right_panel.iou_changed
        self.confidence_changed = self.right_panel.confidence_changed
        self.delay_changed = self.right_panel.delay_changed
        self.line_width_changed = self.right_panel.line_width_changed
        self.save_screenshot = self.right_panel.save_screenshot
        self.start_inference = self.right_panel.start_inference
        self.stop_inference = self.right_panel.stop_inference
    
    def _show_file_menu(self):
        """显示文件下拉菜单"""
        file_menu = QMenu(self)
        file_menu.addAction("初始化", self.file_menu_init.emit)
        file_menu.addAction("另存为", self.file_menu_save_as.emit)
        file_menu.addAction("保存", self.file_menu_save.emit)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.file_menu_exit.emit)
        
        # 查找"文件"按钮位置并显示菜单
        for action in self.findChildren(QAction):
            if action.text() == "文件":
                toolbar = self.findChild(QToolBar)
                if toolbar:
                    for act in toolbar.actions():
                        if act.text() == "文件":
                            tool_btn = toolbar.widgetForAction(act)
                            if tool_btn:
                                # 在按钮下方显示菜单
                                pos = tool_btn.mapToGlobal(tool_btn.rect().bottomLeft())
                                file_menu.exec_(pos)
                                return
        
        #  fallback: 在窗口左上角显示
        file_menu.exec_(self.mapToGlobal(self.rect().topLeft()))
    
    def _show_help_menu(self):
        """显示ctober下拉菜单"""
        help_menu = QMenu(self)
        help_menu.addAction("关于", self.help_menu_about.emit)
        help_menu.addAction("使用说明", self.help_menu_manual.emit)
        
        # 查找"帮助"按钮位置并显示菜单
        for action in self.findChildren(QAction):
            if action.text() == "帮助":
                toolbar = self.findChild(QToolBar)
                if toolbar:
                    for act in toolbar.actions():
                        if act.text() == "帮助":
                            tool_btn = toolbar.widgetForAction(act)
                            if tool_btn:
                                # 在按钮下方显示菜单
                                pos = tool_btn.mapToGlobal(tool_btn.rect().bottomLeft())
                                help_menu.exec_(pos)
                                return
        
        # fallback: 在窗口左上角显示
        help_menu.exec_(self.mapToGlobal(self.rect().topLeft()))
    
    # ===== 公共接口方法 =====
    
    def get_left_panel(self) -> LeftDisplayPanel:
        """获取左侧面板实例"""
        return self.left_panel
    
    def get_right_panel(self) -> RightControlPanel:
        """获取右侧面板实例"""
        return self.right_panel
    
    def update_display(self, pixmap):
        """更新显示图像"""
        self.left_panel.set_display_image(pixmap)
    
    def update_progress(self, value, max_value=None):
        """更新进度条"""
        if max_value is not None:
            self.left_panel.set_progress_range(0, max_value)
        self.left_panel.set_progress_value(value)
    
    def update_time_display(self, current_time, total_time):
        """更新时间显示"""
        self.left_panel.set_time_display(current_time, total_time)
    
    def set_play_state(self, is_playing):
        """设置播放状态"""
        self.left_panel.set_play_state(is_playing)
    
    def update_info(self, file_name="", mode=""):
        """更新信息显示"""
        self.left_panel.update_info(file_name, mode)
    
    def update_model_info(self, model_path="", task_type="", input_size="", class_count=""):
        """更新模型信息"""
        self.right_panel.update_model_info(model_path, task_type, input_size, class_count)
    
    def update_statistics(self, detection_count=0, confidence=0.0, inference_time=0, fps=0.0):
        """更新统计信息"""
        self.right_panel.update_statistics(detection_count, confidence, inference_time, fps)
    
    def set_control_state(self, is_running):
        """设置控制状态"""
        self.right_panel.set_control_state(is_running)
    
    def clear_display(self):
        """清空显示"""
        self.left_panel.clear_display()
    
    def get_parameters(self):
        """获取当前参数值"""
        return self.right_panel.get_parameters()
    
    def set_parameters(self, iou_threshold=None, confidence_threshold=None, delay_ms=None, line_width=None):
        """设置参数值"""
        self.right_panel.set_parameters(iou_threshold, confidence_threshold, delay_ms, line_width)
    
    def set_controls_enabled(self, enabled):
        """启用/禁用控制面板"""
        self.left_panel.set_controls_enabled(enabled)
    
    def show_confirm_exit_dialog(self) -> bool:
        """显示退出确认对话框
        
        Returns:
            bool: 用户是否确认退出
        """
        from PySide6.QtWidgets import QMessageBox
        result = QMessageBox.question(
            self,
            "确认退出",
            "确定要退出程序吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        return result == QMessageBox.Yes
    
    def show_about_dialog(self):
        """显示关于对话框"""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "关于",
            "YOLO多功能检测系统\n"  
            "版本 1.0.0\n"                
            "作者: YOLO团队\n"            
            "用于目标检测和分析的多功能工具"
        )
    
    def show_help_manual_dialog(self):
        """显示使用说明对话框"""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "使用说明",
            "使用步骤:\n"                                  
            "1. 首先加载YOLO模型\n"                        
            "2. 打开图片、视频文件或摄像头\n"             
            "3. 调整推理参数（置信度、IOU等）\n"          
            "4. 点击开始按钮进行目标检测\n"                
            "5. 可随时暂停/继续或停止检测\n"               
            "6. 使用保存按钮保存当前截图"
        )
    
    def show_detect_settings_dialog(self):
        """显示检测设置对话框"""
        # 这里只是一个简单的实现，后续可以根据需要扩展
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "检测设置",
            "检测设置功能正在开发中..."
        )


# ============================================================================
# 测试使用
# 功能：单独运行时显示UI界面，用于测试布局和交互效果
# ============================================================================
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    # 创建应用实例
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格，跨平台一致性更好
    
    # 创建并显示主窗口
    window = YOLOMainWindowUI()
    window.setMinimumSize(1000, 650)  # 设置最小窗口大小
    window.show()
    
    # 测试信息输出
    print("=" * 60)
    print("YOLO GUI界面测试")
    print("=" * 60)
    
    left_panel = window.get_left_panel()
    
    # 测试专用: 在单独运行UI时把显示区域变成白底并显示文件名，便于视觉校验
    try:
        # 把显示标签背景设为白色并显示测试文字
        left_panel.display_label.setStyleSheet("background-color: white; color: black;")
        left_panel.display_label.setText("测试显示窗口")

        # 显示测试文件名（作为叠加标签）并使用与容器内边距一致的位置
        left_panel.update_info(file_name="测试文件名.jpg", mode="image")
        if left_panel.filename_label.isVisible():
            left_panel.filename_label.move(12, 12)

        # 确保图片模式下控制栏被禁用
        left_panel.set_controls_enabled(False)
    except Exception:
        pass

    # 启动应用事件循环
    sys.exit(app.exec())