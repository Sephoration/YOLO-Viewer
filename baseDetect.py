import cv2

class baseDetect(object):
    """基础 Tracker 配置类，放一些通用参数与共用方法。"""

    # ---------------------------------------------------------
    # 2. 目标追踪（Tracking）
    # ---------------------------------------------------------

    def __init__(self):
        # 推理图像尺寸
        self.img_size = 640
        # 置信度阈值
        self.conf = 0.25
        # NMS 的 IoU 阈值
        self.iou = 0.70

    def draw_bboxes(self, im, pred_boxes):
        """
        在图像上画框与标签（OOP 版，作为基类方法）。

        参数
        ----
        im : np.ndarray (BGR)
            原始图像（OpenCV 格式）
        pred_boxes : list of tuple
            每个元素为：
            (x1, y1, x2, y2, lbl, confidence, track_id or None)

        返回
        ----
        im : np.ndarray (BGR)
            画好框的图像
        """
        # 这里用局部 color_map，将来如果有需要可以改成成员变量
        color_map = {
            'person': (0, 255, 0),
            'car': (255, 0, 0),
            'bus': (0, 0, 255),
            'truck': (0, 255, 255),
        }

        for (x1, y1, x2, y2, lbl, confidence, track_id) in pred_boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 类别颜色：若不在 color_map 中，就给默认白色
            color = color_map.get(lbl, (255, 255, 255))

            # 画框
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

            # 文本内容：侦测模式没有 ID，追踪模式有 ID
            if track_id is not None:
                label_text = f'{lbl} ID:{track_id} {confidence:.2f}'
            else:
                label_text = f'{lbl} {confidence:.2f}'

            # 字型与大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            # 计算文字尺寸，方便画背景
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )

            # 背景框位置（在框的左上角上方）
            text_x, text_y = x1, max(0, y1 - 5)
            box_coords = (
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
            )

            # 画实心矩形作为文字背景
            cv2.rectangle(im, box_coords[0], box_coords[1], color, thickness=-1)

            # 再画文字（黑字）
            cv2.putText(
                im,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        return im