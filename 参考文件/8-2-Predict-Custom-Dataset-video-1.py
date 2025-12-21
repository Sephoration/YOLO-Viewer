# -*- coding: utf-8 -*-
"""
脚本名称：8-2-Predict-Custom-Dataset-video-1.py
功能描述：使用自定义训练的YOLOv11-pose模型进行视频姿态检测，支持实时播放和交互控制
使用方法：修改input_video为目标视频路径，直接运行脚本
交互控制：q键或ESC键退出，空格键暂停/继续
依赖库：cv2, torch, time, ctypes, PIL, ultralytics
"""
import cv2
import torch
import time
import ctypes

from PIL import Image
from ultralytics import YOLO

# 设备选择：有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 载入自定义训练的模型
model = YOLO('Triangle_215_yolo11s_pretrain.pt')
model = model.to(device)

# 可视化配置
# 框(rectangle)：颜色、线宽
bbox_color = (150, 0, 0)             # 框的 BGR 颜色
bbox_thickness = 6                   # 框的线宽

# 框类别文字配置
bbox_labelstr = {
    'font_size':4,         # 字体大小
    'font_thickness':10,   # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-80,        # Y 方向，文字偏移距离，向下为正
}

# 关键点 BGR 配色方案
kpt_color_map = {
    0:{'name':'angle_30', 'color':[255, 0, 0], 'radius':40},      # 30度角点
    1:{'name':'angle_60', 'color':[0, 255, 0], 'radius':40},      # 60度角点
    2:{'name':'angle_90', 'color':[0, 0, 255], 'radius':40},      # 90度角点
}

# 关键点类别文字配置
kpt_labelstr = {
    'font_size':4,             # 字体大小
    'font_thickness':10,       # 字体粗细
    'offset_x':30,             # X 方向，文字偏移距离，向右为正
    'offset_y':120,            # Y 方向，文字偏移距离，向下为正
}

# 骨架连接 BGR 配色方案
skeleton_map = [
    {'srt_kpt_id':0, 'dst_kpt_id':1, 'color':[196, 75, 255], 'thickness':12},        # 30度角点-60度角点
    {'srt_kpt_id':0, 'dst_kpt_id':2, 'color':[180, 187, 28], 'thickness':12},        # 30度角点-90度角点
    {'srt_kpt_id':1, 'dst_kpt_id':2, 'color':[47,255, 173], 'thickness':12},         # 60度角点-90度角点
]

# 单帧处理函数
def process_frame(img_bgr):
    """
    处理单帧图像，进行姿态检测和可视化
    参数：img_bgr - BGR格式的输入图像
    返回：处理后的BGR图像
    """
    start_time = time.time()

    # 推理预测
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 pil
    results = model(img_pil, save=False, verbose=False)

    # 目标检测框数量
    num_bbox = len(results[0].boxes.cls)

    # 转成整数的 numpy array
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('int32')
    bboxes_keypoints = results[0].keypoints.xy.cpu().numpy().astype('int32')
    
    # 遍历每个检测框，绘制框与关键点
    for idx in range(num_bbox):  # 遍历每个框
        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]

        # 获取框的预测类别(对于关键点检测，只有一个类别)
        bbox_label = results[0].names[0]

        # 绘制边界框
        img_bgr = cv2.rectangle(img_bgr, 
                              (bbox_xyxy[0], bbox_xyxy[1]), 
                              (bbox_xyxy[2], bbox_xyxy[3]), 
                              bbox_color, bbox_thickness)

        # 写入框类别文字
        img_bgr = cv2.putText(img_bgr, bbox_label,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], 
                               bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              bbox_labelstr['font_size'], 
                              bbox_color,
                              bbox_labelstr['font_thickness'])

        # 获取该框所有关键点坐标
        bbox_keypoints = bboxes_keypoints[idx]  
        
        # 绘制骨架连接
        for skeleton in skeleton_map:
            # 获取起始点坐标
            srt_kpt_id = skeleton['srt_kpt_id']        
            srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
            srt_kpt_y = bbox_keypoints[srt_kpt_id][1]

            # 获取终止点坐标
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = bbox_keypoints[dst_kpt_id][0]
            dst_kpt_y = bbox_keypoints[dst_kpt_id][1]

            # 获取骨架连接颜色和线宽
            skeleton_color = skeleton['color']
            skeleton_thickness = skeleton['thickness']

            # 绘制骨架连接线
            img_bgr = cv2.line(img_bgr, 
                             (srt_kpt_x, srt_kpt_y), 
                             (dst_kpt_x, dst_kpt_y), 
                             color=skeleton_color,
                             thickness=skeleton_thickness)
        
        # 绘制关键点
        for kpt_id in kpt_color_map:
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]

            # 绘制关键点圆（-1为填充）
            img_bgr = cv2.circle(img_bgr, 
                              (kpt_x, kpt_y), 
                              kpt_radius, 
                              kpt_color, -1)

            # 写入关键点类别文字
            kpt_label = str(kpt_id)  # 写关键点类别 ID(二选一)
            # kpt_label = str(kpt_color_map[kpt_id]['name']) # 写关键点类别名称(二选一)

            img_bgr = cv2.putText(img_bgr, kpt_label,
                                  (kpt_x + kpt_labelstr['offset_x'], 
                                   kpt_y + kpt_labelstr['offset_y']),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  kpt_labelstr['font_size'], 
                                  kpt_color,
                                  kpt_labelstr['font_thickness'])
    
    # 计算并显示FPS
    end_time = time.time()
    FPS = 1 / (end_time - start_time)
    FPS_string = 'FPS  {:.2f}'.format(FPS)  # 显示FPS字符串
    img_bgr = cv2.putText(img_bgr, FPS_string, 
                         (25, 60), 
                         cv2.FONT_HERSHEY_SIMPLEX,
                         1.25, (255, 0, 255), 2)

    return img_bgr

# 主函数
if __name__ == '__main__':
    input_video = 'triangle_7.mp4'  # 输入视频路径

    # 打开视频文件
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{input_video}")

    # 设置视频缓冲区大小
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 创建显示窗口
    win_name = "Keypoint Detection (q: quit, SPACE: pause)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    # 先读一帧获取视频尺寸
    success, frame_bgr = cap.read()
    if not success:
        raise RuntimeError("无法读取第一帧")
    
    h, w = frame_bgr.shape[:2]
    cv2.resizeWindow(win_name, w, h)  # 设置窗口大小
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到第0帧
    
    # 计算屏幕中心并移动显示窗口到屏幕中央
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    x = (screen_w - w) // 2
    y = (screen_h - h) // 2
    cv2.moveWindow(win_name, x, y)
    
    paused = False  # 暂停状态标记
    
    # 主循环
    while True:
        if not paused:
            success, frame_bgr = cap.read()

            # 播放到结尾：回到第 0 帧继续循环播放
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 处理当前帧
            out_bgr = process_frame(frame_bgr)

            # 显示处理后的帧
            cv2.imshow(win_name, out_bgr)

        # 键盘交互
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 键或 ESC 键退出
            break
        elif key == ord(' '):  # 空格键暂停/继续
            paused = not paused
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()



