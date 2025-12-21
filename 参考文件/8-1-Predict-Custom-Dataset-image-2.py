# -*- coding: utf-8 -*-
"""
脚本名称：8-1-Predict-Custom-Dataset-image-2.py
功能描述：使用自定义训练的YOLOv11-pose模型进行图像姿态预测，带有自定义可视化设置
使用方法：修改model路径和img_path为目标模型和图像路径，直接运行脚本
依赖库：cv2, torch, matplotlib, ultralytics
"""
import cv2
import torch
from matplotlib import pyplot as plt
from ultralytics import YOLO

# 输入图像路径
# img_path = './images/'
img_path = './images/triangle_4.jpg'

# 模型路径
model = YOLO('Triangle_215_yolo11s_pretrain.pt')

# 推理预测
results = model(img_path)

# 可视化设置
# 框（rectangle）：颜色、线宽
bbox_color = (150, 0, 0)            # 框的 BGR 颜色
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
    0:{'name':'angle_30', 'color':[255, 0, 0], 'radius':40},          # 30度角点
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
    {'srt_kpt_id':0, 'dst_kpt_id':2, 'color':[180, 187, 28], 'thickness':12},     # 30度角点-90度角点
    {'srt_kpt_id':1, 'dst_kpt_id':2, 'color':[47,255, 173], 'thickness':12},      # 60度角点-90度角点
]

# 导入 BGR 格式的图像
img_bgr = cv2.imread(img_path)

# 目标检测框数量
num_bbox = len(results[0].boxes.cls)

# 转成整数的 numpy array，获取边界框坐标
bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('int32')

# 将关键点转为 numpy array，获取关键点坐标
bboxes_keypoints = results[0].keypoints.xy.cpu().numpy().astype('int32')

# 遍历每个检测框
for idx in range(num_bbox):  
    # 获取该框坐标
    bbox_xyxy = bboxes_xyxy[idx]
    # 获取框的预测类别（对于关键点检测，只有一个类别）
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
    
    # 获取该框的所有关键点坐标
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
        kpt_label = str(kpt_id)  # 写关键点类别 ID（二选一）
        # kpt_label = str(kpt_color_map[kpt_id]['name']) # 写关键点类别名称（二选一）
        
        img_bgr = cv2.putText(img_bgr, kpt_label, 
                             (kpt_x + kpt_labelstr['offset_x'], 
                              kpt_y + kpt_labelstr['offset_y']),
                             cv2.FONT_HERSHEY_SIMPLEX, 
                             kpt_labelstr['font_size'], 
                             kpt_color,
                             kpt_labelstr['font_thickness'])

# 显示结果图像
plt.imshow(img_bgr[:,:,::-1])  # BGR转RGB显示
plt.show()

# 保存结果图像
cv2.imwrite('outputs/7-1-output.jpg', img_bgr)
