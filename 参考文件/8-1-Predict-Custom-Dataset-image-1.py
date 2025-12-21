# -*- coding: utf-8 -*-
"""
脚本名称：8-1-Predict-Custom-Dataset-image-1.py
功能描述：使用自定义训练的YOLOv11-pose模型进行图像姿态预测，适用于三角形姿态检测
使用方法：修改model路径和img_path为目标模型和图像路径，直接运行脚本
依赖库：ultralytics, matplotlib
"""
from ultralytics import YOLO
from matplotlib import pyplot as plt
# 模型路径
model = YOLO('Triangle_215_yolo11s_pretrain.pt')
# 输入图像路径
img_path = './images/triangle_4.jpg'
# 推理预测
results = model(img_path,verbose=False)
# 绘制检测结果
# Ultralytics 的 plot() 返回的是适合 RGB 显示的图像
rgb_image = results[0].plot()    # 这里其实可以把名字改成 image 或 rgb_image

# 用 Matplotlib 显示（plt.imshow 期望 RGB）
plt.imshow(rgb_image)
plt.axis('off')  # 去掉坐标轴
plt.show()