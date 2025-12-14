"""
程序入口点
创建Qt应用实例、初始化UI和逻辑、启动事件循环
"""

import sys
from PySide6.QtWidgets import QApplication

# 导入UI和逻辑模块
from window_ui import YOLOMainWindowUI
from logic_controller import YOLOMainController

# ==================== 全局常量 ====================
APP_NAME = "YOLO多模型检测系统"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Sephoration"
APP_INFO = f"{APP_NAME} - {APP_VERSION} - {APP_AUTHOR}"

# ==================== 主程序入口 ====================
def main():
    """主函数：启动应用程序"""
    try:
        # 创建应用实例
        app = QApplication(sys.argv)
        app.setApplicationName(APP_INFO)
        app.setApplicationVersion(APP_VERSION)

        app.setStyle("Fusion")

        # 创建UI窗口
        print(f"{APP_INFO} 启动中...")
        ui_window = YOLOMainWindowUI()
        
        # 创建逻辑控制器并连接UI
        controller = YOLOMainController(ui_window)
        
        # 显示窗口
        ui_window.show()
        print("启动成功！")

        # 启动事件循环
        return app.exec()

    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())