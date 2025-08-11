import pyrealsense2 as rs
import numpy as np
import cv2


class RealSense:
    """
    RealSense相机控制类，用于初始化、获取图像和释放资源。
    """

    def __init__(self, width=640, height=480, fps=30):
        """
        初始化RealSense相机
        :param width: 图像宽度
        :param height: 图像高度
        :param fps: 帧率
        """
        print("正在初始化RealSense相机...")
        self.pipeline = rs.pipeline()
        config = rs.config()

        # 配置彩色和深度流
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # 启动管道
        profile = self.pipeline.start(config)

        # 获取深度传感器的深度比例
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"深度比例系数: {self.depth_scale}")

        # 创建对齐对象（深度对齐到彩色）
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # 获取相机内参
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        self.intrinsics = {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
        }
        self.camera_params = self.intrinsics.copy()
        self.camera_params['depth_scale'] = self.depth_scale
        print(f"相机内参: {self.camera_params}")
        print("RealSense相机初始化完成。")

    def get(self):
        """
        获取一对对齐的彩色和深度帧。

        :return: (color_image, depth_image) 元组。如果获取失败则返回 (None, None)。
        """
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_image
        except RuntimeError as e:
            print(f"无法获取帧: {e}")
            return None, None

    def release(self):
        """
        停止管道并释放相机资源。
        """
        print("正在停止RealSense相机...")
        self.pipeline.stop()
        print("RealSense相机已停止。")


if __name__ == "__main__":
    rs_cam = None
    try:
        rs_cam = RealSense()

        cv2.namedWindow('RealSense Live', cv2.WINDOW_AUTOSIZE)

        while True:
            color_image, depth_image = rs_cam.get()

            if color_image is not None and depth_image is not None:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                images = np.hstack((color_image, depth_colormap))
                cv2.imshow('RealSense Live', images)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # 按 'q' 或 'ESC' 键退出
                break
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        if rs_cam:
            rs_cam.release()
        cv2.destroyAllWindows()
        print("程序已退出。")
