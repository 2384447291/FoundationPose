#!/usr/bin/env python3
"""
跟踪控制和位姿可视化程序

这个程序可以：
1. 发送开启跟踪命令（包含 object_prompt 和 object_name）
2. 发送停止跟踪命令
3. 实时接收并可视化位姿数据（包含坐标轴显示）

使用方法：
python tracking_controller_example.py
"""

import time
import logging
import numpy as np
import cv2
import threading
from shm_lib.pubsub_manager import PubSubManager
from shm_lib.shared_memory_util import encode_text_prompt, decode_text_prompt


# 配置
port = 10000
CONTROL_TOPIC = "tracking_control"
POSE_TOPIC = "tracking_poses"

OBJECT_PROMPT = "a green bittermelon"
OBJECT_NAME = "bittermelon"

# 全局变量用于位姿数据
latest_pose = None
pose_count = 0
latest_object_name = ""
pose_lock = threading.Lock()

# 可视化配置
VISUALIZATION_SIZE = (800, 600)
AXIS_LENGTH = 0.1  # 坐标轴长度
ORIGIN_OFFSET = np.array([0.4, 0.3, 1.0])  # 显示原点偏移


def draw_coordinate_axes(img, pose_matrix, camera_matrix, axis_length=0.1):
    """在图像上绘制坐标轴。"""
    # 定义坐标轴端点（相对于物体坐标系）
    axes_points = np.array([
        [0, 0, 0, 1],                    # 原点
        [axis_length, 0, 0, 1],          # X轴 (红色)
        [0, axis_length, 0, 1],          # Y轴 (绿色)  
        [0, 0, axis_length, 1]           # Z轴 (蓝色)
    ]).T
    
    # 变换到相机坐标系
    cam_points = pose_matrix @ axes_points
    
    # 投影到图像平面
    cam_points_3d = cam_points[:3, :] / cam_points[3, :]
    
    # 使用相机内参投影
    img_points = camera_matrix @ cam_points_3d
    img_points = img_points[:2, :] / img_points[2, :]
    
    # 转换为整数像素坐标
    img_points = img_points.astype(int)
    
    # 检查点是否在图像范围内
    h, w = img.shape[:2]
    valid_points = []
    for i in range(img_points.shape[1]):
        x, y = img_points[:, i]
        if 0 <= x < w and 0 <= y < h:
            valid_points.append((x, y))
        else:
            valid_points.append(None)
    
    # 绘制坐标轴
    if valid_points[0] is not None:  # 原点
        origin = valid_points[0]
        
        # X轴 (红色)
        if valid_points[1] is not None:
            cv2.arrowedLine(img, origin, valid_points[1], (0, 0, 255), 3, tipLength=0.3)
            cv2.putText(img, 'X', (valid_points[1][0]+5, valid_points[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Y轴 (绿色)
        if valid_points[2] is not None:
            cv2.arrowedLine(img, origin, valid_points[2], (0, 255, 0), 3, tipLength=0.3)
            cv2.putText(img, 'Y', (valid_points[2][0]+5, valid_points[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Z轴 (蓝色)
        if valid_points[3] is not None:
            cv2.arrowedLine(img, origin, valid_points[3], (255, 0, 0), 3, tipLength=0.3)
            cv2.putText(img, 'Z', (valid_points[3][0]+5, valid_points[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 在原点绘制小圆点
        cv2.circle(img, origin, 5, (255, 255, 255), -1)
    
    return img


def update_pose_visualization():
    """定时更新位姿可视化显示。"""
    global latest_pose, pose_count, latest_object_name
    
    # 创建虚拟相机内参用于可视化
    fx, fy = 500, 500
    cx, cy = VISUALIZATION_SIZE[0]//2, VISUALIZATION_SIZE[1]//2
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy], 
        [0, 0, 1]
    ])
    
    while True:
        with pose_lock:
            current_pose = latest_pose.copy() if latest_pose is not None else None
            current_count = pose_count
            current_name = latest_object_name
        
        if current_pose is not None:
            # 创建可视化图像
            vis_img = np.zeros((VISUALIZATION_SIZE[1], VISUALIZATION_SIZE[0], 3), dtype=np.uint8)
            
            # 调整位姿以便可视化（添加偏移让坐标轴出现在图像中心）
            display_pose = current_pose.copy()
            display_pose[:3, 3] += ORIGIN_OFFSET
            
            # 绘制坐标轴
            vis_img = draw_coordinate_axes(vis_img, display_pose, camera_matrix, AXIS_LENGTH)
            
            # 添加文本信息
            cv2.putText(vis_img, f"Object: {current_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, f"Frame: {current_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示位置信息
            pos = current_pose[:3, 3]
            cv2.putText(vis_img, f"Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示旋转信息（欧拉角）
            rotation_matrix = current_pose[:3, :3]
            def rotation_matrix_to_euler(R):
                sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
                singular = sy < 1e-6
                if not singular:
                    x = np.arctan2(R[2,1], R[2,2])
                    y = np.arctan2(-R[2,0], sy)
                    z = np.arctan2(R[1,0], R[0,0])
                else:
                    x = np.arctan2(-R[1,2], R[1,1])
                    y = np.arctan2(-R[2,0], sy)
                    z = 0
                return np.array([x, y, z]) * 180.0 / np.pi
            
            euler = rotation_matrix_to_euler(rotation_matrix)
            cv2.putText(vis_img, f"Rot: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示图像
            cv2.imshow('Pose Visualization', vis_img)
            cv2.waitKey(1)
        
        time.sleep(0.033)  # ~30 FPS


def get_latest_pose_data(pubsub):
    """定时获取最新的位姿数据（非中断方式）。"""
    global latest_pose, pose_count, latest_object_name
    
    try:
        data = pubsub.get_latest_data(POSE_TOPIC)
        if data is not None:
            with pose_lock:
                latest_pose = data['pose_matrix']
                pose_count = data['frame_idx']
                latest_object_name = decode_text_prompt(data['object_name'])
            return True
    except Exception as e:
        logging.debug(f"No pose data available: {e}")
    
    return False


def rotation_matrix_to_quaternion(R):
    """将3x3旋转矩阵转换为四元数 (w, x, y, z)。"""
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return np.array([w, x, y, z], dtype=np.float64)


def quaternion_to_rotation_matrix(q):
    """将四元数 (w, x, y, z) 转换为3x3旋转矩阵。"""
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    R = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)]
    ], dtype=np.float64)
    return R


def get_final_averaged_pose_from_buffer(pubsub, expected_samples=50, wait_timeout=10.0):
    """
    从广播buffer中读取50个位姿进行平均。
    
    当tracking_client停止时，它会发布50次最终位姿到广播buffer中。
    这个函数等待并读取这50个位姿，然后计算平均值。
    """
    logging.info("Waiting for final poses to be published to buffer...")
    
    translations = []
    quaternions = []
    start_time = time.time()
    last_frame_idx = -1
    
    # 先发送停止命令
    send_stop_command(pubsub)
    
    # 等待收集50个位姿样本
    while len(translations) < expected_samples and (time.time() - start_time) < wait_timeout:
        try:
            data = pubsub.get_latest_data(POSE_TOPIC)
            if data is not None:
                frame_idx = int(data.get('frame_idx', 0))
                
                # 确保是新的数据（基于frame_idx）
                if frame_idx > last_frame_idx:
                    last_frame_idx = frame_idx
                    pose = data['pose_matrix']
                    R = pose[:3, :3].astype(np.float64)
                    t = pose[:3, 3].astype(np.float64)
                    translations.append(t)
                    
                    q = rotation_matrix_to_quaternion(R)
                    # 统一四元数符号避免平均抵消
                    if len(quaternions) > 0 and np.dot(q, quaternions[-1]) < 0:
                        q = -q
                    quaternions.append(q)
                    
                    if len(translations) % 10 == 0:
                        logging.info(f"Collected {len(translations)}/{expected_samples} poses from buffer")
            
            time.sleep(0.01)  # 短暂等待
            
        except Exception as e:
            logging.error(f"Error reading from pose buffer: {e}")
            time.sleep(0.1)
    
    if len(translations) == 0:
        logging.warning("No poses collected from buffer")
        return None
    
    logging.info(f"Collected {len(translations)} poses from buffer. Computing average...")
    
    # 平移直接平均
    t_avg = np.mean(np.stack(translations, axis=0), axis=0)
    
    # 四元数归一化后平均
    Q = np.stack(quaternions, axis=0)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    q_avg = np.mean(Q, axis=0)
    q_avg = q_avg / (np.linalg.norm(q_avg) + 1e-12)
    R_avg = quaternion_to_rotation_matrix(q_avg)
    
    T_avg = np.eye(4, dtype=np.float32)
    T_avg[:3, :3] = R_avg.astype(np.float32)
    T_avg[:3, 3] = t_avg.astype(np.float32)
    
    logging.info(f"Final averaged pose computed from {len(translations)} samples")
    return T_avg


def request_average_pose(pubsub, samples=50, per_sample_timeout=0.5):
    """
    请求当前位姿的平均值。
    
    新逻辑：停止跟踪并从广播buffer中读取50个最终位姿进行平均。
    """
    return get_final_averaged_pose_from_buffer(pubsub, samples, wait_timeout=samples * per_sample_timeout)


def send_start_command(pubsub, object_prompt, object_name):
    """发送开启跟踪命令。"""
    command_data = {
        'command': encode_text_prompt('start'),
        'object_prompt': encode_text_prompt(object_prompt),
        'object_name': encode_text_prompt(object_name)
    }
    
    success = pubsub.publish(CONTROL_TOPIC, command_data)
    if success:
        logging.info(f"Sent START command - Object: {object_name}, Prompt: '{object_prompt}'")
    else:
        logging.error("Failed to send START command")
    
    return success


def send_stop_command(pubsub):
    """发送停止跟踪命令。"""
    command_data = {
        'command': encode_text_prompt('stop'),
        'object_prompt': np.zeros(256, dtype=np.uint8),
        'object_name': encode_text_prompt('')
    }
    
    success = pubsub.publish(CONTROL_TOPIC, command_data)
    if success:
        logging.info("Sent STOP command")
    else:
        logging.error("Failed to send STOP command")
    
    return success


def main():
    """主函数：演示外部程序如何控制跟踪客户端。"""
    global latest_pose, pose_count, latest_object_name
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
    
    # 初始化PubSub管理器
    pubsub = PubSubManager(port=port, authkey=b'foundationpose')
    pubsub.start(role='both')
    
    # 配置topics
    topics_config = {
        CONTROL_TOPIC: {
            'examples': {
                'command': encode_text_prompt('start'),
                'object_prompt': np.zeros(256, dtype=np.uint8),
                'object_name': encode_text_prompt('default_object')
            },
            'buffer_size': 5,
            'mode': 'consumer'  # 控制命令使用消费者模式
        },
        POSE_TOPIC: {
            'examples': {
                'pose_matrix': np.eye(4, dtype=np.float32),
                'timestamp': np.float64(0.0),
                'frame_idx': np.int32(0),
                'object_name': encode_text_prompt('default_object')
            },
            'buffer_size': 50,
            'mode': 'broadcast'  # 位姿数据使用广播模式
        }
    }
    
    pubsub.setup_subscriber(topics_config)
    
    # Create topics for publishing
    for topic_name, config in topics_config.items():
        pubsub.create_topic(
            topic_name, 
            config['examples'], 
            config['buffer_size'],
            config['mode']
        )
    
    # 启动位姿可视化线程
    vis_thread = threading.Thread(target=update_pose_visualization, daemon=True)
    vis_thread.start()
    
    try:
        print("\n=== 跟踪控制和位姿可视化程序 ===")
        print("命令:")
        print("  's' - 开始跟踪 (默认: bittermelon)")
        print("  'k' - 停止当前跟踪 / 请求当前位姿")
        print("(Ctrl+C 退出)")
        print("=" * 40)
        
        # 主循环：处理用户命令和更新位姿数据
        while True:
            # 定时获取位姿数据
            get_latest_pose_data(pubsub)
            
            # 检查用户输入（非阻塞）
            try:
                import select
                import sys
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    command = input("\n请输入命令 (s=开始, k=停止): ").strip().lower()
                    if command == 's':
                        # 开始跟踪 (默认对象)
                        object_prompt = OBJECT_PROMPT
                        object_name = OBJECT_NAME
                        send_start_command(pubsub, object_prompt, object_name)
                    elif command == 'k':
                        # 请求20次位姿并计算平均，然后停止跟踪
                        avg_pose = request_average_pose(pubsub, samples=20, per_sample_timeout=0.5)
                        if avg_pose is not None:
                            with pose_lock:
                                latest_pose = avg_pose
                        send_stop_command(pubsub)
                    else:
                        print("无效命令，仅支持 s/k")
            except ImportError:
                # 无 select 模块时的简化处理（阻塞式输入）
                command = input("请输入命令 (s=开始, k=停止): ").strip().lower()
                if command == 's':
                    object_prompt = OBJECT_PROMPT
                    object_name = OBJECT_NAME
                    send_start_command(pubsub, object_prompt, object_name)
                elif command == 'k':
                    avg_pose = request_average_pose(pubsub, samples=20, per_sample_timeout=0.5)
                    if avg_pose is not None:
                        with pose_lock:
                            latest_pose = avg_pose
                    send_stop_command(pubsub)
                else:
                    print("无效命令，仅支持 s/k")
            
            time.sleep(0.1)  # 100ms循环间隔
    
    except KeyboardInterrupt:
        logging.info("程序被用户中断")
    except Exception as e:
        logging.error(f"程序出现错误: {e}")
    finally:
        cv2.destroyAllWindows()
        pubsub.stop(role='both')
        logging.info("跟踪控制器已关闭")


if __name__ == '__main__':
    main()