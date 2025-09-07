import os

# 等价于：
# export PATH=$CONDA_PREFIX/bin:$PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
_conda_prefix = os.environ.get('CONDA_PREFIX', '')
_orig_path = os.environ.get('PATH', '')
os.environ['PATH'] = f"{_conda_prefix}/bin:{_orig_path}" if _orig_path else f"{_conda_prefix}/bin"
_orig_ld = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = f"{_orig_ld}:{_conda_prefix}/lib" if _orig_ld else f"{_conda_prefix}/lib"

    

import cv2
import numpy as np
import trimesh
import imageio
import logging
import time
import threading
import nvdiffrast.torch as dr

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import *

# 导入相机模组和共享内存模组
from realsense.realsense import RealSense
from shm_lib.pubsub_manager import PubSubManager
from shm_lib.shared_memory_util import encode_text_prompt, decode_text_prompt


# 默认配置
port = 10000

# 单位转换配置（保持与 run_realtime_tracking 一致：不进行单位缩放）

# 共享内存的topic定义
REQUEST_TOPIC = "segmentation_requests"
MASK_TOPIC = "segmentation_masks"
CONTROL_TOPIC = "tracking_control"      # 新增：接收开启/停止命令
POSE_TOPIC = "tracking_poses"           # 新增：发送位姿数据

# Global variables for mask receiving
received_mask = None
mask_received_event = threading.Event()

# Global variables for tracking control
tracking_active = False
current_object_prompt = None
current_object_name = None
control_received_event = threading.Event()
_PUBSUB_REF = None
latest_pose = None
pose_lock = threading.Lock()


def handle_segmentation_mask(data):
    """Handle received segmentation mask."""
    global received_mask, mask_received_event
    
    mask = data['mask']
    received_mask = mask
    mask_received_event.set()  # Signal that mask has been received
    logging.info(f"Received segmentation mask with {np.sum(mask)} positive pixels")
def pubsub_global_ref():
    return _PUBSUB_REF



def handle_tracking_control(data):
    """Handle tracking control commands from external program."""
    global tracking_active, current_object_prompt, current_object_name, control_received_event
    
    command = decode_text_prompt(data.get('command', ''))
    
    if command == 'start':
        # 从外部程序接收开启命令和对象信息
        object_prompt = decode_text_prompt(data.get('object_prompt', ''))
        object_name = decode_text_prompt(data.get('object_name', ''))
        
        if object_prompt and object_name:
            current_object_prompt = object_prompt
            current_object_name = object_name
            tracking_active = True
            control_received_event.set()
            logging.info(f"Received START command - Object: {object_name}, Prompt: '{object_prompt}'")
        else:
            logging.error("START command received but missing object_prompt or object_name")
            
    elif command == 'stop':
        # 接收停止命令
        tracking_active = False
        control_received_event.set()
        logging.info("Received STOP command - Stopping tracking")
        
    elif command == 'get_pose':
        # 按需返回当前位姿（单次回应）
        with pose_lock:
            pose_to_send = None if latest_pose is None else latest_pose.copy()
            object_name = current_object_name
        if pose_to_send is not None and object_name is not None:
            pose_data = {
                'pose_matrix': pose_to_send.astype(np.float32),
                'timestamp': time.time(),
                'frame_idx': np.int32(0),
                'object_name': encode_text_prompt(object_name)
            }
            try:
                PubSubManager_instance = pubsub_global_ref()
                if PubSubManager_instance is not None:
                    PubSubManager_instance.publish(POSE_TOPIC, pose_data)
            except Exception as e:
                logging.error(f"Failed to publish on-demand pose: {e}")
        else:
            logging.debug("No pose available to respond to get_pose")
        
    else:
        logging.warning(f"Unknown control command: {command}")


def capture_and_process_frame(rs_cam):
    """Capture a new frame from camera and process depth."""
    color, depth = None, None
    while color is None:
        color, depth = rs_cam.get()
        time.sleep(0.1)
    
    # Process depth image to be consistent with datareader
    if depth is not None:
        depth = depth.astype(np.float64) * rs_cam.depth_scale
        depth[depth < 0.001] = 0
    
    return color, depth


def send_segmentation_request(pubsub, color, object_prompt):
    """Send segmentation request and wait for response."""
    global mask_received_event
    
    # Reset the event before sending request
    mask_received_event.clear()
    
    # Send segmentation request
    req_data = {
        'rgb': color,
        'prompt': encode_text_prompt(object_prompt)
    }
    success = pubsub.publish(REQUEST_TOPIC, req_data)
    if success:
        logging.info(f"Sent segmentation request with prompt: '{object_prompt}'")
    else:
        logging.error("Failed to send segmentation request")
        raise RuntimeError("Failed to send segmentation request")
    
    # Wait for segmentation response
    logging.info("Waiting for segmentation mask from client...")
    if not mask_received_event.wait(timeout=30):
        logging.error("Timeout: Did not receive segmentation mask from client within 30 seconds.")
        raise TimeoutError("No segmentation mask received within 30 seconds")
    
    return received_mask


def get_validated_segmentation_mask(pubsub, rs_cam, initial_color, object_prompt):
    """Get segmentation mask with user validation loop."""
    color = initial_color
    
    # Send initial segmentation request
    mask = send_segmentation_request(pubsub, color, object_prompt)
    
    # Validation loop
    while True:
        # Visualize the received mask
        mask_vis = (mask.astype(np.uint8) * 255)
        mask_vis_color = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        combined_vis = np.hstack((color, mask_vis_color))
        cv2.imshow('Segmentation Result Validation', combined_vis)
        logging.info("Segmentation result displayed. Press SPACE to accept, 'q' to retry with new frame, or ESC to exit...")
        
        key = cv2.waitKey(0)
        if key == ord(' '):  # Space key - accept the mask
            logging.info("Segmentation mask accepted by user.")
            cv2.destroyWindow('Segmentation Result Validation')
            return mask
        elif key == ord('q'):  # 'q' key - retry segmentation
            logging.info("User requested to retry segmentation with new frame.")
            cv2.destroyWindow('Segmentation Result Validation')
            
            # Get new frame and request new segmentation
            color, _ = capture_and_process_frame(rs_cam)
            mask = send_segmentation_request(pubsub, color, object_prompt)
            
        elif key == 27:  # ESC key - exit
            logging.info("User pressed ESC, exiting...")
            cv2.destroyWindow('Segmentation Result Validation')
            raise KeyboardInterrupt("User cancelled segmentation")
        else:
            logging.info("Invalid key pressed. Press SPACE to accept, 'q' to retry, or ESC to exit.")


def run_tracking_session(pubsub, rs_cam, object_prompt, object_name):
    """Run a complete tracking session for the given object."""
    global tracking_active
    
    # 清除所有相关topic中的旧数据
    logging.info("Clearing all topic buffers before starting new tracking session...")
    topics_to_clear = [POSE_TOPIC, REQUEST_TOPIC, MASK_TOPIC, CONTROL_TOPIC]
    
    for topic_name in topics_to_clear:
        try:
            topic = pubsub.get_topic(topic_name)
            if topic and hasattr(topic.storage, 'clear'):
                topic.storage.clear()
                logging.info(f"Successfully cleared {topic_name} buffer")
        except Exception as e:
            logging.warning(f"Failed to clear {topic_name} buffer: {e}")
    
    logging.info("All topic buffers cleared")
    
    code_dir = os.path.dirname(os.path.realpath(__file__))
    mesh_file = os.path.join(code_dir, f'Object_data/{object_name}/{object_name}.obj')
    
    # 检查mesh文件是否存在
    if not os.path.exists(mesh_file):
        logging.error(f"Mesh file not found: {mesh_file}")
        return False
    
    debug = 1
    debug_dir = os.path.join(code_dir, 'debug_realtime')
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, 'track_vis'), exist_ok=True)

    try:
        # --- Get Initial Segmentation with Validation ---
        logging.info(f"Starting segmentation process for object: {object_name}")
        color, depth = capture_and_process_frame(rs_cam)
        mask = get_validated_segmentation_mask(pubsub, rs_cam, color, object_prompt)
        
        # --- FoundationPose Initialization ---
        mesh = trimesh.load(mesh_file)
        
        to_origin_transform, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(
            model_pts=mesh.vertices, 
            model_normals=mesh.vertex_normals, 
            mesh=mesh, 
            scorer=scorer, 
            refiner=refiner, 
            debug_dir=debug_dir, 
            debug=debug, 
            glctx=glctx
        )
        logging.info("FoundationPose estimator initialization done.")

        intrinsics = rs_cam.intrinsics
        cam_K = np.array([
            [intrinsics['fx'], 0, intrinsics['ppx']],
            [0, intrinsics['fy'], intrinsics['ppy']],
            [0, 0, 1]
        ])

        # Register object with the first frame
        pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask.astype(bool), iteration=5)
        with pose_lock:
            latest_pose = pose.copy()
        
        # --- Real-time Tracking Loop ---
        logging.info("Starting real-time tracking...")
        frame_idx = 0
        
        while tracking_active:
            color, depth = rs_cam.get()
            if color is None or depth is None:
                time.sleep(0.01)
                continue

            # 在处理每帧前再次检查tracking_active状态
            if not tracking_active:
                break

            # Process depth image to be consistent with datareader
            depth = depth.astype(np.float64) * rs_cam.depth_scale
            depth[depth < 0.001] = 0 # zfar is np.inf in test.py
            
            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=2)

            # 仅更新最新位姿，按需响应时再发送
            with pose_lock:
                latest_pose = pose.copy()
            
            if debug >= 1:
                center_pose = pose @ np.linalg.inv(to_origin_transform)
                vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imshow('Real-time Tracking', vis[...,::-1])

            if debug >= 2:
                imageio.imwrite(f'{debug_dir}/track_vis/{frame_idx:06d}.png', vis)

            frame_idx += 1
            key = cv2.waitKey(1)
            if key == ord('q'):
                tracking_active = False
                break
            
            # 在循环末尾再次检查tracking_active状态
            if not tracking_active:
                break
        
        # 关闭可视化窗口
        if debug >= 1:
            cv2.destroyWindow('Real-time Tracking')
            logging.info("Closed tracking visualization window")
        
        # 停止时发布50次最终位姿到广播buffer
        logging.info("Tracking stopped. Publishing final poses to buffer for averaging...")
        with pose_lock:
            final_pose = latest_pose.copy() if latest_pose is not None else None
        
        if final_pose is not None:
            for i in range(50):
                pose_data = {
                    'pose_matrix': final_pose.astype(np.float32),
                    'timestamp': time.time(),
                    'frame_idx': np.int32(frame_idx + i),
                    'object_name': encode_text_prompt(object_name)
                }
                try:
                    pubsub.publish(POSE_TOPIC, pose_data)
                    time.sleep(0.001)  # 小延迟确保数据写入
                except Exception as e:
                    logging.error(f"Failed to publish final pose {i}: {e}")
            logging.info("Published 50 final poses to buffer for controller averaging.")
        
        logging.info(f"Tracking session for {object_name} completed.")
        return True
        
    except KeyboardInterrupt:
        logging.info("Tracking session interrupted by user.")
        return False
    except Exception as e:
        logging.error(f"An error occurred during tracking: {e}")
        return False


def wait_for_tracking_command():
    """Wait for tracking start command from external program."""
    global control_received_event, tracking_active, current_object_prompt, current_object_name
    
    logging.info("Waiting for tracking command from external program...")
    
    while True:
        control_received_event.clear()
        
        # Wait for control command
        if control_received_event.wait():
            if tracking_active and current_object_prompt and current_object_name:
                logging.info(f"Starting tracking for: {current_object_name} with prompt: '{current_object_prompt}'")
                return current_object_prompt, current_object_name
            elif not tracking_active:
                logging.info("Received stop command or tracking deactivated")
                return None, None
        
        time.sleep(0.1)


def main():
    """主函数：实现基于外部命令控制的跟踪客户端。"""
    set_logging_format()
    set_seed(0)

    ####################################共享内存的初始化####################################
    pubsub = PubSubManager(port=port, authkey=b'foundationpose')
    global _PUBSUB_REF
    _PUBSUB_REF = pubsub
    pubsub.start(role='both')
    
    # 配置所有需要的topics
    topics_config = {
        REQUEST_TOPIC: {
            'examples': {
                'rgb': np.zeros((480, 640, 3), dtype=np.uint8),
                'prompt': np.zeros(256, dtype=np.uint8)
            },
            'buffer_size': 10,
            'mode': 'consumer'  # 分割请求使用消费者模式
        },
        MASK_TOPIC: {
            'examples': {
                'mask': np.zeros((480, 640), dtype=bool)
            },
            'buffer_size': 10,
            'mode': 'consumer'  # 分割结果使用消费者模式
        },
        CONTROL_TOPIC: {
            'examples': {
                'command': encode_text_prompt('start'),  # 'start' or 'stop'
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
    
    # Register handlers
    pubsub.register_topic_handler(MASK_TOPIC, handle_segmentation_mask, check_interval=0.001)
    pubsub.register_topic_handler(CONTROL_TOPIC, handle_tracking_control, check_interval=0.001)

    # --- Main Application Logic ---
    rs_cam = None
    try:
        # --- RealSense Initialization ---
        rs_cam = RealSense()
        logging.info("RealSense camera initialized successfully.")
        
        # --- Main Control Loop ---
        logging.info("Tracking client started. Waiting for external commands...")
        
        while True:
            # 等待外部程序的开启命令
            object_prompt, object_name = wait_for_tracking_command()
            
            if object_prompt is None or object_name is None:
                # 收到停止命令或无效命令，继续等待
                continue
            
            # 开始跟踪会话
            logging.info(f"Starting tracking session for object: {object_name}")
            success = run_tracking_session(pubsub, rs_cam, object_prompt, object_name)
            
            if success:
                logging.info(f"Tracking session for {object_name} completed successfully.")
            else:
                logging.warning(f"Tracking session for {object_name} ended with issues.")
            logging.info("Ready for next tracking command...")
        
    except KeyboardInterrupt:
        logging.info("Tracking client interrupted by user.")
    except Exception as e:
        logging.error(f"An error occurred in tracking client: {e}")
    finally:
        if rs_cam:
            rs_cam.release()
        cv2.destroyAllWindows()
        pubsub.stop(role='both')
        logging.info("Tracking client finished and PubSub manager shut down.")


if __name__ == '__main__':
    import os
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    main()