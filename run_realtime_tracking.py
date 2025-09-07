import os


import cv2
import numpy as np
import trimesh
import imageio
import logging
import time
import threading

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import *

# 导入相机模组和共享内存模组
from realsense.realsense import RealSense
from shm_lib.pubsub_manager import PubSubManager
from shm_lib.shared_memory_util import encode_text_prompt


object_prompt = "a green bittermelon"
object_name = "bittermelon"
port = 10000

# 共享内存的topic
REQUEST_TOPIC = "segmentation_requests"
MASK_TOPIC = "segmentation_masks"

# Global variables for mask receiving
received_mask = None
mask_received_event = threading.Event()


def handle_segmentation_mask(data):
    """Handle received segmentation mask."""
    global received_mask, mask_received_event
    
    mask = data['mask']
    received_mask = mask
    mask_received_event.set()  # Signal that mask has been received
    logging.info(f"Received segmentation mask with {np.sum(mask)} positive pixels")


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


def main():
    # --- Configuration ---
    code_dir = os.path.dirname(os.path.realpath(__file__))
    mesh_file = os.path.join(code_dir, f'Object_data/{object_name}/{object_name}.obj')
    
    debug = 1
    debug_dir = os.path.join(code_dir, 'debug_realtime')
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, 'track_vis'), exist_ok=True)

    set_logging_format()
    set_seed(0)
 
    ####################################共享内存的初始化####################################
    pubsub = PubSubManager(port=port, authkey=b'foundationpose')
    pubsub.start(role='both')
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
    
    # Register handler for receiving segmentation masks
    pubsub.register_topic_handler(MASK_TOPIC, handle_segmentation_mask, check_interval=0.001)

    # --- Main Application Logic ---
    rs_cam = None
    try:
        # --- RealSense Initialization ---
        rs_cam = RealSense()
        
        # --- Get Initial Segmentation with Validation ---
        logging.info("Starting segmentation process...")
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
        # --- Real-time Tracking Loop ---
        logging.info("Starting real-time tracking...")
        frame_idx = 0
        while True:
            color, depth = rs_cam.get()
            if color is None or depth is None:
                continue

            # Process depth image to be consistent with datareader
            depth = depth.astype(np.float64) * rs_cam.depth_scale
            depth[depth < 0.001] = 0 # zfar is np.inf in test.py
            
            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=2)
            
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
                break
        
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if rs_cam:
            rs_cam.release()
        cv2.destroyAllWindows()
        pubsub.stop(role='both')
        logging.info("Tracking finished and PubSub manager shut down.")


if __name__ == '__main__':
    import os
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    main()