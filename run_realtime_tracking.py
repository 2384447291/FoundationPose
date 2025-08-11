import cv2
import numpy as np
import trimesh
import imageio
import os
import logging
import time
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager, BaseManager
from queue import Empty

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import *

# Import local modules
from realsense.realsense import RealSense
from shm_lib.shared_memory_queue import SharedMemoryQueue

object_prompt = "a yellow and black utility knife"
object_name = "knife"
port = 5000

# Define a manager class to host the shared memory queues
class QueueManager(BaseManager):
    pass

def encode_text_prompt(text: str, max_length: int = 256) -> np.ndarray:
    """Encode a string into a fixed-size numpy array."""
    encoded = text.encode('utf-8')
    if len(encoded) > max_length:
        raise ValueError("Prompt is too long.")
    
    buffer = np.zeros(max_length, dtype=np.uint8)
    buffer[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return buffer

def decode_text_prompt(encoded_array: np.ndarray) -> str:
    """Decode a numpy array back to a string."""
    null_idx = np.where(encoded_array == 0)[0]
    if len(null_idx) > 0:
        encoded_array = encoded_array[:null_idx[0]]
    return encoded_array.tobytes().decode('utf-8')


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

    # --- Shared Memory Server Initialization ---
    with SharedMemoryManager() as shm_manager:
        # Create request and response queues
        req_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={
                'rgb': np.zeros((480, 640, 3), dtype=np.uint8),
                'prompt': np.zeros(256, dtype=np.uint8)
            },
            buffer_size=1
        )
        res_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={'mask': np.zeros((480, 640), dtype=bool)},
            buffer_size=1
        )
        
        # Register queues with our custom manager
        QueueManager.register('get_req_queue', callable=lambda: req_queue)
        QueueManager.register('get_res_queue', callable=lambda: res_queue)
        
        # Start the manager server
        manager = QueueManager(address=('', port), authkey=b'foundationpose')
        server = manager.get_server()
        
        server_process = mp.Process(target=server.serve_forever)
        server_process.daemon = True
        server_process.start()
        logging.info(f"Shared Memory server started at port {port}.")

        # --- Main Application Logic ---
        rs_cam = None
        try:
            # --- RealSense Initialization ---
            rs_cam = RealSense()
            
            # --- Get Mask for First Frame ---
            logging.info("Waiting for a client to connect and process the first frame...")
            color, depth = None, None
            while color is None:
                color, depth = rs_cam.get()
                time.sleep(0.1) # Wait briefly for camera to warm up

            # Process depth image to be consistent with datareader
            if depth is not None:
                depth = depth.astype(np.float64) * rs_cam.depth_scale
                depth[depth < 0.001] = 0 # zfar is np.inf in test.py

            # Send segmentation request
            req_data = {
                'rgb': color,
                'prompt': encode_text_prompt(object_prompt)
            }
            req_queue.put(req_data)
            logging.info(f"Sent request with prompt: '{object_prompt}'")
            
            # Wait for segmentation response
            logging.info("Waiting for segmentation mask from client...")
            try:
                # The get() method is now blocking by default.
                res_data = res_queue.get(timeout=30) # Wait for 30 seconds
            except Empty:
                logging.error("Timeout: Did not receive segmentation mask from client within 30 seconds.")
                raise
            
            mask = res_data['mask']
            logging.info("Received segmentation mask.")

            # Visualize the received mask
            if debug >= 1:
                mask_vis = (mask.astype(np.uint8) * 255)
                # Convert mask to 3 channels to stack with color image
                mask_vis_color = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                combined_vis = np.hstack((color, mask_vis_color))
                cv2.imshow('Initial Segmentation Result (Original vs Mask)', combined_vis)
                logging.info("Displaying segmentation result. Press any key in the window to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow('Initial Segmentation Result (Original vs Mask)')
            
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
            server_process.terminate()
            logging.info("Tracking finished and server shut down.")


if __name__ == '__main__':
    import os
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    main()