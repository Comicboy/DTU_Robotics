import cv2
import json
import numpy as np
import os

# Import your vision module
# (Ensure cv_pipeline_planb.py is in the same directory)
import cv_pipeline_planb as vision

def load_calibration(path="camera_calibration.json"):
    """Helper to load K and D once at startup"""
    if not os.path.exists(path):
        print(f"Error: Calibration file '{path}' not found.")
        return None, None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    K = np.array(data["K"])
    D = np.array(data["D"])
    return K, D

def robot_task():
    # 1. INITIALIZATION
    print("Initializing Robot Vision...")
    K, D = load_calibration()
    
    if K is None:
        return

    # 2. ROBOT MOVEMENT (Pseudo-code)
    # robot.move_to_home()
    # robot.look_at_keyboard()
    
    # 3. GET DATA FROM ROBOT
    # Replace this with your actual camera getter, e.g., cam.get_frame()
    # For now, we load the file on disk
    image = cv2.imread("robot_pose_2.jpg")
    
    if image is None:
        print("Failed to get image from robot camera.")
        return

    # Inputs from Robot Controller
    # You can now pass dynamic values if the robot moves to different heights/angles
    current_pitch = 20.0   # Degrees (Positive = Pitch Down)
    current_height = 82.0 # Millimeters (Lens to Table)

    # 4. CALL THE VISION PIPELINE
    print(f"Processing image with Pitch={current_pitch}° and Height={current_height}mm...")
    
    try:
        # This single line runs Warping, OCR, and 3D Projection
        keys_data, debug_image = vision.process_keyboard_from_robot(
            image=image,
            pitch_deg=current_pitch,
            camera_height_mm=current_height,
            K=K,
            D=D,
            output_json_path="robot_execution_coords.json"
        )
        
        # 5. EXECUTE ROBOT LOGIC
        print(f"\nVision processing complete. Found {len(keys_data)} keys.")
        
        target_key = "H"
        target_location = next((k for k in keys_data if k['label'] == target_key), None)
        
        if target_location:
            coords = target_location['coords_cam']
            print(f"Moving robot to press '{target_key}' at:")
            print(f"  X (Right):   {coords['x']:.2f} mm")
            print(f"  Y (Forward): {coords['y']:.2f} mm")
            print(f"  Z (Depth):   {coords['z']:.2f} mm")
            
            # robot.move_tcp(coords['x'], coords['y'], coords['z'])
        else:
            print(f"Key '{target_key}' not found in the scene.")

        # Optional: Show what the robot 'saw'
        cv2.imshow("Robot Vision Debug", debug_image)
        cv2.waitKey(0)

    except Exception as e:
        print(f"Vision Pipeline Failed: {e}")

if __name__ == "__main__":
    robot_task()