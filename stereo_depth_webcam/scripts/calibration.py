#!/usr/bin/env python3
# calibration.py
import cv2
import numpy as np
import json
import os
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

class StereoCameraCalibrator:
    def __init__(self, checkerboard_size=(8, 6), square_size=0.019, 
                 combined_size=(1280, 480), target_frames=50, baseline=0.065):  # Default 65mm baseline
        """
        Initialize the stereo camera calibrator.
        
        Args:
            checkerboard_size: Number of internal corners on checkerboard (width, height)
            square_size: Physical size of each checkerboard square in meters
            combined_size: Size of the combined stereo feed (width, height)
            target_frames: Number of successful calibration frames to collect
        """
        if combined_size[0] % 2 != 0:
            raise ValueError("Combined width must be even for stereo split")
            
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.combined_size = combined_size
        self.single_size = (combined_size[0] // 2, combined_size[1])
        self.target_frames = target_frames
        
        # Create the object points array
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                   0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Storage for calibration points
        self.left_imgpoints = []
        self.right_imgpoints = []
        self.objpoints = []
        
        # Store baseline distance between cameras
        self.baseline = baseline
        
        # Optional: Storage for calibration frames
        self.save_frames = False
        self.frame_dir = "calibration_frames"
        if self.save_frames:
            os.makedirs(self.frame_dir, exist_ok=True)
    
    def add_frames(self, left_img: np.ndarray, 
                  right_img: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Process a pair of frames for calibration.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            
        Returns:
            Tuple of (success, visualization image)
        """
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Create visualization image
        vis_img = np.hstack((left_img, right_img))
        
        # Find checkerboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + \
                cv2.CALIB_CB_FAST_CHECK
        ret_left, corners_left = cv2.findChessboardCorners(left_gray, 
                                                         self.checkerboard_size, flags)
        ret_right, corners_right = cv2.findChessboardCorners(right_gray, 
                                                           self.checkerboard_size, flags)
        
        if ret_left and ret_right:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), 
                                          (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), 
                                           (-1, -1), criteria)
            
            # Store points
            self.objpoints.append(self.objp)
            self.left_imgpoints.append(corners_left)
            self.right_imgpoints.append(corners_right)
            
            # Save calibration frames if enabled
            if self.save_frames:
                frame_num = len(self.objpoints)
                cv2.imwrite(os.path.join(self.frame_dir, f"calib_{frame_num:03d}.png"), 
                           vis_img)
            
            # Draw corners
            cv2.drawChessboardCorners(left_img, self.checkerboard_size, 
                                    corners_left, ret_left)
            cv2.drawChessboardCorners(right_img, self.checkerboard_size, 
                                    corners_right, ret_right)
            
            # Update visualization
            vis_img = np.hstack((left_img, right_img))
            cv2.putText(vis_img, f"Frames: {len(self.objpoints)}", 
                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return ret_left and ret_right, vis_img
    
    def calibrate(self) -> Dict:
        """
        Perform stereo camera calibration.
        
        Returns:
            Dictionary containing calibration parameters
        """
        if not self.objpoints:
            raise ValueError("No calibration data collected")
        
        print("Calibrating individual cameras...")
        # Calibrate each camera individually
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            self.objpoints, self.left_imgpoints, self.single_size, None, None)
        
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            self.objpoints, self.right_imgpoints, self.single_size, None, None)
        
        print("Performing stereo calibration...")
        # Perform stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = \
            cv2.stereoCalibrate(
                self.objpoints, self.left_imgpoints, self.right_imgpoints,
                mtx_left, dist_left, mtx_right, dist_right, self.single_size,
                criteria=criteria, flags=flags)
        
        print("Computing rectification transforms...")
        # Compute rectification transforms
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right, self.single_size, R, T)
        
        # Verify translation magnitude matches baseline
        translation_magnitude = np.linalg.norm(T)
        scale_factor = self.baseline / translation_magnitude
        
        # Scale translation vector to match known baseline
        T = T * scale_factor
        
        print(f"Translation magnitude before scaling: {translation_magnitude:.3f} units")
        print(f"Applied scale factor: {scale_factor:.3f}")
        print(f"Final baseline: {self.baseline:.3f} meters")
        
        # Package results
        params = {
            'resolution': {
                'combined': list(self.combined_size),
                'single': list(self.single_size)
            },
            'left_camera': {
                'camera_matrix': mtx_left.tolist(),
                'dist_coeffs': dist_left.tolist(),
                'calibration_error': float(ret_left)
            },
            'right_camera': {
                'camera_matrix': mtx_right.tolist(),
                'dist_coeffs': dist_right.tolist(),
                'calibration_error': float(ret_right)
            },
            'stereo': {
                'R': R.tolist(),
                'T': T.tolist(),
                'E': E.tolist(),
                'F': F.tolist(),
                'Q': Q.tolist(),
                'error': float(ret_stereo),
                'rectification': {
                    'R1': R1.tolist(),
                    'R2': R2.tolist(),
                    'P1': P1.tolist(),
                    'P2': P2.tolist(),
                    'roi_left': [int(x) for x in roi_left],
                    'roi_right': [int(x) for x in roi_right]
                }
            }
        }
        
        return params

class StereoCamera:
    def __init__(self, camera_index=0, combined_size=(1280, 480)):
        """
        Initialize a stereo camera using a single camera with side-by-side output.
        
        Args:
            camera_index: Index of the camera
            combined_size: Size of the combined stereo feed (width, height)
        """
        self.cap = cv2.VideoCapture(camera_index)
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, combined_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, combined_size[1])
        
        # Store frame size
        self.combined_size = combined_size
        
        # Verify camera is open
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {camera_index}")
        
        # First frame
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame")
        
        print(f"Camera opened with resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    def get_frames(self):
        """
        Capture a frame and split it into left and right images.
        
        Returns:
            Tuple of (left_frame, right_frame) or (None, None) if capture failed
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None
        
        # Split frame into left and right
        mid = frame.shape[1] // 2
        left_frame = frame[:, :mid]
        right_frame = frame[:, mid:]
        
        return left_frame, right_frame
    
    def cleanup(self):
        """Release camera resources."""
        if self.cap.isOpened():
            self.cap.release()

def main():
    """Main entry point for stereo camera calibration."""
    parser = argparse.ArgumentParser(description='Stereo Camera Calibration Tool')
    parser.add_argument('--device', type=int, default=0, help='Camera device index')
    parser.add_argument('--width', type=int, default=1280, help='Combined frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--cb-width', type=int, default=8, help='Checkerboard width (inner corners)')
    parser.add_argument('--cb-height', type=int, default=6, help='Checkerboard height (inner corners)')
    parser.add_argument('--square-size', type=float, default=0.019, help='Size of checkerboard square in meters')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames to capture')
    parser.add_argument('--baseline', type=float, default=0.065, help='Baseline between cameras in meters')
    parser.add_argument('--output', type=str, default='stereo_calibration.json', help='Output file path')
    args = parser.parse_args()
    
    print("Initializing stereo calibration...")
    
    # Create output directory if necessary
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize calibrator and camera
    calibrator = StereoCameraCalibrator(
        checkerboard_size=(args.cb_width, args.cb_height),
        square_size=args.square_size,
        combined_size=(args.width, args.height),
        target_frames=args.frames,
        baseline=args.baseline
    )
    
    print("Opening stereo camera...")
    try:
        camera = StereoCamera(camera_index=args.device, combined_size=(args.width, args.height))
    except Exception as e:
        print(f"Error opening camera: {str(e)}")
        return
    
    # Verify camera setup
    left_frame, right_frame = camera.get_frames()
    if left_frame is None or right_frame is None:
        print("Error: Could not capture frames from camera")
        camera.cleanup()
        return
    
    print(f"\nCamera setup verification:")
    print(f"Combined frame size: {args.width}x{args.height}")
    print(f"Individual frame size: {left_frame.shape[:2][::-1]}")
    
    frame_count = 0
    
    print("\nStarting stereo calibration:")
    print("1. Hold the checkerboard visible to both cameras")
    print("2. Move it slowly to different positions and angles")
    print("3. Try to cover the entire field of view")
    print(f"Need {args.frames} good captures. Press 'q' to quit.\n")
    
    cv2.namedWindow('Stereo Calibration', cv2.WINDOW_NORMAL)
    
    try:
        while frame_count < args.frames:
            left_frame, right_frame = camera.get_frames()
            if left_frame is None or right_frame is None:
                print("Failed to capture frames")
                break
            
            success, vis_img = calibrator.add_frames(left_frame, right_frame)
            
            if success:
                frame_count += 1
                print(f"\rSuccess! Captured frame {frame_count}/{args.frames}", end='')
            
            if vis_img is not None:
                cv2.imshow('Stereo Calibration', vis_img)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                print("\nCalibration cancelled by user")
                break
    
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
    
    finally:
        print("\nCleaning up...")
        camera.cleanup()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    
    if frame_count > 0:
        print("\nCalculating calibration parameters...")
        try:
            params = calibrator.calibrate()
            
            with open(args.output, 'w') as f:
                json.dump(params, f, indent=4)
                
            print(f"\nCalibration complete! Parameters saved to '{args.output}'")
            
            # Print calibration results summary
            print("\nCalibration Results Summary:")
            print(f"Left camera error: {params['left_camera']['calibration_error']:.6f}")
            print(f"Right camera error: {params['right_camera']['calibration_error']:.6f}")
            print(f"Stereo calibration error: {params['stereo']['error']:.6f}")
            
        except Exception as e:
            print(f"\nError during calibration: {str(e)}")
    else:
        print("\nNo frames captured. Calibration failed.")

if __name__ == "__main__":
    main()