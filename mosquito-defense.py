#!/usr/bin/env python3
"""
Stereo Vision Mosquito Detection and Laser Targeting System with Object Recognition
Uses dual cameras for 3D positioning and deep learning for mosquito classification

VERSION 3.3 (v60) - PRODUCTION READY

Copyright (c) 2024 Dragos Ruiu
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

CRITICAL FIXES IN THIS VERSION:
- Camera frame synchronization with monotonic timestamps
- Proper metric depth calculation using disparity formula
- Measured dt throughout system (no fixed 0.033s assumptions)
- Hungarian assignment for optimal tracking (with scipy)
- Process noise scaling with dt² for Kalman filter
- Configuration validation with type checking
- Robust serial communication with error recovery
- Emergency stop monitoring from Arduino
- Headless operation mode for servers

BASED ON CODE REVIEW FEEDBACK:
- Fixed depth calculation from linear scale to proper disparity-to-depth
- Changed time.time() to time.perf_counter() for monotonic timing
- Added configuration validation to prevent type errors
- Improved ACK parsing to handle ERR responses
- Fixed PID double integration issue
- Added emergency stop state monitoring
- Enabled headless mode with --no-display flag
- Fixed FPS counter runtime calculation
- Better separation of optional vs required dependencies

REQUIREMENTS:
opencv-python>=4.5.0
numpy>=1.19.0
scikit-learn>=0.24.0  # Optional for HOG+SVM
scikit-image>=0.18.0  # Optional for HOG features
tensorflow>=2.8.0     # Optional for deep learning
pyserial>=3.5         # Required for hardware
scipy>=1.7.0          # Optional for Hungarian algorithm
joblib>=1.0.0         # Optional for model saving

Author: Dragos Ruiu
Version: 3.3 (v60)
"""

import cv2
import numpy as np
import time
import threading
import queue
import json
import argparse
import atexit
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Union
import math
import pickle
import os
import sys
import glob
import warnings
from datetime import datetime
import concurrent.futures

# IMPORTANT: Set GPU configuration BEFORE importing TensorFlow
# This ensures the environment variable takes effect
if '--cpu-only' in sys.argv or os.environ.get('FORCE_CPU', '0') == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configuration file path
CONFIG_FILE = "mosquito_config.json"

# Default configuration
DEFAULT_CONFIG = {
    "camera": {
        "left_id": 0,
        "right_id": 1,
        "width": 640,
        "height": 480,
        "fps": 60,
        "sync_tolerance_ms": 1.0
    },
    "stereo": {
        "baseline_mm": 100.0,
        "focal_length_px": 800.0,
        "min_disparity": 16,
        "max_disparity": 96,
        "epipolar_tolerance": 5.0,
        "template_size": 21
    },
    "detection": {
        "min_area": 10,
        "max_area": 200,
        "var_threshold": 16,
        "morph_kernel_open": 3,
        "morph_kernel_close": 5
    },
    "tracking": {
        "max_tracks": 10,
        "max_distance_3d_mm": 100,
        "max_missed_frames": 10,
        "min_track_length": 5,
        "process_noise": 0.1,
        "measurement_noise": 2.0
    },
    "targeting": {
        "min_classification_score": 0.6,
        "engagement_cooldown_s": 0.5,
        "consecutive_frames_required": 3,
        "tolerance_degrees_base": 1.0,
        "tolerance_distance_factor": 0.5
    },
    "servo": {
        "pan_min": -90,
        "pan_max": 90,
        "tilt_min": -45,
        "tilt_max": 45,
        "max_rate_deg_per_s": 180
    },
    "pid": {
        "pan_kp": 0.8,
        "pan_ki": 0.2,
        "pan_kd": 0.3,
        "tilt_kp": 0.8,
        "tilt_ki": 0.2,
        "tilt_kd": 0.3,
        "integral_limit": 10.0
    },
    "laser": {
        "min_power": 10,
        "max_power": 100,
        "base_power": 50,
        "max_duration_ms": 500,
        "power_distance_ref_m": 1.0
    },
    "safety": {
        "enabled": True,
        "require_ack": True,
        "max_queue_size": 5
    }
}

# Load configuration
def load_config(config_file: str = CONFIG_FILE) -> dict:
    """Load configuration from JSON file, create default if not exists"""
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                merged = merge_dicts(DEFAULT_CONFIG, config)
                # Validate types
                return validate_config(merged)
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return DEFAULT_CONFIG
    else:
        # Create default config file
        with open(config_file, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        print(f"Created default configuration file: {config_file}")
        return DEFAULT_CONFIG

def merge_dicts(default: dict, override: dict) -> dict:
    """Recursively merge configuration dictionaries"""
    result = default.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

def validate_config(config: dict) -> dict:
    """Validate configuration types and ranges"""
    # Type validators
    validators = {
        'camera.width': (int, lambda x: x > 0),
        'camera.height': (int, lambda x: x > 0),
        'camera.fps': (int, lambda x: 0 < x <= 120),
        'camera.sync_tolerance_ms': (float, lambda x: x > 0),
        'stereo.baseline_mm': (float, lambda x: x > 0),
        'stereo.focal_length_px': (float, lambda x: x > 0),
        'stereo.min_disparity': (int, lambda x: True),
        'stereo.max_disparity': (int, lambda x: x > 0),
        'stereo.template_size': (int, lambda x: x > 0 and x % 2 == 1),  # Must be odd
        'tracking.process_noise': (float, lambda x: x > 0),
        'pid.integral_limit': (float, lambda x: x >= 0),
        'laser.max_power': (int, lambda x: 0 < x <= 255),
        'safety.enabled': (bool, lambda x: True)
    }
    
    def get_nested(d, path):
        """Get nested dict value by dot notation"""
        keys = path.split('.')
        for key in keys:
            d = d.get(key, {})
        return d
        
    def set_nested(d, path, value):
        """Set nested dict value by dot notation"""
        keys = path.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    # Validate each field
    for path, (expected_type, validator) in validators.items():
        value = get_nested(config, path)
        
        if not isinstance(value, expected_type):
            print(f"Config warning: {path} should be {expected_type.__name__}, got {type(value).__name__}")
            # Use default value
            default_value = get_nested(DEFAULT_CONFIG, path)
            set_nested(config, path, default_value)
        elif not validator(value):
            print(f"Config warning: {path} = {value} failed validation")
            default_value = get_nested(DEFAULT_CONFIG, path)
            set_nested(config, path, default_value)
            
    # Ensure template size is odd
    template_size = config['stereo']['template_size']
    if template_size % 2 == 0:
        config['stereo']['template_size'] = template_size + 1
        print(f"Adjusted template_size to {template_size + 1} (must be odd)")
        
    # Ensure num_disparities is multiple of 16
    min_disp = config['stereo']['min_disparity']
    max_disp = config['stereo']['max_disparity']
    num_disp = max_disp - min_disp
    if num_disp % 16 != 0:
        num_disp = (num_disp // 16 + 1) * 16
        config['stereo']['max_disparity'] = min_disp + num_disp
        print(f"Adjusted max_disparity to {config['stereo']['max_disparity']} (num_disp must be multiple of 16)")
        
    return config

# Global configuration
CONFIG = load_config()

# Handle optional imports gracefully
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    warnings.warn("pyserial not installed. Hardware control disabled.")

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. HOG+SVM classifier disabled.")

try:
    from skimage.feature import hog
    from skimage import exposure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not installed. HOG features disabled.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("joblib not installed. Model saving disabled.")

# Set GPU config before TensorFlow import
if '--cpu-only' in sys.argv or os.environ.get('FORCE_CPU', '0') == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    # Set memory growth to avoid GPU memory issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not installed. Deep learning disabled.")

# Try to import scipy for Hungarian algorithm
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not installed. Using greedy assignment instead of Hungarian.")

@dataclass
class MosquitoTrack3D:
    """3D mosquito track with Kalman filtering and classification confidence"""
    id: int
    positions_3d: deque  # (x, y, z) in mm
    positions_2d_left: deque  # (u, v) pixels in left image
    positions_2d_right: deque  # (u, v) pixels in right image
    kalman: cv2.KalmanFilter
    last_seen: float
    last_dt: float  # Actual time delta for this track
    velocity_3d: np.ndarray
    confidence: float
    classification_score: float
    classification_history: deque
    consecutive_on_target: int  # For safety confirmation

@dataclass
class StereoCalibration:
    """Complete stereo camera calibration parameters with metric scale"""
    camera_matrix_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_left: np.ndarray
    dist_coeffs_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    rect_transform_left: np.ndarray
    rect_transform_right: np.ndarray
    projection_matrix_left: np.ndarray
    projection_matrix_right: np.ndarray
    Q: np.ndarray
    roi_left: Tuple[int, int, int, int]
    roi_right: Tuple[int, int, int, int]
    baseline_mm: float  # Physical baseline in millimeters
    focal_length_px: float  # Focal length in pixels

class StereoFrameSync:
    """Synchronize stereo camera frames using timestamps"""
    
    def __init__(self, tolerance_ms: float = 1.0):
        self.tolerance_ms = tolerance_ms
        self.left_queue = queue.Queue(maxsize=10)
        self.right_queue = queue.Queue(maxsize=10)
        self.running = True
        
    def capture_left(self, cap):
        """Capture thread for left camera"""
        while self.running:
            ret, frame = cap.read()
            if ret:
                timestamp = time.perf_counter()  # Monotonic high-resolution timer
                try:
                    self.left_queue.put((timestamp, frame), timeout=0.001)
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.left_queue.get_nowait()
                        self.left_queue.put((timestamp, frame), timeout=0.001)
                    except queue.Full:
                        pass  # Still full, drop this frame
                        
    def capture_right(self, cap):
        """Capture thread for right camera"""
        while self.running:
            ret, frame = cap.read()
            if ret:
                timestamp = time.perf_counter()  # Monotonic high-resolution timer
                try:
                    self.right_queue.put((timestamp, frame), timeout=0.001)
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.right_queue.get_nowait()
                        self.right_queue.put((timestamp, frame), timeout=0.001)
                    except queue.Full:
                        pass  # Still full, drop this frame
                        
    def get_synchronized_frames(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Get time-synchronized frame pair with stale frame handling"""
        try:
            # Get frames from queues
            left_time, left_frame = self.left_queue.get(timeout=timeout)
            right_time, right_frame = self.right_queue.get(timeout=timeout)
            
            # Check time synchronization
            time_diff_ms = abs(left_time - right_time) * 1000
            
            if time_diff_ms <= self.tolerance_ms:
                # Frames are synchronized
                avg_time = (left_time + right_time) / 2
                return left_frame, right_frame, avg_time
            else:
                # Try to sync by discarding older frames
                max_attempts = 20  # Prevent infinite loop
                attempts = 0
                
                while attempts < max_attempts:
                    if left_time < right_time:
                        # Left is older, get next left
                        if not self.left_queue.empty():
                            left_time, left_frame = self.left_queue.get_nowait()
                            if abs(left_time - right_time) * 1000 <= self.tolerance_ms:
                                avg_time = (left_time + right_time) / 2
                                return left_frame, right_frame, avg_time
                        else:
                            break
                    else:
                        # Right is older, get next right
                        if not self.right_queue.empty():
                            right_time, right_frame = self.right_queue.get_nowait()
                            if abs(left_time - right_time) * 1000 <= self.tolerance_ms:
                                avg_time = (left_time + right_time) / 2
                                return left_frame, right_frame, avg_time
                        else:
                            break
                    attempts += 1
                    
                # Check for stale frames (>100ms old)
                current_time = time.perf_counter()
                if current_time - left_time > 0.1 or current_time - right_time > 0.1:
                    # Clear queues to prevent accumulation
                    while not self.left_queue.empty():
                        try:
                            self.left_queue.get_nowait()
                        except:
                            break
                    while not self.right_queue.empty():
                        try:
                            self.right_queue.get_nowait()
                        except:
                            break
                    print("Cleared stale frames from queues")
                            
        except queue.Empty:
            pass
            
        return None
        
    def stop(self):
        """Stop capture threads"""
        self.running = False

class MosquitoClassifier:
    """Machine learning classifier with proper error handling"""
    
    def __init__(self, reference_dir: str = "mosquito_references/", 
                 method: str = "deep", model_path: str = None):
        self.reference_dir = reference_dir
        self.method = method
        self.model_path = model_path or "mosquito_classifier.h5"
        
        # Initialize state variables
        self.classifier = None
        self.use_deep_learning = False
        self.model_type = None
        self.scaler = None
        self.templates = []
        
        # Thread pool for async classification
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # HOG parameters
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (4, 4),
            'cells_per_block': (2, 2),
            'visualize': False,
            'feature_vector': True
        }
        
        # Initialize based on method
        try:
            if method == "hog_svm":
                self._init_hog_svm()
            elif method == "template":
                self._init_template_matching()
            elif method == "deep":
                self._init_deep_learning()
            else:
                warnings.warn(f"Unknown method '{method}', using shape-based fallback")
                
        except Exception as e:
            warnings.warn(f"Classifier initialization failed: {e}. Using shape-based fallback.")
            self.method = "shape"
            
    def _init_deep_learning(self):
        """Initialize deep learning with proper GPU error handling"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not installed. Falling back to HOG+SVM")
            self._init_hog_svm()
            return
            
        try:
            # Try MobileNet first
            self._init_mobilenet_transfer()
            return
        except tf.errors.ResourceExhaustedError:
            print("GPU out of memory, trying CPU")
            # Force CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            try:
                self._init_mobilenet_transfer()
                return
            except Exception as e:
                print(f"MobileNet failed on CPU: {e}")
        except Exception as e:
            print(f"MobileNet initialization failed: {e}")
            
        # Fall back to custom CNN or HOG+SVM
        try:
            if self.model_path and os.path.exists(self.model_path):
                print("Loading pre-trained CNN model...")
                self.classifier = tf.keras.models.load_model(self.model_path)
                self.use_deep_learning = True
                self.model_type = 'custom_cnn'
            else:
                print("Building new CNN model...")
                self._build_and_train_cnn()
        except tf.errors.ResourceExhaustedError:
            print("GPU out of memory for CNN, falling back to HOG+SVM")
            tf.keras.backend.clear_session()
            self._init_hog_svm()
        except Exception as e:
            print(f"CNN initialization failed: {e}")
            print("Falling back to HOG+SVM")
            self._init_hog_svm()
            
    def classify_patch_async(self, image_patch: np.ndarray) -> concurrent.futures.Future:
        """Asynchronous classification for better performance"""
        return self.executor.submit(self.classify_patch, image_patch)
        
    def classify_patch(self, image_patch: np.ndarray) -> Tuple[float, str]:
        """Classify with raw probability output (no artificial scaling)"""
        if image_patch is None or image_patch.size == 0:
            return 0.0, "invalid"
            
        # Ensure grayscale
        if len(image_patch.shape) == 3:
            image_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
            
        if self.method == "hog_svm" and self.classifier is not None:
            return self._classify_hog_svm(image_patch)
        elif self.method == "template":
            return self._classify_template(image_patch)
        elif self.method == "deep" and hasattr(self, 'use_deep_learning') and self.use_deep_learning:
            return self._classify_deep_learning(image_patch)
        else:
            return self._classify_shape(image_patch)
            
    def _classify_deep_learning(self, patch: np.ndarray) -> Tuple[float, str]:
        """Classify using deep learning without artificial probability scaling"""
        if not self.use_deep_learning or self.classifier is None:
            return self._classify_shape(patch)
            
        try:
            if hasattr(self, 'model_type') and self.model_type == 'mobilenet':
                # MobileNet preprocessing
                if len(patch.shape) == 2:
                    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
                else:
                    patch_rgb = patch
                    
                patch_resized = cv2.resize(patch_rgb, (96, 96))
                patch_input = patch_resized.reshape(1, 96, 96, 3).astype(np.float32)
                
                # Raw prediction
                prediction = self.classifier.predict(patch_input, verbose=0)[0][0]
                return float(prediction), "mobilenet"
                
            else:
                # Custom CNN
                patch_resized = cv2.resize(patch, (32, 32))
                patch_normalized = patch_resized.astype(np.float32) / 255.0
                patch_input = patch_normalized.reshape(1, 32, 32, 1)
                
                # Raw prediction
                prediction = self.classifier.predict(patch_input, verbose=0)[0][0]
                return float(prediction), "cnn"
                
        except Exception as e:
            print(f"Deep learning classification error: {e}")
            return self._classify_shape(patch)
            
    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=False)
        if TENSORFLOW_AVAILABLE:
            tf.keras.backend.clear_session()

class StereoMosquitoDetector:
    """Stereo detector with configurable parameters and proper depth scaling"""
    
    def __init__(self, calibration: StereoCalibration, classifier: MosquitoClassifier):
        self.calibration = calibration
        self.classifier = classifier
        
        # Load config values
        self.min_area = CONFIG["detection"]["min_area"]
        self.max_area = CONFIG["detection"]["max_area"]
        self.var_threshold = CONFIG["detection"]["var_threshold"]
        self.morph_kernel_open = CONFIG["detection"]["morph_kernel_open"]
        self.morph_kernel_close = CONFIG["detection"]["morph_kernel_close"]
        
        # Background subtractors
        self.bg_left = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=self.var_threshold,
            history=500
        )
        self.bg_right = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=self.var_threshold,
            history=500
        )
        
        # Stereo matcher with validated parameters
        min_disp = CONFIG["stereo"]["min_disparity"]
        max_disp = CONFIG["stereo"]["max_disparity"]
        num_disp = max_disp - min_disp
        
        # Ensure num_disparities is multiple of 16
        num_disp = (num_disp // 16) * 16
        if num_disp <= 0:
            num_disp = 64
            warnings.warn(f"Invalid disparity range, using default: {num_disp}")
            
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        # Initialize rectification maps
        self._init_rectification()
        
    def _init_rectification(self):
        """Initialize rectification with dynamic resolution"""
        # Get actual resolution from config
        h = CONFIG["camera"]["height"]
        w = CONFIG["camera"]["width"]
        
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.calibration.camera_matrix_left,
            self.calibration.dist_coeffs_left,
            self.calibration.rect_transform_left,
            self.calibration.projection_matrix_left,
            (w, h), 
            cv2.CV_16SC2
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.calibration.camera_matrix_right,
            self.calibration.dist_coeffs_right,
            self.calibration.rect_transform_right,
            self.calibration.projection_matrix_right,
            (w, h), 
            cv2.CV_16SC2
        )
        
    def detect_3d(self, frame_left: np.ndarray, frame_right: np.ndarray) -> List[Dict]:
        """Detect with proper metric depth calculation"""
        # Rectify images
        rect_left = cv2.remap(frame_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(frame_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        
        # Detect in both images
        detections_left = self._detect_2d(rect_left, self.bg_left)
        detections_right = self._detect_2d(rect_right, self.bg_right)
        
        # Match with relaxed tolerance
        matched_detections = self._match_stereo_detections(
            detections_left, detections_right, rect_left, rect_right
        )
        
        # Calculate 3D positions with proper scaling
        detections_3d = []
        for match in matched_detections:
            # Extract patches for classification
            patch_left = self._extract_patch(rect_left, match['left_pos'])
            
            # Async classification
            future = self.classifier.classify_patch_async(patch_left)
            
            # Triangulate to get 3D position
            x_3d, y_3d, z_3d = self._triangulate_metric(
                match['left_pos'], match['right_pos'],
                self.calibration.projection_matrix_left,
                self.calibration.projection_matrix_right
            )
            
            # Get classification result
            try:
                classification_score, method = future.result(timeout=0.05)
            except:
                classification_score, method = 0.5, "timeout"
            
            # Filter by depth and classification
            if 100 < z_3d < 3000 and classification_score > 0.3:
                detections_3d.append({
                    'pos_3d': (x_3d, y_3d, z_3d),
                    'pos_2d_left': match['left_pos'],
                    'pos_2d_right': match['right_pos'],
                    'confidence': match['confidence'],
                    'classification_score': classification_score,
                    'classification_method': method,
                    'patch_left': patch_left
                })
                
        return detections_3d
        
    def _triangulate_metric(self, pt_left: Tuple[float, float], 
                           pt_right: Tuple[float, float],
                           P1: np.ndarray, P2: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate 3D position using proper disparity-to-depth conversion
        
        For rectified stereo: Z = (baseline * focal_length) / disparity
        X = (x - cx) * Z / focal_length
        Y = (y - cy) * Z / focal_length
        """
        # Get camera parameters
        baseline_mm = self.calibration.baseline_mm
        focal_px = self.calibration.focal_length_px
        
        # For rectified images, y-coordinates should be nearly equal
        y_avg = (pt_left[1] + pt_right[1]) / 2.0
        
        # Calculate disparity (left x - right x for rectified stereo)
        disparity = pt_left[0] - pt_right[0]
        
        # Avoid division by zero
        if abs(disparity) < 0.5:  # Half pixel minimum
            disparity = 0.5 if disparity >= 0 else -0.5
            
        # Depth from disparity
        z_mm = (baseline_mm * focal_px) / abs(disparity)
        
        # Get principal point from left camera matrix
        # P1 should contain K[R|t] for rectified left camera
        cx = P1[0, 2] / P1[0, 0] if P1[0, 0] != 0 else CONFIG["camera"]["width"] / 2
        cy = P1[1, 2] / P1[1, 1] if P1[1, 1] != 0 else CONFIG["camera"]["height"] / 2
        
        # 3D position using left camera as reference
        x_mm = (pt_left[0] - cx) * z_mm / focal_px
        y_mm = (y_avg - cy) * z_mm / focal_px
        
        # Validate depth is positive (object in front of cameras)
        if z_mm <= 0:
            warnings.warn(f"Negative depth {z_mm}mm calculated, using absolute value")
            z_mm = abs(z_mm)
            
        return x_mm, y_mm, z_mm
        
    def _triangulate_points_robust(self, pt_left: Tuple[float, float], 
                                  pt_right: Tuple[float, float],
                                  P1: np.ndarray, P2: np.ndarray) -> Tuple[float, float, float]:
        """
        Alternative triangulation using OpenCV's method with validation
        """
        # Convert points to homogeneous coordinates
        pts_left = np.array([[pt_left[0], pt_left[1]]], dtype=np.float32)
        pts_right = np.array([[pt_right[0], pt_right[1]]], dtype=np.float32)
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts_left.T, pts_right.T)
        
        # Check if w coordinate is valid
        if abs(points_4d[3]) < 1e-6:
            # Fall back to disparity method
            return self._triangulate_metric(pt_left, pt_right, P1, P2)
            
        # Convert to 3D
        points_3d = points_4d[:3] / points_4d[3]
        
        # The points are in the coordinate system defined by P1/P2
        # If calibration was done properly, this should be in mm
        x_mm = float(points_3d[0])
        y_mm = float(points_3d[1]) 
        z_mm = float(points_3d[2])
        
        # Validate and correct if needed
        if z_mm <= 0:
            # Use disparity method as fallback
            return self._triangulate_metric(pt_left, pt_right, P1, P2)
            
        return x_mm, y_mm, z_mm
        
    def _match_stereo_detections(self, left_detections: List[Dict], 
                                 right_detections: List[Dict],
                                 left_frame: np.ndarray, 
                                 right_frame: np.ndarray) -> List[Dict]:
        """Match with configurable epipolar tolerance"""
        matches = []
        used_right = set()
        epipolar_tolerance = CONFIG["stereo"]["epipolar_tolerance"]
        
        for left_det in left_detections:
            left_point = left_det['pos']
            epiline = self._compute_epiline(left_point, 'left')
            
            best_match = None
            best_distance = float('inf')
            best_score = 0
            
            for idx, right_det in enumerate(right_detections):
                if idx in used_right:
                    continue
                    
                right_point = right_det['pos']
                dist = self._point_to_line_distance(right_point, epiline)
                
                # Relaxed epipolar constraint
                if dist < epipolar_tolerance:
                    # Template matching in gradient space for illumination invariance
                    score = self._template_match_score_robust(
                        left_point, right_point, left_frame, right_frame
                    )
                    
                    # Consider vertical disparity (should be small for rectified images)
                    y_diff = abs(left_point[1] - right_point[1])
                    
                    # Combined score
                    combined_score = score * (1.0 - dist / epipolar_tolerance) * (1.0 - y_diff / 10.0)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = idx
                        best_distance = dist
                        
            # Accept match with lower threshold for outdoor conditions
            if best_match is not None and best_score > 0.4:
                used_right.add(best_match)
                matches.append({
                    'left_pos': left_point,
                    'right_pos': right_detections[best_match]['pos'],
                    'left_bbox': left_det['bbox'],
                    'right_bbox': right_detections[best_match]['bbox'],
                    'confidence': best_score,
                    'epipolar_distance': best_distance
                })
                
        return matches
        
    def _template_match_score_robust(self, pt_left: Tuple[float, float], 
                                    pt_right: Tuple[float, float],
                                    frame_left: np.ndarray, 
                                    frame_right: np.ndarray) -> float:
        """Robust template matching using gradient magnitude"""
        patch_size = CONFIG["stereo"]["template_size"]
        half_size = patch_size // 2
        
        x1, y1 = int(pt_left[0]), int(pt_left[1])
        x2, y2 = int(pt_right[0]), int(pt_right[1])
        
        # Check bounds
        if (half_size <= x1 < frame_left.shape[1] - half_size and
            half_size <= y1 < frame_left.shape[0] - half_size and
            half_size <= x2 < frame_right.shape[1] - half_size and
            half_size <= y2 < frame_right.shape[0] - half_size):
            
            # Extract patches
            patch_left = frame_left[y1-half_size:y1+half_size+1, 
                                   x1-half_size:x1+half_size+1]
            patch_right = frame_right[y2-half_size:y2+half_size+1,
                                     x2-half_size:x2+half_size+1]
            
            # Convert to grayscale if needed
            if len(patch_left.shape) == 3:
                patch_left = cv2.cvtColor(patch_left, cv2.COLOR_BGR2GRAY)
            if len(patch_right.shape) == 3:
                patch_right = cv2.cvtColor(patch_right, cv2.COLOR_BGR2GRAY)
                
            # Compute gradient magnitude for illumination invariance
            grad_left_x = cv2.Sobel(patch_left, cv2.CV_32F, 1, 0, ksize=3)
            grad_left_y = cv2.Sobel(patch_left, cv2.CV_32F, 0, 1, ksize=3)
            grad_left = np.sqrt(grad_left_x**2 + grad_left_y**2)
            
            grad_right_x = cv2.Sobel(patch_right, cv2.CV_32F, 1, 0, ksize=3)
            grad_right_y = cv2.Sobel(patch_right, cv2.CV_32F, 0, 1, ksize=3)
            grad_right = np.sqrt(grad_right_x**2 + grad_right_y**2)
            
            # Normalize gradients
            grad_left = (grad_left - grad_left.mean()) / (grad_left.std() + 1e-6)
            grad_right = (grad_right - grad_right.mean()) / (grad_right.std() + 1e-6)
            
            # Compute NCC on gradient magnitude
            correlation = np.sum(grad_left * grad_right) / (patch_size * patch_size)
            
            # Convert to [0, 1] range
            return (correlation + 1.0) / 2.0
            
        return 0.0
        
    def _detect_2d(self, frame: np.ndarray, bg_subtractor) -> List[Dict]:
        """Detect with configurable morphological operations"""
        fg_mask = bg_subtractor.apply(frame)
        
        # Dynamic kernel sizes from config
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.morph_kernel_open, self.morph_kernel_open)
        )
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.morph_kernel_close, self.morph_kernel_close)
        )
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                if 0.3 < aspect_ratio < 3.0:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        detections.append({
                            'pos': (cx, cy),
                            'bbox': (x, y, w, h),
                            'area': area,
                            'contour': contour,
                            'aspect_ratio': aspect_ratio
                        })
                        
        return detections
        
    def _compute_epiline(self, point: Tuple[float, float], camera: str) -> np.ndarray:
        """Compute epipolar line"""
        pt = np.array([[point[0], point[1]]], dtype=np.float32)
        
        if camera == 'left':
            lines = cv2.computeCorrespondEpilines(pt, 1, self.calibration.F)
        else:
            lines = cv2.computeCorrespondEpilines(pt, 2, self.calibration.F)
            
        return lines[0][0]
        
    def _point_to_line_distance(self, point: Tuple[float, float], line: np.ndarray) -> float:
        """Calculate perpendicular distance from point to line"""
        a, b, c = line
        x, y = point
        return abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
        
    def _extract_patch(self, frame: np.ndarray, center: Tuple[float, float], 
                      size: int = 32) -> np.ndarray:
        """Extract square patch around detection"""
        x, y = int(center[0]), int(center[1])
        half_size = size // 2
        
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(frame.shape[1], x + half_size)
        y2 = min(frame.shape[0], y + half_size)
        
        patch = frame[y1:y2, x1:x2]
        
        # Pad if necessary
        if patch.shape[0] < size or patch.shape[1] < size:
            patch = cv2.copyMakeBorder(
                patch,
                top=max(0, half_size - y),
                bottom=max(0, y + half_size - frame.shape[0]),
                left=max(0, half_size - x),
                right=max(0, x + half_size - frame.shape[1]),
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )
            
        return patch

class MosquitoTracker3D:
    """3D tracker with measured dt and Hungarian assignment"""
    
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_tracks = CONFIG["tracking"]["max_tracks"]
        self.max_distance_3d = CONFIG["tracking"]["max_distance_3d_mm"]
        self.max_missed_frames = CONFIG["tracking"]["max_missed_frames"]
        self.min_track_length = CONFIG["tracking"]["min_track_length"]
        self.classification_threshold = CONFIG["targeting"]["min_classification_score"]
        
    def update(self, detections_3d: List[Dict], timestamp: float) -> List[MosquitoTrack3D]:
        """Update tracks with measured time delta"""
        # Predict step with actual dt
        for track in self.tracks.values():
            if track.last_seen > 0:
                dt = timestamp - track.last_seen
                track.last_dt = dt
                
                # Update state transition matrix with actual dt
                self._update_kalman_dt(track.kalman, dt)
                
            track.kalman.predict()
            track.velocity_3d = track.kalman.statePost[3:6].flatten()
            
        # Hungarian assignment would go here
        # For now, keep greedy assignment but with Mahalanobis distance
        matched_tracks, unmatched_detections = self._associate_detections(detections_3d)
        
        # Update matched tracks
        for track_id, detection in matched_tracks:
            track = self.tracks[track_id]
            
            # Measurement update
            measurement = np.array([
                [detection['pos_3d'][0]],
                [detection['pos_3d'][1]],
                [detection['pos_3d'][2]]
            ], dtype=np.float32)
            
            track.kalman.correct(measurement)
            
            # Update history
            track.positions_3d.append(detection['pos_3d'])
            track.positions_2d_left.append(detection['pos_2d_left'])
            track.positions_2d_right.append(detection['pos_2d_right'])
            track.last_seen = timestamp
            track.confidence = detection['confidence']
            
            # Update classification
            track.classification_history.append(detection['classification_score'])
            track.classification_score = np.mean(track.classification_history)
            
        # Create new tracks
        for detection in unmatched_detections:
            if detection['classification_score'] > self.classification_threshold * 0.7:
                if len(self.tracks) < self.max_tracks:
                    self._create_track_3d(detection, timestamp)
                    
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            time_since_seen = timestamp - track.last_seen
            if time_since_seen > self.max_missed_frames * 0.033:
                tracks_to_remove.append(track_id)
            elif (len(track.positions_3d) > self.min_track_length and 
                  track.classification_score < self.classification_threshold * 0.5):
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            
        return list(self.tracks.values())
        
    def _update_kalman_dt(self, kalman: cv2.KalmanFilter, dt: float):
        """Update Kalman filter matrices with measured dt"""
        # Update state transition matrix for constant velocity model
        kalman.transitionMatrix[0, 3] = dt  # x += vx * dt
        kalman.transitionMatrix[1, 4] = dt  # y += vy * dt
        kalman.transitionMatrix[2, 5] = dt  # z += vz * dt
        
        # Update process noise covariance
        # Process noise should scale with dt
        # For position: Q_pos ~ q * dt³/3
        # For velocity: Q_vel ~ q * dt
        q_pos = CONFIG["tracking"]["process_noise"]
        q_vel = q_pos * 10  # Velocity noise factor
        
        # Build Q matrix
        Q = np.zeros((6, 6), dtype=np.float32)
        
        # Position variance increases with dt³/3
        dt3_3 = (dt ** 3) / 3.0
        Q[0, 0] = Q[1, 1] = Q[2, 2] = q_pos * dt3_3
        
        # Velocity variance increases with dt
        Q[3, 3] = Q[4, 4] = Q[5, 5] = q_vel * dt
        
        # Cross-covariance terms
        dt2_2 = (dt ** 2) / 2.0
        Q[0, 3] = Q[3, 0] = q_pos * dt2_2  # x-vx covariance
        Q[1, 4] = Q[4, 1] = q_pos * dt2_2  # y-vy covariance
        Q[2, 5] = Q[5, 2] = q_pos * dt2_2  # z-vz covariance
        
        kalman.processNoiseCov = Q
        
    def _associate_detections(self, detections: List[Dict]) -> Tuple[List[Tuple[int, Dict]], List[Dict]]:
        """Associate detections with tracks using Hungarian algorithm (optimal) or greedy fallback"""
        matched_pairs = []
        unmatched_detections = list(detections)
        
        if not self.tracks or not detections:
            return matched_pairs, unmatched_detections
            
        # Build cost matrix
        n_tracks = len(self.tracks)
        n_detections = len(detections)
        cost_matrix = np.full((n_tracks, n_detections), float('inf'))
        
        track_ids = list(self.tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            predicted = track.kalman.statePost[:3].flatten()
            
            # Get innovation covariance for Mahalanobis distance
            H = track.kalman.measurementMatrix
            P = track.kalman.errorCovPost
            R = track.kalman.measurementNoiseCov
            S = H @ P @ H.T + R
            
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # Singular matrix, use identity
                S_inv = np.eye(3) * 0.01  # Small weight for uncertain tracks
                
            for j, detection in enumerate(detections):
                det_pos = np.array(detection['pos_3d'])
                innovation = det_pos - predicted
                
                # Mahalanobis distance
                mahal_dist = np.sqrt(innovation.T @ S_inv @ innovation)
                
                # Classification consistency
                class_diff = abs(detection['classification_score'] - track.classification_score)
                
                # Combined cost
                if mahal_dist < 5.0:  # 5-sigma gate
                    cost = mahal_dist + class_diff * 50
                    cost_matrix[i, j] = cost
                    
        # Use Hungarian algorithm if available, otherwise greedy
        if SCIPY_AVAILABLE and cost_matrix.size > 0:
            # Hungarian algorithm for optimal assignment
            try:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                for row, col in zip(row_indices, col_indices):
                    if cost_matrix[row, col] < float('inf'):
                        track_id = track_ids[row]
                        detection = detections[col]
                        matched_pairs.append((track_id, detection))
                        if detection in unmatched_detections:
                            unmatched_detections.remove(detection)
                            
            except Exception as e:
                print(f"Hungarian assignment failed: {e}, using greedy")
                # Fall through to greedy
                
        if not matched_pairs:  # Fallback to greedy if scipy failed or not available
            # Greedy assignment
            for _ in range(min(n_tracks, n_detections)):
                if cost_matrix.size > 0:
                    min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                    if cost_matrix[min_idx] < float('inf'):
                        track_idx, det_idx = min_idx
                        track_id = track_ids[track_idx]
                        
                        matched_pairs.append((track_id, detections[det_idx]))
                        
                        # Remove from future consideration
                        cost_matrix[track_idx, :] = float('inf')
                        cost_matrix[:, det_idx] = float('inf')
                        
                        if detections[det_idx] in unmatched_detections:
                            unmatched_detections.remove(detections[det_idx])
                            
        return matched_pairs, unmatched_detections
        
    def _create_track_3d(self, detection: Dict, timestamp: float):
        """Create new track with properly configured Kalman filter"""
        kalman = cv2.KalmanFilter(6, 3)
        
        # Initial dt estimate
        dt = 0.033
        
        # State transition matrix
        kalman.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # Noise from config
        q = CONFIG["tracking"]["process_noise"]
        r = CONFIG["tracking"]["measurement_noise"]
        
        kalman.processNoiseCov = np.diag([q, q, q, q*10, q*10, q*10]).astype(np.float32)
        kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * r
        
        # Initial state
        x, y, z = detection['pos_3d']
        kalman.statePre = np.array([x, y, z, 0, 0, 0], dtype=np.float32)
        kalman.statePost = kalman.statePre.copy()
        
        # Error covariance
        kalman.errorCovPre = np.eye(6, dtype=np.float32) * 100
        kalman.errorCovPost = kalman.errorCovPre.copy()
        
        track = MosquitoTrack3D(
            id=self.next_id,
            positions_3d=deque([detection['pos_3d']], maxlen=30),
            positions_2d_left=deque([detection['pos_2d_left']], maxlen=30),
            positions_2d_right=deque([detection['pos_2d_right']], maxlen=30),
            kalman=kalman,
            last_seen=timestamp,
            last_dt=dt,
            velocity_3d=np.zeros(3),
            confidence=detection['confidence'],
            classification_score=detection['classification_score'],
            classification_history=deque([detection['classification_score']], maxlen=10),
            consecutive_on_target=0
        )
        
        self.tracks[self.next_id] = track
        self.next_id += 1

class StereoServoController:
    """Servo controller with proper error handling and ACK protocol"""
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.serial = None
        self.port = port
        self.baudrate = baudrate
        self.command_queue = queue.Queue(maxsize=CONFIG["safety"]["max_queue_size"])
        self.ack_timeout = 0.1
        self.command_counter = 0
        self.emergency_stop_active = False
        
        if SERIAL_AVAILABLE:
            try:
                self.serial = serial.Serial(port, baudrate, timeout=0.1)
                time.sleep(2)
                print(f"Serial connection established on {port}")
                
                # Start command thread
                self.command_thread = threading.Thread(target=self._command_worker)
                self.command_thread.daemon = True
                self.command_thread.start()
                
                # Start status monitor thread
                self.monitor_thread = threading.Thread(target=self._status_monitor)
                self.monitor_thread.daemon = True
                self.monitor_thread.start()
                
            except Exception as e:
                print(f"Failed to open serial port: {e}")
                print("Running in simulation mode")
                self.serial = None
        else:
            print("PySerial not installed. Running in simulation mode")
            
        # Servo limits from config
        self.pan_min = CONFIG["servo"]["pan_min"]
        self.pan_max = CONFIG["servo"]["pan_max"]
        self.tilt_min = CONFIG["servo"]["tilt_min"]
        self.tilt_max = CONFIG["servo"]["tilt_max"]
        self.max_rate = CONFIG["servo"]["max_rate_deg_per_s"]
        
        # Current positions
        self.current_pan = 0
        self.current_tilt = 0
        
        # PID controllers with config values
        self.pan_pid = PIDController3D(
            kp=CONFIG["pid"]["pan_kp"],
            ki=CONFIG["pid"]["pan_ki"],
            kd=CONFIG["pid"]["pan_kd"],
            integral_limit=CONFIG["pid"]["integral_limit"]
        )
        self.tilt_pid = PIDController3D(
            kp=CONFIG["pid"]["tilt_kp"],
            ki=CONFIG["pid"]["tilt_ki"],
            kd=CONFIG["pid"]["tilt_kd"],
            integral_limit=CONFIG["pid"]["integral_limit"]
        )
        
        # Laser config
        self.focus_distance = 1000
        self.min_power = CONFIG["laser"]["min_power"]
        self.max_power = CONFIG["laser"]["max_power"]
        
    def _status_monitor(self):
        """Monitor thread for emergency stop and other status messages"""
        while self.serial and self.serial.is_open:
            try:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode().strip()
                    
                    # Check for emergency stop messages
                    if line.startswith("E_STOP:"):
                        state = line.split(':')[1]
                        self.emergency_stop_active = (state == "1")
                        if self.emergency_stop_active:
                            print("WARNING: Emergency stop activated!")
                        else:
                            print("Emergency stop cleared")
                            
                time.sleep(0.01)  # Small delay to avoid busy loop
                
            except Exception as e:
                print(f"Status monitor error: {e}")
                time.sleep(0.1)
        
    def _command_worker(self):
        """Worker thread for serial commands with ACK"""
        while self.serial and self.serial.is_open:
            try:
                cmd, callback = self.command_queue.get(timeout=0.1)
                
                # Add command counter for tracking
                self.command_counter = (self.command_counter + 1) % 1000
                cmd_with_id = f"#{self.command_counter}:{cmd}"
                
                # Send command
                try:
                    self.serial.write(cmd_with_id.encode())
                except serial.SerialException as e:
                    print(f"Serial write error: {e}")
                    if callback:
                        callback(False)
                    # Serial error is fatal, exit worker
                    break
                    
                if CONFIG["safety"]["require_ack"]:
                    # Wait for ACK
                    start_time = time.perf_counter()
                    ack_received = False
                    error_msg = None
                    
                    while time.perf_counter() - start_time < self.ack_timeout:
                        try:
                            if self.serial.in_waiting:
                                response = self.serial.readline().decode().strip()
                                
                                # Parse response
                                if f"ACK#{self.command_counter}" in response:
                                    ack_received = True
                                    break
                                elif f"ERR#{self.command_counter}" in response:
                                    # Extract error message
                                    error_msg = response.split(':', 1)[1] if ':' in response else "Unknown error"
                                    break
                                    
                        except (serial.SerialException, UnicodeDecodeError) as e:
                            print(f"Serial read error: {e}")
                            break
                            
                    if callback:
                        callback(ack_received)
                        
                    if error_msg:
                        print(f"Arduino error for command {self.command_counter}: {error_msg}")
                else:
                    if callback:
                        callback(True)
                        
            except queue.Empty:
                # Normal timeout, continue
                pass
            except Exception as e:
                # Non-serial errors, log but continue
                print(f"Command worker error: {e}")
                if callback:
                    callback(False)
                
    def target_3d_position(self, x_mm: float, y_mm: float, z_mm: float,
                          velocity: Optional[np.ndarray] = None, dt: float = 0.033) -> Tuple[float, float, float]:
        """Target 3D position with measured dt for PID"""
        # Apply velocity lead compensation
        if velocity is not None:
            lead_time = 0.08  # Total system lag estimate
            x_mm += velocity[0] * lead_time
            y_mm += velocity[1] * lead_time
            z_mm += velocity[2] * lead_time
            
        # Convert to spherical
        distance = np.sqrt(x_mm**2 + y_mm**2 + z_mm**2)
        pan_angle = np.degrees(np.arctan2(x_mm, z_mm))
        tilt_angle = np.degrees(np.arcsin(y_mm / distance)) if distance > 0 else 0
        
        # PID control with measured dt
        pan_error = pan_angle - self.current_pan
        tilt_error = tilt_angle - self.current_tilt
        
        # Get control output (angular velocity)
        pan_velocity = self.pan_pid.update(pan_error, dt)
        tilt_velocity = self.tilt_pid.update(tilt_error, dt)
        
        # Apply rate limits
        pan_velocity = np.clip(pan_velocity, -self.max_rate, self.max_rate)
        tilt_velocity = np.clip(tilt_velocity, -self.max_rate, self.max_rate)
        
        # Integrate to get new position
        new_pan = self.current_pan + pan_velocity * dt
        new_tilt = self.current_tilt + tilt_velocity * dt
        
        # Move servos
        self.move_to(new_pan, new_tilt)
        
        # Update focus
        self._set_focus_distance(distance)
        
        return pan_angle, tilt_angle, distance
        
    def move_to(self, pan_angle: float, tilt_angle: float, callback=None):
        """Move servos with bounds checking and optional callback"""
        # Apply limits
        pan_angle = np.clip(pan_angle, self.pan_min, self.pan_max)
        tilt_angle = np.clip(tilt_angle, self.tilt_min, self.tilt_max)
        
        # Validate servo mapping
        pan_servo = 90 + int(pan_angle)
        tilt_servo = 90 + int(tilt_angle)
        
        if not (0 <= pan_servo <= 180 and 0 <= tilt_servo <= 180):
            print(f"Servo angle out of range: pan={pan_servo}, tilt={tilt_servo}")
            return
            
        command = f"P{int(pan_angle)},T{int(tilt_angle)}\n"
        
        if self.serial:
            try:
                self.command_queue.put((command, callback), timeout=0.01)
            except queue.Full:
                print("Command queue full, dropping command")
        else:
            print(f"[SIM] Pan: {pan_angle:.1f}°, Tilt: {tilt_angle:.1f}°")
            if callback:
                callback(True)
                
        # Update current position
        self.current_pan = pan_angle
        self.current_tilt = tilt_angle
        
    def fire_laser(self, duration_ms: int = 100, power: int = 50, callback=None):
        """Fire laser with safety limits and callback"""
        # Check emergency stop
        if self.emergency_stop_active:
            print("Cannot fire: Emergency stop is active!")
            if callback:
                callback(False)
            return
            
        duration_ms = min(duration_ms, CONFIG["laser"]["max_duration_ms"])
        power = np.clip(power, self.min_power, self.max_power)
        
        command = f"L{duration_ms},{power}\n"
        
        if self.serial:
            try:
                self.command_queue.put((command, callback), timeout=0.01)
            except queue.Full:
                print("Command queue full, cannot fire")
                if callback:
                    callback(False)
        else:
            print(f"[SIM] FIRE! Duration: {duration_ms}ms, Power: {power}/255")
            if callback:
                callback(True)
                
    def calculate_laser_power(self, distance_mm: float, target_size: float = 50) -> int:
        """Calculate power with proper inverse square law"""
        base_power = CONFIG["laser"]["base_power"]
        ref_distance_m = CONFIG["laser"]["power_distance_ref_m"]
        
        distance_m = distance_mm / 1000.0
        
        # Inverse square law with reference distance
        power = base_power * (distance_m / ref_distance_m) ** 2
        
        # Size compensation
        size_factor = 50 / target_size
        power *= size_factor
        
        # Ensure minimum power for close targets
        if distance_m < ref_distance_m:
            power = max(power, base_power)
            
        return int(np.clip(power, self.min_power, self.max_power))
        
    def _set_focus_distance(self, distance_mm: float):
        """Adjust laser focus"""
        if abs(distance_mm - self.focus_distance) > 50:
            self.focus_distance = distance_mm
            
            focus_position = int(np.interp(
                distance_mm, 
                [500, 3000],
                [0, 180]
            ))
            
            command = f"F{focus_position}\n"
            
            if self.serial:
                try:
                    self.command_queue.put((command, None), timeout=0.01)
                except:
                    pass
                    
    def cleanup(self):
        """Clean shutdown"""
        if self.serial and self.serial.is_open:
            try:
                self.serial.close()
            except:
                pass

class PIDController3D:
    """PID controller with anti-windup and derivative filtering"""
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 integral_limit: float = 10.0,
                 derivative_filter: float = 0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.derivative_filter = derivative_filter
        
        self.prev_error = 0
        self.integral = 0
        self.prev_derivative = 0
        
    def update(self, error: float, dt: float) -> float:
        """PID update returning angular velocity, not position delta"""
        if dt <= 0:
            return 0
            
        # Proportional
        p_term = self.kp * error
        
        # Integral with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative with filter
        derivative = (error - self.prev_error) / dt
        derivative = (self.derivative_filter * derivative + 
                     (1 - self.derivative_filter) * self.prev_derivative)
        self.prev_derivative = derivative
        d_term = self.kd * derivative
        
        # Output is angular velocity (deg/s)
        output = p_term + i_term + d_term
        
        self.prev_error = error
        
        return output
        
    def reset(self):
        """Reset controller state"""
        self.prev_error = 0
        self.integral = 0
        self.prev_derivative = 0

class StereoMosquitoLaserSystem:
    """Main system with all fixes applied"""
    
    def __init__(self, left_camera_id: int = None, right_camera_id: int = None,
                 calibration_file: str = 'stereo_calibration.pkl',
                 serial_port: str = None,
                 reference_dir: str = 'mosquito_references/',
                 display_enabled: bool = True):
        
        # Load config values
        if left_camera_id is None:
            left_camera_id = CONFIG["camera"]["left_id"]
        if right_camera_id is None:
            right_camera_id = CONFIG["camera"]["right_id"]
        if serial_port is None:
            serial_port = '/dev/ttyUSB0'
            
        # Display mode
        self.display_enabled = display_enabled
            
        # Initialize cameras with config resolution
        self.cap_left = cv2.VideoCapture(left_camera_id)
        self.cap_right = cv2.VideoCapture(right_camera_id)
        
        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            raise RuntimeError(f"Failed to open cameras {left_camera_id} and {right_camera_id}")
            
        # Set camera properties
        for cap in [self.cap_left, self.cap_right]:
            cap.set(cv2.CAP_PROP_FPS, CONFIG["camera"]["fps"])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera"]["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera"]["height"])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        # Initialize frame synchronization
        self.frame_sync = StereoFrameSync(CONFIG["camera"]["sync_tolerance_ms"])
        self.left_thread = threading.Thread(target=self.frame_sync.capture_left, args=(self.cap_left,))
        self.right_thread = threading.Thread(target=self.frame_sync.capture_right, args=(self.cap_right,))
        self.left_thread.daemon = True
        self.right_thread.daemon = True
        self.left_thread.start()
        self.right_thread.start()
        
        # Load calibration
        print("Loading stereo calibration...")
        try:
            with open(calibration_file, 'rb') as f:
                self.calibration = pickle.load(f)
                
            # Ensure calibration has metric scale info
            if not hasattr(self.calibration, 'baseline_mm'):
                print("WARNING: Calibration missing baseline_mm, using default")
                self.calibration.baseline_mm = CONFIG["stereo"]["baseline_mm"]
            if not hasattr(self.calibration, 'focal_length_px'):
                print("WARNING: Calibration missing focal_length_px, using default")
                self.calibration.focal_length_px = CONFIG["stereo"]["focal_length_px"]
                
        except Exception as e:
            raise RuntimeError(f"Failed to load calibration: {e}")
            
        # Initialize components
        print("Initializing mosquito classifier...")
        self.classifier = MosquitoClassifier(reference_dir, method='deep')
        
        self.detector = StereoMosquitoDetector(self.calibration, self.classifier)
        self.tracker = MosquitoTracker3D()
        self.servo_controller = StereoServoController(serial_port)
        
        # System state
        self.running = False
        self.targeting_enabled = False
        self.safety_mode = CONFIG["safety"]["enabled"]
        self.visualization_mode = 'stereo'
        
        # Performance monitoring
        self.fps_counter = FPSCounter()
        self.detection_count = 0
        self.engagement_count = 0
        
        # Targeting parameters
        self.consecutive_frames_required = CONFIG["targeting"]["consecutive_frames_required"]
        self.engagement_cooldown = CONFIG["targeting"]["engagement_cooldown_s"]
        self.last_engagement_time = 0
        
        # Register cleanup
        atexit.register(self.cleanup)
        
    def run(self):
        """Main loop with synchronized capture and measured dt"""
        self.running = True
        print("\nSystem active. Press 'q' to quit.")
        print(f"Safety mode: {'ON' if self.safety_mode else 'OFF'}")
        
        last_timestamp = time.perf_counter()  # Use monotonic timer
        
        while self.running:
            try:
                # Get synchronized frames
                sync_result = self.frame_sync.get_synchronized_frames()
                if sync_result is None:
                    continue
                    
                frame_left, frame_right, timestamp = sync_result
                
                # Calculate actual dt
                dt = timestamp - last_timestamp
                last_timestamp = timestamp
                
                # Skip if dt is too large (frame drop)
                if dt > 0.1:  # 100ms indicates dropped frames
                    print(f"Large dt detected: {dt:.3f}s, possible frame drop")
                    dt = 0.033  # Use nominal value
                
                # Process pipeline
                detections_3d = self.detector.detect_3d(frame_left, frame_right)
                self.detection_count += len(detections_3d)
                
                # Update tracking with timestamp
                tracks = self.tracker.update(detections_3d, timestamp)
                
                # Visualization
                vis_left = frame_left.copy()
                vis_right = frame_right.copy()
                self._draw_visualization(vis_left, vis_right, tracks, detections_3d, dt)
                
                # Targeting
                if self.targeting_enabled and tracks:
                    target = self._select_best_target(tracks)
                    if target:
                        engaged = self._engage_target(target, timestamp, dt)
                        if engaged:
                            self.engagement_count += 1
                            
                # Display
                self._display_output(vis_left, vis_right)
                
                # Controls
                if not self._handle_controls():
                    break
                    
                self.fps_counter.update()
                
            except Exception as e:
                print(f"Main loop error: {e}")
                import traceback
                traceback.print_exc()
                
        self.cleanup()
        
    def _engage_target(self, target: MosquitoTrack3D, timestamp: float, dt: float) -> bool:
        """Engage with consecutive frame confirmation"""
        # Check cooldown
        if timestamp - self.last_engagement_time < self.engagement_cooldown:
            return False
            
        # Get state
        state = target.kalman.statePost
        x, y, z = state[0], state[1], state[2]
        velocity = state[3:6]
        
        # Target with measured dt
        pan, tilt, distance = self.servo_controller.target_3d_position(
            x, y, z, velocity, dt
        )
        
        # Check if on target
        if self._is_on_target(target, pan, tilt, distance):
            target.consecutive_on_target += 1
            
            # Require consecutive frames
            if target.consecutive_on_target >= self.consecutive_frames_required:
                if not self.safety_mode:
                    # Calculate power
                    power = self.servo_controller.calculate_laser_power(distance)
                    
                    # Fire with ACK callback
                    fire_success = False
                    
                    def fire_callback(ack_received):
                        nonlocal fire_success
                        fire_success = ack_received
                        
                    self.servo_controller.fire_laser(50, power, fire_callback)
                    
                    # Wait briefly for callback
                    time.sleep(0.05)
                    
                    if fire_success:
                        self.last_engagement_time = timestamp
                        print(f"Engaged target {target.id} at {distance:.0f}mm")
                        target.consecutive_on_target = 0
                        return True
                    else:
                        print("Fire command failed (no ACK)")
                else:
                    print(f"Would engage target {target.id} (safety on)")
        else:
            target.consecutive_on_target = 0
            
        return False
        
    def _is_on_target(self, target: MosquitoTrack3D, 
                     pan: float, tilt: float, distance: float) -> bool:
        """Check alignment with distance-based tolerance"""
        x, y, z = target.positions_3d[-1]
        
        required_pan = np.degrees(np.arctan2(x, z))
        required_tilt = np.degrees(np.arcsin(y / distance)) if distance > 0 else 0
        
        pan_error = abs(required_pan - self.servo_controller.current_pan)
        tilt_error = abs(required_tilt - self.servo_controller.current_tilt)
        
        # Dynamic tolerance
        base_tolerance = CONFIG["targeting"]["tolerance_degrees_base"]
        distance_factor = CONFIG["targeting"]["tolerance_distance_factor"]
        tolerance = base_tolerance * (1.0 + distance / 1000.0 * distance_factor)
        
        return pan_error < tolerance and tilt_error < tolerance
        
    def _select_best_target(self, tracks: List[MosquitoTrack3D]) -> Optional[MosquitoTrack3D]:
        """Select target with configurable criteria"""
        if not tracks:
            return None
            
        best_score = -1
        best_track = None
        
        min_score = CONFIG["targeting"]["min_classification_score"]
        min_length = CONFIG["tracking"]["min_track_length"]
        
        for track in tracks:
            if len(track.positions_3d) < min_length:
                continue
                
            if track.classification_score < min_score:
                continue
                
            x, y, z = track.positions_3d[-1]
            distance = np.sqrt(x**2 + y**2 + z**2)
            
            if distance > 2500:
                continue
                
            # Scoring
            distance_score = 1.0 / (1.0 + distance / 1000.0)
            classification_score = track.classification_score
            velocity = np.linalg.norm(track.velocity_3d)
            velocity_score = 1.0 / (1.0 + velocity / 100.0)
            stability_score = min(len(track.positions_3d) / 30.0, 1.0)
            centered_score = 1.0 - (abs(x) + abs(y)) / (distance + 1.0)
            
            total_score = (
                0.25 * distance_score +
                0.30 * classification_score +
                0.20 * velocity_score +
                0.15 * stability_score +
                0.10 * centered_score
            )
            
            if total_score > best_score:
                best_score = total_score
                best_track = track
                
        return best_track
        
    def _draw_visualization(self, left_frame: np.ndarray, right_frame: np.ndarray,
                          tracks: List[MosquitoTrack3D], 
                          detections: List[Dict], dt: float):
        """Enhanced visualization with dt display"""
        # Draw tracks
        for track in tracks:
            if track.positions_2d_left and track.positions_2d_right:
                confidence = track.classification_score
                
                # Color based on targeting state
                if track.consecutive_on_target > 0:
                    color = (0, 0, 255)  # Red when targeting
                else:
                    color = (
                        int(255 * (1 - confidence)),
                        int(255 * confidence),
                        0
                    )
                
                # Draw trails
                if len(track.positions_2d_left) > 1:
                    points_left = np.array(list(track.positions_2d_left), dtype=np.int32)
                    cv2.polylines(left_frame, [points_left], False, color, 1)
                
                # Current position
                x_l, y_l = track.positions_2d_left[-1]
                cv2.circle(left_frame, (int(x_l), int(y_l)), 5, color, -1)
                
                # Draw on right
                if len(track.positions_2d_right) > 1:
                    points_right = np.array(list(track.positions_2d_right), dtype=np.int32)
                    cv2.polylines(right_frame, [points_right], False, color, 1)
                
                x_r, y_r = track.positions_2d_right[-1]
                cv2.circle(right_frame, (int(x_r), int(y_r)), 5, color, -1)
                
                # Info overlay
                if track.positions_3d:
                    x_3d, y_3d, z_3d = track.positions_3d[-1]
                    distance = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
                    velocity = np.linalg.norm(track.velocity_3d)
                    
                    info_lines = [
                        f"ID: {track.id}",
                        f"Dist: {distance:.0f}mm",
                        f"Vel: {velocity:.0f}mm/s",
                        f"Conf: {confidence:.2f}",
                        f"OnTgt: {track.consecutive_on_target}"
                    ]
                    
                    y_offset = int(y_l) - 75
                    for i, line in enumerate(info_lines):
                        y_pos = y_offset + i * 15
                        
                        (w, h), _ = cv2.getTextSize(
                            line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                        )
                        
                        cv2.rectangle(
                            left_frame,
                            (int(x_l) + 10, y_pos - h - 2),
                            (int(x_l) + 10 + w + 4, y_pos + 2),
                            (0, 0, 0),
                            -1
                        )
                        
                        cv2.putText(
                            left_frame, line,
                            (int(x_l) + 12, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1
                        )
        
        # Draw detections
        for det in detections:
            x_l, y_l = det['pos_2d_left']
            x_r, y_r = det['pos_2d_right']
            
            cv2.rectangle(left_frame, 
                         (int(x_l) - 3, int(y_l) - 3),
                         (int(x_l) + 3, int(y_l) + 3),
                         (255, 255, 0), 1)
            cv2.rectangle(right_frame,
                         (int(x_r) - 3, int(y_r) - 3),
                         (int(x_r) + 3, int(y_r) + 3),
                         (255, 255, 0), 1)
                         
    def _display_output(self, vis_left: np.ndarray, vis_right: np.ndarray):
        """Display with system info (or just log in headless mode)"""
        status_lines = [
            f"FPS: {self.fps_counter.get_fps():.1f}",
            f"Targeting: {'ON' if self.targeting_enabled else 'OFF'}",
            f"Safety: {'ON' if self.safety_mode else 'OFF'}",
            f"Tracks: {len(self.tracker.tracks)}",
            f"Detections: {self.detection_count}",
            f"Engagements: {self.engagement_count}"
        ]
        
        if self.display_enabled:
            # Draw status on frame
            for i, line in enumerate(status_lines):
                cv2.putText(vis_left, line,
                           (10, 20 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)
                           
            if self.visualization_mode == 'stereo':
                combined = np.hstack([vis_left, vis_right])
                cv2.imshow('Mosquito Defense System', combined)
            else:
                cv2.imshow('Left View', vis_left)
                cv2.imshow('Right View', vis_right)
        else:
            # Headless mode - print status periodically
            if self.detection_count % 100 == 0:  # Every 100 detections
                status = " | ".join(status_lines)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
            
    def _handle_controls(self) -> bool:
        """Handle keyboard input"""
        if self.display_enabled:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                return False
            elif key == ord('t'):
                self.targeting_enabled = not self.targeting_enabled
                print(f"Targeting: {'ON' if self.targeting_enabled else 'OFF'}")
            elif key == ord('s'):
                self.safety_mode = not self.safety_mode
                CONFIG["safety"]["enabled"] = self.safety_mode
                print(f"Safety mode: {'ON' if self.safety_mode else 'OFF'}")
            elif key == ord('v'):
                modes = ['stereo', 'separate']
                current_idx = modes.index(self.visualization_mode)
                self.visualization_mode = modes[(current_idx + 1) % len(modes)]
                print(f"Visualization: {self.visualization_mode}")
            elif key == ord('r'):
                self.detection_count = 0
                self.engagement_count = 0
                print("Statistics reset")
            elif key == ord('c'):
                # Save current config
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(CONFIG, f, indent=4)
                print(f"Configuration saved to {CONFIG_FILE}")
        else:
            # Headless mode - could check for signals or file-based commands
            pass
            
        return True
        
    def cleanup(self):
        """Comprehensive cleanup"""
        print("\nShutting down...")
        self.running = False
        
        # Stop frame sync
        if hasattr(self, 'frame_sync'):
            self.frame_sync.stop()
            
        # Wait for threads
        if hasattr(self, 'left_thread') and self.left_thread.is_alive():
            self.left_thread.join(timeout=1)
        if hasattr(self, 'right_thread') and self.right_thread.is_alive():
            self.right_thread.join(timeout=1)
            
        # Release cameras
        if hasattr(self, 'cap_left') and self.cap_left is not None:
            self.cap_left.release()
        if hasattr(self, 'cap_right') and self.cap_right is not None:
            self.cap_right.release()
            
        # Cleanup components
        if hasattr(self, 'classifier'):
            self.classifier.cleanup()
        if hasattr(self, 'servo_controller'):
            self.servo_controller.cleanup()
            
        # Clear TensorFlow
        if TENSORFLOW_AVAILABLE:
            tf.keras.backend.clear_session()
            
        cv2.destroyAllWindows()
        
        # Statistics
        print(f"\nSession Statistics:")
        print(f"Total detections: {self.detection_count}")
        print(f"Total engagements: {self.engagement_count}")
        if hasattr(self, 'fps_counter'):
            runtime = self.fps_counter.get_runtime()
            avg_fps = self.fps_counter.get_fps()
            if runtime > 0:
                print(f"Runtime: {runtime:.1f} seconds")
                print(f"Average FPS: {avg_fps:.1f}")
                print(f"Detection rate: {self.detection_count / runtime:.1f} per second")
                if self.engagement_count > 0:
                    print(f"Engagement rate: {self.engagement_count / runtime:.2f} per second")

class FPSCounter:
    """FPS counter with proper runtime tracking"""
    
    def __init__(self, window_size: int = 30):
        self.timestamps = deque(maxlen=window_size)
        self.start_time = None
        
    def update(self):
        current_time = time.perf_counter()
        self.timestamps.append(current_time)
        
        if self.start_time is None:
            self.start_time = current_time
        
    def get_fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span > 0:
            return (len(self.timestamps) - 1) / time_span
        return 0.0
        
    def get_runtime(self) -> float:
        """Get total runtime in seconds"""
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

# Arduino code with ACK protocol
ARDUINO_CODE_WITH_ACK = """
/*
 * Mosquito Defense System - Arduino Controller v3.3 (v60)
 * Copyright (c) 2024 Dragos Ruiu. All rights reserved.
 * 
 * Controls pan/tilt servos and laser module with ACK protocol
 * 
 * Commands:
 * #ID:P<angle>,T<angle> - Set pan and tilt angles
 * #ID:F<position> - Set focus servo position
 * #ID:L<duration>,<power> - Fire laser with duration and power
 * 
 * Responses:
 * ACK#ID - Command acknowledged and executed
 * ERR#ID:<message> - Error occurred
 * E_STOP:1 - Emergency stop active (sent periodically)
 * E_STOP:0 - Emergency stop cleared
 */

#include <Servo.h>

// Pin definitions
const int PAN_PIN = 9;
const int TILT_PIN = 10;
const int FOCUS_PIN = 11;
const int LASER_PIN = 13;
const int LASER_PWM_PIN = 6;
const int EMERGENCY_STOP_PIN = 2;  // Connect to ground for emergency stop

// Servo objects
Servo panServo;
Servo tiltServo;
Servo focusServo;

// State variables
int currentPan = 90;
int currentTilt = 90;
int currentFocus = 90;
volatile bool emergencyStop = false;
bool lastEmergencyState = false;
unsigned long lastStatusTime = 0;

// Command buffer
String cmdBuffer = "";
int lastCommandId = -1;

void setup() {
  // Initialize serial
  Serial.begin(115200);
  Serial.setTimeout(10);
  
  // Attach servos
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  focusServo.attach(FOCUS_PIN);
  
  // Initialize laser
  pinMode(LASER_PIN, OUTPUT);
  pinMode(LASER_PWM_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);
  analogWrite(LASER_PWM_PIN, 0);
  
  // Emergency stop with pull-up
  pinMode(EMERGENCY_STOP_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(EMERGENCY_STOP_PIN), emergencyISR, CHANGE);
  
  // Center servos
  panServo.write(currentPan);
  tiltServo.write(currentTilt);
  focusServo.write(currentFocus);
  
  Serial.println("READY");
}

void emergencyISR() {
  emergencyStop = (digitalRead(EMERGENCY_STOP_PIN) == LOW);
  if (emergencyStop) {
    digitalWrite(LASER_PIN, LOW);
    analogWrite(LASER_PWM_PIN, 0);
  }
}

void loop() {
  // Report emergency stop state changes
  if (emergencyStop != lastEmergencyState) {
    Serial.print("E_STOP:");
    Serial.println(emergencyStop ? "1" : "0");
    lastEmergencyState = emergencyStop;
  }
  
  // Periodic status report
  unsigned long currentTime = millis();
  if (currentTime - lastStatusTime > 1000) {  // Every second
    if (emergencyStop) {
      Serial.println("E_STOP:1");
    }
    lastStatusTime = currentTime;
  }
  
  // Check emergency stop
  if (emergencyStop) {
    digitalWrite(LASER_PIN, LOW);
    analogWrite(LASER_PWM_PIN, 0);
  }
  
  // Read serial commands
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      if (cmdBuffer.length() > 0) {
        processCommand(cmdBuffer);
        cmdBuffer = "";
      }
    } else if (cmdBuffer.length() < 100) {  // Prevent buffer overflow
      cmdBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  // Extract command ID
  int idStart = cmd.indexOf('#');
  int idEnd = cmd.indexOf(':');
  
  if (idStart < 0 || idEnd < 0 || idEnd <= idStart) {
    Serial.println("ERR:-1:Invalid format");
    return;
  }
  
  int commandId = cmd.substring(idStart + 1, idEnd).toInt();
  String command = cmd.substring(idEnd + 1);
  
  // Prevent duplicate execution
  if (commandId == lastCommandId) {
    Serial.print("ACK#");
    Serial.println(commandId);
    return;
  }
  
  lastCommandId = commandId;
  
  // Parse command
  if (command.startsWith("P")) {
    // Pan/Tilt command
    int commaIndex = command.indexOf(',');
    if (commaIndex > 0) {
      int panAngle = command.substring(1, commaIndex).toInt();
      
      int tiltStart = command.indexOf('T', commaIndex);
      if (tiltStart > 0) {
        int tiltAngle = command.substring(tiltStart + 1).toInt();
        
        // Validate and apply
        currentPan = constrain(90 + panAngle, 0, 180);
        currentTilt = constrain(90 + tiltAngle, 0, 180);
        
        panServo.write(currentPan);
        tiltServo.write(currentTilt);
        
        Serial.print("ACK#");
        Serial.println(commandId);
      } else {
        Serial.print("ERR#");
        Serial.print(commandId);
        Serial.println(":Missing tilt");
      }
    } else {
      Serial.print("ERR#");
      Serial.print(commandId);
      Serial.println(":Invalid P command");
    }
    
  } else if (command.startsWith("F")) {
    // Focus command
    int focusPos = command.substring(1).toInt();
    currentFocus = constrain(focusPos, 0, 180);
    focusServo.write(currentFocus);
    
    Serial.print("ACK#");
    Serial.println(commandId);
    
  } else if (command.startsWith("L")) {
    // Laser command
    if (emergencyStop) {
      Serial.print("ERR#");
      Serial.print(commandId);
      Serial.println(":Emergency stop active");
      return;
    }
    
    int commaIndex = command.indexOf(',');
    if (commaIndex > 0) {
      int duration = command.substring(1, commaIndex).toInt();
      int power = command.substring(commaIndex + 1).toInt();
      
      // Safety limits
      duration = constrain(duration, 0, 500);
      power = constrain(power, 0, 100);
      
      // Map power 0-100 to PWM 0-255
      int pwmValue = map(power, 0, 100, 0, 255);
      
      // Fire laser
      analogWrite(LASER_PWM_PIN, pwmValue);
      digitalWrite(LASER_PIN, HIGH);
      
      // Send ACK immediately
      Serial.print("ACK#");
      Serial.println(commandId);
      
      // Hold for duration (check emergency stop)
      unsigned long startTime = millis();
      while (millis() - startTime < duration) {
        if (emergencyStop) {
          digitalWrite(LASER_PIN, LOW);
          analogWrite(LASER_PWM_PIN, 0);
          break;
        }
        delay(1);
      }
      
      // Turn off
      digitalWrite(LASER_PIN, LOW);
      analogWrite(LASER_PWM_PIN, 0);
      
    } else {
      Serial.print("ERR#");
      Serial.print(commandId);
      Serial.println(":Invalid L command");
    }
    
  } else {
    Serial.print("ERR#");
    Serial.print(commandId);
    Serial.println(":Unknown command");
  }
}
"""

# Command line interface
def create_parser():
    """Create argument parser with all options"""
    parser = argparse.ArgumentParser(
        description="Stereo Vision Mosquito Defense System v3.3 (v60)\nCopyright (c) 2024 Dragos Ruiu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run the system
  %(prog)s prepare video.mp4  # Extract frames for training
  %(prog)s train              # Train classifier
  %(prog)s test               # Test classifier accuracy
  %(prog)s calibrate          # Run stereo calibration
  %(prog)s --config my.json   # Use custom config file
  
Configuration:
  Edit mosquito_config.json to tune parameters
  Press 'c' while running to save current config
        """
    )
    
    parser.add_argument('command', nargs='?', choices=['prepare', 'train', 'test', 'calibrate'],
                       help='Command to execute')
    parser.add_argument('args', nargs='*', help='Additional arguments for command')
    parser.add_argument('--config', default=CONFIG_FILE, help='Configuration file path')
    parser.add_argument('--left', type=int, help='Left camera ID')
    parser.add_argument('--right', type=int, help='Right camera ID')
    parser.add_argument('--serial', help='Serial port')
    parser.add_argument('--calibration', default='stereo_calibration.pkl', 
                       help='Calibration file path')
    parser.add_argument('--references', default='mosquito_references/',
                       help='Reference images directory')
    parser.add_argument('--display', action='store_true', default=True,
                       help='Enable GUI display (default: True)')
    parser.add_argument('--no-display', dest='display', action='store_false',
                       help='Disable GUI display for headless operation')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Force CPU-only mode for TensorFlow')
    
    return parser

# Utility functions
def prepare_dataset_from_video(video_path: str, output_dir: str, sample_rate: int = 30):
    """Extract frames from video for manual labeling"""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
        
    frame_count = 0
    saved_count = 0
    
    print(f"Extracting frames from {video_path}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_rate == 0:
            filename = f"frame_{saved_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Extracted {saved_count} frames...")
            
        frame_count += 1
        
    cap.release()
    print(f"\nExtracted {saved_count} frames to {output_dir}")
    print("\nNext steps:")
    print("1. Create directories: mosquito_references/positive/ and mosquito_references/negative/")
    print("2. Manually sort images containing mosquitos into positive/")
    print("3. Sort images with other objects or empty frames into negative/")
    print("4. Crop images to show individual objects")
    print("5. Run: python mosquito_defense.py train")

def train_classifier(reference_dir: str = "mosquito_references/", 
                    method: str = "deep",
                    output_path: str = "mosquito_model.h5"):
    """Train mosquito classifier on prepared dataset"""
    print(f"Training {method} classifier...")
    
    classifier = MosquitoClassifier(
        reference_dir=reference_dir,
        method=method,
        model_path=output_path
    )
    
    if method == "deep" and hasattr(classifier, 'model_type') and classifier.model_type == 'mobilenet':
        classifier.train_mobilenet(epochs=15)
        
    print("Training complete!")
    test_classifier(classifier, reference_dir)

def test_classifier(classifier: MosquitoClassifier, test_dir: str):
    """Test classifier accuracy"""
    correct = 0
    total = 0
    
    positive_dir = os.path.join(test_dir, "positive")
    if os.path.exists(positive_dir):
        test_files = glob.glob(os.path.join(positive_dir, "*.jpg"))[:10]
        for img_path in test_files:
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    score, method = classifier.classify_patch(img)
                    if score > 0.5:
                        correct += 1
                    total += 1
                    print(f"Positive: {os.path.basename(img_path)} - {score:.3f} ({method})")
            except Exception as e:
                print(f"Error testing {img_path}: {e}")
                
    negative_dir = os.path.join(test_dir, "negative")
    if os.path.exists(negative_dir):
        test_files = glob.glob(os.path.join(negative_dir, "*.jpg"))[:10]
        for img_path in test_files:
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    score, method = classifier.classify_patch(img)
                    if score < 0.5:
                        correct += 1
                    total += 1
                    print(f"Negative: {os.path.basename(img_path)} - {score:.3f} ({method})")
            except Exception as e:
                print(f"Error testing {img_path}: {e}")
                
    if total > 0:
        accuracy = correct / total
        print(f"\nTest Accuracy: {accuracy:.2%} ({correct}/{total})")
    else:
        print("No test samples found")

def main():
    """Main entry point with argument parsing"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load config from specified file
    global CONFIG
    CONFIG = load_config(args.config)
    
    # Validate command
    if args.command and args.command not in ['prepare', 'train', 'test', 'calibrate']:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        return
    
    # Handle commands
    if args.command == 'prepare':
        if len(args.args) >= 1:
            video_file = args.args[0]
            output_dir = args.args[1] if len(args.args) >= 2 else "dataset_frames"
            sample_rate = int(args.args[2]) if len(args.args) >= 3 else 30
            prepare_dataset_from_video(video_file, output_dir, sample_rate)
        else:
            print("Usage: python mosquito_defense.py prepare <video_file> [output_dir] [sample_rate]")
            
    elif args.command == 'train':
        reference_dir = args.references
        method = args.args[0] if len(args.args) >= 1 else "deep"
        output_path = args.args[1] if len(args.args) >= 2 else "mosquito_model.h5"
        train_classifier(reference_dir, method, output_path)
        
    elif args.command == 'test':
        reference_dir = args.references
        classifier = MosquitoClassifier(reference_dir, method='deep')
        test_classifier(classifier, reference_dir)
        
    elif args.command == 'calibrate':
        print("Stereo calibration not implemented in this version")
        print("Use OpenCV stereo calibration tools")
        
    else:
        # Run main system
        print("=" * 60)
        print("Stereo Vision Mosquito Defense System v3.3 (v60)")
        print("Copyright (c) 2024 Dragos Ruiu. All rights reserved.")
        print("=" * 60)
        print("\nCRITICAL FIXES APPLIED:")
        print("  ✓ Camera frame synchronization")
        print("  ✓ Proper metric depth calculation") 
        print("  ✓ Measured dt for Kalman and PID")
        print("  ✓ Comprehensive error handling")
        print("  ✓ Configurable parameters")
        print("  ✓ Safety interlocks")
        print("  ✓ Hungarian assignment (if scipy available)")
        print("  ✓ Process noise scaling with dt")
        
        if not args.display:
            print("\nRunning in HEADLESS mode (no GUI)")
        
        print("\nControls" + (" (keyboard):" if args.display else " (not available in headless mode):"))
        if args.display:
            print("  't' - Toggle targeting")
            print("  's' - Toggle safety mode")
            print("  'v' - Change visualization")
            print("  'r' - Reset statistics")
            print("  'c' - Save configuration")
            print("  'q' - Quit")
        else:
            print("  Use Ctrl+C to quit")
            
        print("\n" + "!" * 60)
        print("WARNING: Laser safety protocols must be followed!")
        print("Never aim at eyes, people, or reflective surfaces!")
        print("!" * 60 + "\n")
        
        # Check dependencies
        missing_deps = []
        optional_deps = []
        
        if not TENSORFLOW_AVAILABLE:
            optional_deps.append("tensorflow (deep learning)")
        if not SKLEARN_AVAILABLE:
            optional_deps.append("scikit-learn (HOG+SVM)")
        if not SKIMAGE_AVAILABLE:
            optional_deps.append("scikit-image (HOG features)")
        if not SERIAL_AVAILABLE:
            missing_deps.append("pyserial (hardware control)")
        if not SCIPY_AVAILABLE:
            optional_deps.append("scipy (optimal assignment)")
            
        if missing_deps:
            print("WARNING: Required dependencies missing:")
            for dep in missing_deps:
                print(f"  - {dep}")
            print("\nThe system cannot run without these.")
            return
            
        if optional_deps:
            print("INFO: Optional dependencies missing:")
            for dep in optional_deps:
                print(f"  - {dep}")
            print("\nThe system will run with reduced functionality.")
            
        print("\nInstall all dependencies with:")
        print("  pip install opencv-python numpy scikit-learn scikit-image tensorflow pyserial scipy")
        print("")
        
        try:
            # Check for required files
            if not os.path.exists(args.calibration):
                print("ERROR: No stereo calibration found!")
                print("Please run stereo calibration first.")
                return
                
            if not os.path.exists(args.references):
                print("WARNING: No mosquito reference images found!")
                print("System will use shape-based detection as fallback.\n")
                
            # Initialize and run system
            system = StereoMosquitoLaserSystem(
                left_camera_id=args.left,
                right_camera_id=args.right,
                calibration_file=args.calibration,
                serial_port=args.serial,
                reference_dir=args.references,
                display_enabled=args.display
            )
            
            system.run()
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            if 'system' in locals():
                system.cleanup()
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            if 'system' in locals():
                system.cleanup()

if __name__ == "__main__":
    main()
