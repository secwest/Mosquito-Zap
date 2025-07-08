#!/usr/bin/env python3
"""
Stereo Vision Mosquito Detection and Laser Targeting System with Object Recognition
Uses dual cameras for 3D positioning and deep learning for mosquito classification

QUICK START GUIDE:
1. Install dependencies:
   pip install opencv-python numpy scikit-learn scikit-image tensorflow pyserial

2. Prepare training data:
   python mosquito_defense.py prepare mosquito_video.mp4
   Then manually sort extracted frames into positive/negative folders

3. Train the classifier:
   python mosquito_defense.py train

4. Run the system:
   python mosquito_defense.py

REQUIREMENTS.TXT:
opencv-python>=4.5.0
numpy>=1.19.0
scikit-learn>=0.24.0
scikit-image>=0.18.0
tensorflow>=2.8.0
pyserial>=3.5
joblib>=1.0.0

Author: Assistant
Version: 3.0
"""

import cv2
import numpy as np
import serial
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import math
import pickle
import os
import glob
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import exposure
import joblib

@dataclass
class MosquitoTrack3D:
    """
    3D mosquito track with Kalman filtering and classification confidence
    
    Attributes:
        id: Unique identifier for this track
        positions_3d: Queue of (x,y,z) positions in millimeters
        positions_2d_left: Queue of (u,v) pixel positions in left camera
        positions_2d_right: Queue of (u,v) pixel positions in right camera
        kalman: OpenCV Kalman filter for state estimation
        last_seen: Timestamp of last detection
        velocity_3d: Current 3D velocity vector (mm/s)
        confidence: Stereo matching confidence (0-1)
        classification_score: Mosquito classification confidence (0-1)
        classification_history: Recent classification scores for stability
    """
    id: int
    positions_3d: deque  # (x, y, z) in mm
    positions_2d_left: deque  # (u, v) pixels in left image
    positions_2d_right: deque  # (u, v) pixels in right image
    kalman: cv2.KalmanFilter
    last_seen: float
    velocity_3d: np.ndarray
    confidence: float
    classification_score: float
    classification_history: deque

@dataclass
class StereoCalibration:
    """
    Complete stereo camera calibration parameters
    
    Contains all intrinsic and extrinsic parameters needed for:
    - Undistortion
    - Rectification
    - 3D triangulation
    - Disparity-to-depth conversion
    """
    camera_matrix_left: np.ndarray  # 3x3 intrinsic matrix for left camera
    camera_matrix_right: np.ndarray  # 3x3 intrinsic matrix for right camera
    dist_coeffs_left: np.ndarray  # Distortion coefficients for left camera
    dist_coeffs_right: np.ndarray  # Distortion coefficients for right camera
    R: np.ndarray  # 3x3 rotation matrix between cameras
    T: np.ndarray  # 3x1 translation vector between cameras
    E: np.ndarray  # 3x3 essential matrix
    F: np.ndarray  # 3x3 fundamental matrix
    rect_transform_left: np.ndarray  # 3x3 rectification transform for left
    rect_transform_right: np.ndarray  # 3x3 rectification transform for right
    projection_matrix_left: np.ndarray  # 3x4 projection matrix for left
    projection_matrix_right: np.ndarray  # 3x4 projection matrix for right
    Q: np.ndarray  # 4x4 disparity-to-depth mapping matrix
    roi_left: Tuple[int, int, int, int]  # Valid region of interest after rectification
    roi_right: Tuple[int, int, int, int]  # Valid region of interest after rectification

class MosquitoClassifier:
    """
    Machine learning classifier for distinguishing mosquitos from other flying objects
    
    Uses either:
    1. HOG (Histogram of Oriented Gradients) + SVM for traditional ML approach
    2. Template matching with reference images
    3. Deep learning with CNN (custom or pre-trained MobileNet)
    """
    
    def __init__(self, reference_dir: str = "mosquito_references/", 
                 method: str = "deep", model_path: str = None):
        """
        Initialize the mosquito classifier
        
        Args:
            reference_dir: Directory containing mosquito reference images
            method: Classification method ('hog_svm', 'template', 'deep')
            model_path: Path to pre-trained model (if available)
        """
        self.reference_dir = reference_dir
        self.method = method
        self.model_path = model_path or "mosquito_classifier.h5"
        
        # HOG feature extractor parameters
        # These are tuned for small flying objects
        self.hog_params = {
            'orientations': 9,  # Number of gradient orientations
            'pixels_per_cell': (4, 4),  # Size of cells for histogram computation
            'cells_per_block': (2, 2),  # Blocks for normalization
            'visualize': False,
            'feature_vector': True
        }
        
        # Initialize based on method
        if method == "hog_svm":
            self._init_hog_svm()
        elif method == "template":
            self._init_template_matching()
        elif method == "deep":
            self._init_deep_learning()
            
    def _init_mobilenet_transfer(self):
        """
        Initialize with pre-trained MobileNetV2 for transfer learning
        This is the most effective and easy-to-implement approach
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras import layers, Model
            
            print("Initializing MobileNetV2 transfer learning model...")
            
            # Load pre-trained MobileNetV2 (without top layers)
            base_model = MobileNetV2(
                input_shape=(96, 96, 3),  # Larger input for better features
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Add custom classification head
            inputs = layers.Input(shape=(96, 96, 3))
            
            # Preprocess for MobileNet
            x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
            
            # Base model feature extraction
            x = base_model(x, training=False)
            
            # Custom classification layers
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            # Output layer
            outputs = layers.Dense(1, activation='sigmoid')(x)
            
            # Create model
            self.classifier = Model(inputs, outputs)
            
            # Compile
            self.classifier.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Check if fine-tuned model exists
            fine_tuned_path = self.model_path.replace('.h5', '_mobilenet.h5')
            if os.path.exists(fine_tuned_path):
                print("Loading fine-tuned MobileNet model...")
                self.classifier.load_weights(fine_tuned_path)
            else:
                print("No fine-tuned model found. Use train_mobilenet() to fine-tune.")
                
            self.use_deep_learning = True
            self.model_type = 'mobilenet'
            
        except ImportError:
            print("TensorFlow not available, falling back to simple CNN")
            self._build_and_train_cnn()
            
    def _init_hog_svm(self):
        """Initialize HOG+SVM classifier"""
        # Load or train model
        if self.model_path and os.path.exists(self.model_path):
            # Load pre-trained model
            self.classifier = joblib.load(self.model_path)
            self.scaler = joblib.load(self.model_path.replace('.pkl', '_scaler.pkl'))
            print("Loaded pre-trained HOG+SVM model")
        else:
            # Train new model
            self._train_hog_svm()
            
    def _train_hog_svm(self):
        """
        Train HOG+SVM classifier on reference images
        
        Expects directory structure:
        mosquito_references/
            positive/  (contains mosquito images)
            negative/  (contains non-mosquito flying objects)
        """
        print("Training HOG+SVM classifier...")
        
        # Collect training data
        X_train = []
        y_train = []
        
        # Load positive samples (mosquitos)
        positive_dir = os.path.join(self.reference_dir, "positive")
        if os.path.exists(positive_dir):
            for img_path in glob.glob(os.path.join(positive_dir, "*.jpg")):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to standard size
                    img_resized = cv2.resize(img, (32, 32))
                    
                    # Extract HOG features
                    features = hog(img_resized, **self.hog_params)
                    X_train.append(features)
                    y_train.append(1)  # Positive class
                    
        # Load negative samples (other objects)
        negative_dir = os.path.join(self.reference_dir, "negative")
        if os.path.exists(negative_dir):
            for img_path in glob.glob(os.path.join(negative_dir, "*.jpg")):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (32, 32))
                    features = hog(img_resized, **self.hog_params)
                    X_train.append(features)
                    y_train.append(0)  # Negative class
                    
        if len(X_train) > 0:
            # Convert to numpy arrays
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Feature scaling
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train SVM with RBF kernel
            self.classifier = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
            self.classifier.fit(X_train_scaled, y_train)
            
            print(f"Trained on {len(X_train)} samples")
            
            # Save model if path specified
            if self.model_path:
                joblib.dump(self.classifier, self.model_path)
                joblib.dump(self.scaler, self.model_path.replace('.pkl', '_scaler.pkl'))
        else:
            print("No training data found! Using fallback classifier")
            self.classifier = None
            
    def _init_template_matching(self):
        """Initialize template matching with reference images"""
        self.templates = []
        
        # Load all mosquito templates
        template_dir = os.path.join(self.reference_dir, "templates")
        if os.path.exists(template_dir):
            for img_path in glob.glob(os.path.join(template_dir, "*.jpg")):
                template = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    # Store multiple scales of each template
                    for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
                        h, w = template.shape
                        new_h, new_w = int(h * scale), int(w * scale)
                        scaled_template = cv2.resize(template, (new_w, new_h))
                        self.templates.append(scaled_template)
                        
        print(f"Loaded {len(self.templates)} template variations")
        
    def _init_deep_learning(self):
        """
        Initialize deep learning model using TensorFlow/Keras
        Tries MobileNet first (best), then custom CNN, then falls back to HOG+SVM
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # First try MobileNet transfer learning (best option)
            try:
                self._init_mobilenet_transfer()
                return
            except Exception as e:
                print(f"MobileNet initialization failed: {e}")
                
            # Fall back to custom CNN
            from tensorflow.keras import layers
            
            # Check if pre-trained model exists
            if self.model_path and os.path.exists(self.model_path):
                print("Loading pre-trained CNN model...")
                self.classifier = keras.models.load_model(self.model_path)
                self.use_deep_learning = True
                self.model_type = 'custom_cnn'
            else:
                # Build and train new model
                print("Building new CNN model...")
                self._build_and_train_cnn()
                
        except ImportError:
            print("TensorFlow not installed. Install with: pip install tensorflow")
            print("Falling back to HOG+SVM")
            self._init_hog_svm()
            
    def _build_and_train_cnn(self):
        """
        Build and train a lightweight CNN for mosquito classification
        Architecture optimized for 32x32 grayscale images
        """
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Build model architecture
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(32, 32, 1)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),  # Better than flatten for small objects
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer (binary classification)
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        # Load training data
        X_train, y_train, X_val, y_val = self._load_training_data_cnn()
        
        if X_train is not None and len(X_train) > 0:
            # Data augmentation for better generalization
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            # Train model
            print(f"Training CNN on {len(X_train)} samples...")
            
            # Early stopping to prevent overfitting
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Train
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=32),
                epochs=20,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Save model
            if self.model_path:
                model.save(self.model_path)
                print(f"Model saved to {self.model_path}")
                
            self.classifier = model
            self.use_deep_learning = True
            self.model_type = 'custom_cnn'
            
            # Print final metrics
            val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val, y_val, verbose=0)
            print(f"Validation - Accuracy: {val_acc:.3f}, Precision: {val_prec:.3f}, Recall: {val_rec:.3f}")
            
        else:
            print("No training data found for CNN")
            self.classifier = None
            self.use_deep_learning = False
            
    def train_mobilenet(self, epochs: int = 10):
        """
        Fine-tune MobileNet on mosquito dataset
        Call this after initialization to adapt to your specific mosquitos
        """
        if not hasattr(self, 'model_type') or self.model_type != 'mobilenet':
            print("Initialize with method='deep' first")
            return
            
        import tensorflow as tf
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        # Load training data
        X = []
        y = []
        
        # Load and preprocess for MobileNet (96x96 RGB)
        positive_dir = os.path.join(self.reference_dir, "positive")
        if os.path.exists(positive_dir):
            for img_path in glob.glob(os.path.join(positive_dir, "*.jpg")):
                img = cv2.imread(img_path)  # Load as BGR
                if img is not None:
                    # Convert to RGB and resize
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (96, 96))
                    X.append(img_resized)
                    y.append(1)
                    
        negative_dir = os.path.join(self.reference_dir, "negative")
        if os.path.exists(negative_dir):
            for img_path in glob.glob(os.path.join(negative_dir, "*.jpg")):
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (96, 96))
                    X.append(img_resized)
                    y.append(0)
                    
        if len(X) > 10:  # Need minimum samples
            X = np.array(X, dtype=np.float32)
            y = np.array(y)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Data augmentation
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
            
            # Fine-tune
            print(f"Fine-tuning MobileNet on {len(X_train)} samples...")
            
            history = self.classifier.fit(
                datagen.flow(X_train, y_train, batch_size=16),
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
                ]
            )
            
            # Save fine-tuned model
            save_path = self.model_path.replace('.h5', '_mobilenet.h5')
            self.classifier.save_weights(save_path)
            print(f"Fine-tuned model saved to {save_path}")
            
            # Evaluate
            val_loss, val_acc = self.classifier.evaluate(X_val, y_val, verbose=0)
            print(f"Validation Accuracy: {val_acc:.3f}")
        else:
            print("Insufficient training data for fine-tuning")
            
    def _load_training_data_cnn(self):
        """
        Load and preprocess training data for CNN
        Returns train and validation sets
        """
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        X = []
        y = []
        
        # Load positive samples (mosquitos)
        positive_dir = os.path.join(self.reference_dir, "positive")
        if os.path.exists(positive_dir):
            for img_path in glob.glob(os.path.join(positive_dir, "*.jpg")):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize and normalize
                    img_resized = cv2.resize(img, (32, 32))
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    X.append(img_normalized)
                    y.append(1)
                    
        # Load negative samples
        negative_dir = os.path.join(self.reference_dir, "negative")
        if os.path.exists(negative_dir):
            for img_path in glob.glob(os.path.join(negative_dir, "*.jpg")):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (32, 32))
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    X.append(img_normalized)
                    y.append(0)
                    
        if len(X) > 0:
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Add channel dimension for CNN
            X = X.reshape(-1, 32, 32, 1)
            
            # Split into train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            return X_train, y_train, X_val, y_val
            
        return None, None, None, None
        
    def classify_patch(self, image_patch: np.ndarray) -> Tuple[float, str]:
        """
        Classify an image patch as mosquito or not
        
        Args:
            image_patch: Grayscale image patch containing potential mosquito
            
        Returns:
            confidence: Classification confidence (0-1)
            method_used: Classification method that was used
        """
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
            # Fallback: simple shape analysis
            return self._classify_shape(image_patch)
            
    def _classify_deep_learning(self, patch: np.ndarray) -> Tuple[float, str]:
        """
        Classify using deep learning model (MobileNet or custom CNN)
        Optimized for real-time inference
        """
        import numpy as np
        
        if hasattr(self, 'model_type') and self.model_type == 'mobilenet':
            # MobileNet expects RGB 96x96 input
            # Convert grayscale to RGB if needed
            if len(patch.shape) == 2:
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
            else:
                patch_rgb = patch
                
            # Resize to MobileNet input size
            patch_resized = cv2.resize(patch_rgb, (96, 96))
            
            # Add batch dimension
            patch_input = patch_resized.reshape(1, 96, 96, 3).astype(np.float32)
            
            # Run inference
            prediction = self.classifier.predict(patch_input, verbose=0)[0][0]
            
            # MobileNet tends to be well-calibrated
            confidence = float(prediction)
            
            return confidence, "mobilenet"
            
        else:
            # Custom CNN expects grayscale 32x32 input
            # Resize to model input size
            patch_resized = cv2.resize(patch, (32, 32))
            
            # Normalize pixel values
            patch_normalized = patch_resized.astype(np.float32) / 255.0
            
            # Add batch and channel dimensions
            patch_input = patch_normalized.reshape(1, 32, 32, 1)
            
            # Run inference
            prediction = self.classifier.predict(patch_input, verbose=0)[0][0]
            
            # Apply confidence threshold adjustment
            # CNN tends to be overconfident, so we scale the output
            confidence = float(prediction * 0.9 + 0.05)  # Slight dampening
            
            return confidence, "cnn"
        
    def _classify_hog_svm(self, patch: np.ndarray) -> Tuple[float, str]:
        """Classify using HOG features and SVM"""
        # Resize to standard size
        patch_resized = cv2.resize(patch, (32, 32))
        
        # Extract HOG features
        features = hog(patch_resized, **self.hog_params)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Get probability of being a mosquito
        proba = self.classifier.predict_proba(features_scaled)[0]
        confidence = proba[1]  # Probability of positive class
        
        return confidence, "hog_svm"
        
    def _classify_template(self, patch: np.ndarray) -> Tuple[float, str]:
        """Classify using template matching"""
        best_score = 0.0
        
        for template in self.templates:
            # Skip if template is larger than patch
            if template.shape[0] > patch.shape[0] or template.shape[1] > patch.shape[1]:
                continue
                
            # Perform template matching
            result = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            best_score = max(best_score, max_val)
            
        return best_score, "template"
        
    def _classify_shape(self, patch: np.ndarray) -> Tuple[float, str]:
        """
        Fallback classification based on shape characteristics
        Mosquitos typically have:
        - Elongated body
        - Small area
        - High aspect ratio variation due to wing movement
        """
        # Calculate shape metrics
        _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0, "shape"
            
        # Analyze largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        # Check area range (mosquitos are small)
        if area < 10 or area > 200:
            return 0.0, "shape"
            
        # Calculate shape features
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Fit ellipse if possible
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (MA, ma), angle = ellipse
            aspect_ratio = ma / MA if MA > 0 else 0
            
            # Mosquitos are elongated when wings are visible
            # But can appear round when wings are blurred
            if 0.3 < aspect_ratio < 0.8 and 0.4 < circularity < 0.9:
                confidence = 0.7  # Base confidence
                
                # Adjust based on size
                size_factor = 1.0 - abs(area - 50) / 150  # Peak at 50 pixels
                confidence *= max(0.5, size_factor)
                
                return confidence, "shape"
                
        return 0.2, "shape"  # Low confidence fallback

class StereoMosquitoDetector:
    """
    Detects and classifies mosquitos in stereo image pairs
    Combines motion detection, stereo matching, and object classification
    """
    
    def __init__(self, calibration: StereoCalibration, 
                 classifier: MosquitoClassifier,
                 min_area=10, max_area=200, 
                 min_disparity=16, max_disparity=96):
        """
        Initialize stereo mosquito detector
        
        Args:
            calibration: Stereo camera calibration data
            classifier: Mosquito classification model
            min_area: Minimum blob area in pixels
            max_area: Maximum blob area in pixels
            min_disparity: Minimum stereo disparity (affects max depth)
            max_disparity: Maximum stereo disparity (affects min depth)
        """
        self.calibration = calibration
        self.classifier = classifier
        self.min_area = min_area
        self.max_area = max_area
        
        # Background subtractors for motion detection
        # MOG2 is effective for detecting small moving objects
        self.bg_left = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,  # Helps filter out shadows
            varThreshold=16,  # Sensitivity to motion
            history=500  # Frames to consider for background model
        )
        self.bg_right = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )
        
        # Stereo block matching for depth estimation
        # SGBM provides better results than simple block matching
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=max_disparity - min_disparity,
            blockSize=5,  # Smaller block size for small objects
            P1=8 * 3 * 5**2,  # Penalty for small disparity changes
            P2=32 * 3 * 5**2,  # Penalty for large disparity changes
            disp12MaxDiff=1,  # Maximum allowed difference in left-right check
            uniquenessRatio=10,  # Margin for uniqueness
            speckleWindowSize=100,  # Filter out small noise blobs
            speckleRange=32  # Maximum disparity variation within blob
        )
        
        # Initialize rectification maps for stereo processing
        self._init_rectification()
        
    def _init_rectification(self):
        """
        Pre-compute rectification maps for real-time performance
        Rectification aligns stereo images so epipolar lines are horizontal
        """
        # Assume standard camera resolution
        h, w = 480, 640
        
        # Left camera rectification maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.calibration.camera_matrix_left,
            self.calibration.dist_coeffs_left,
            self.calibration.rect_transform_left,
            self.calibration.projection_matrix_left,
            (w, h), 
            cv2.CV_16SC2  # Use integer maps for speed
        )
        
        # Right camera rectification maps
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.calibration.camera_matrix_right,
            self.calibration.dist_coeffs_right,
            self.calibration.rect_transform_right,
            self.calibration.projection_matrix_right,
            (w, h), 
            cv2.CV_16SC2
        )
        
    def detect_3d(self, frame_left: np.ndarray, frame_right: np.ndarray) -> List[Dict]:
        """
        Main detection pipeline: detect, match, classify, and triangulate mosquitos
        
        Args:
            frame_left: Left camera frame (BGR)
            frame_right: Right camera frame (BGR)
            
        Returns:
            List of detection dictionaries containing:
                - pos_3d: (x,y,z) position in mm
                - pos_2d_left/right: Image coordinates
                - confidence: Stereo matching confidence
                - classification_score: Mosquito classification confidence
        """
        # Step 1: Rectify images to align epipolar lines
        rect_left = cv2.remap(frame_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(frame_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        
        # Step 2: Detect motion in both images independently
        detections_left = self._detect_2d(rect_left, self.bg_left)
        detections_right = self._detect_2d(rect_right, self.bg_right)
        
        # Step 3: Match detections between cameras using epipolar constraints
        matched_detections = self._match_stereo_detections(
            detections_left, detections_right, rect_left, rect_right
        )
        
        # Step 4: Classify matched detections and calculate 3D positions
        detections_3d = []
        for match in matched_detections:
            # Extract patches for classification
            patch_left = self._extract_patch(rect_left, match['left_pos'])
            
            # Classify the detection
            classification_score, method = self.classifier.classify_patch(patch_left)
            
            # Only proceed if classification confidence is sufficient
            if classification_score > 0.3:  # Threshold can be tuned
                # Triangulate 3D position
                x_3d, y_3d, z_3d = self._triangulate(
                    match['left_pos'], match['right_pos'],
                    self.calibration.projection_matrix_left,
                    self.calibration.projection_matrix_right
                )
                
                # Filter by realistic depth range for indoor use
                if 100 < z_3d < 3000:  # 10cm to 3m
                    detections_3d.append({
                        'pos_3d': (x_3d, y_3d, z_3d),
                        'pos_2d_left': match['left_pos'],
                        'pos_2d_right': match['right_pos'],
                        'confidence': match['confidence'],
                        'classification_score': classification_score,
                        'classification_method': method,
                        'patch_left': patch_left  # For visualization
                    })
                    
        return detections_3d
        
    def _detect_2d(self, frame: np.ndarray, bg_subtractor) -> List[Dict]:
        """
        Detect moving objects in a single camera frame
        
        Uses background subtraction followed by morphological operations
        and contour analysis to find mosquito-like objects
        """
        # Apply background subtraction to find moving pixels
        fg_mask = bg_subtractor.apply(frame)
        
        # Morphological operations to clean up the mask
        # Opening removes small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Closing fills small holes in detections
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_area < area < self.max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Mosquitos can have various aspect ratios due to wing positions
                if 0.3 < aspect_ratio < 3.0:
                    # Calculate center of mass for more accurate position
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
        
    def _match_stereo_detections(self, left_detections: List[Dict], 
                                 right_detections: List[Dict],
                                 left_frame: np.ndarray, 
                                 right_frame: np.ndarray) -> List[Dict]:
        """
        Match detections between stereo cameras using epipolar geometry
        
        For each detection in the left image:
        1. Compute its epipolar line in the right image
        2. Find right detections near this line
        3. Use template matching to verify correspondence
        """
        matches = []
        used_right = set()
        
        for left_det in left_detections:
            left_point = left_det['pos']
            
            # Compute epipolar line in right image
            # For point p_l in left image, corresponding point p_r in right image
            # must satisfy: p_r^T * F * p_l = 0 (epipolar constraint)
            epiline = self._compute_epiline(left_point, 'left')
            
            best_match = None
            best_distance = float('inf')
            best_score = 0
            
            for idx, right_det in enumerate(right_detections):
                if idx in used_right:
                    continue
                    
                right_point = right_det['pos']
                
                # Calculate distance from point to epipolar line
                dist = self._point_to_line_distance(right_point, epiline)
                
                # Points should be near the epipolar line (within a few pixels)
                if dist < 5:  # Threshold in pixels
                    # Additional verification using template matching
                    # This helps reject false matches on the epipolar line
                    score = self._template_match_score(
                        left_point, right_point, left_frame, right_frame
                    )
                    
                    # Also check y-coordinate similarity (for rectified images)
                    y_diff = abs(left_point[1] - right_point[1])
                    
                    # Combined score considering epipolar distance and appearance
                    combined_score = score * (1.0 - dist / 5.0) * (1.0 - y_diff / 10.0)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = idx
                        best_distance = dist
                        
            # Accept match if confidence is sufficient
            if best_match is not None and best_score > 0.5:
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
        
    def _compute_epiline(self, point: Tuple[float, float], camera: str) -> np.ndarray:
        """
        Compute epipolar line for a point in one image
        
        The fundamental matrix F encodes the epipolar geometry:
        For point p in first image, the epipolar line l' in second image is: l' = F^T * p
        """
        # Convert point to homogeneous coordinates
        pt = np.array([[point[0], point[1]]], dtype=np.float32)
        
        # Compute epipolar lines
        # Side 1 = left camera, Side 2 = right camera
        if camera == 'left':
            lines = cv2.computeCorrespondEpilines(pt, 1, self.calibration.F)
        else:
            lines = cv2.computeCorrespondEpilines(pt, 2, self.calibration.F)
            
        return lines[0][0]  # Returns line as [a, b, c] where ax + by + c = 0
        
    def _point_to_line_distance(self, point: Tuple[float, float], line: np.ndarray) -> float:
        """
        Calculate perpendicular distance from point to line
        Line equation: ax + by + c = 0
        Distance = |ax + by + c| / sqrt(a² + b²)
        """
        a, b, c = line
        x, y = point
        return abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
        
    def _template_match_score(self, pt_left: Tuple[float, float], 
                             pt_right: Tuple[float, float],
                             frame_left: np.ndarray, 
                             frame_right: np.ndarray) -> float:
        """
        Compare image patches around matched points using normalized cross-correlation
        This helps verify that matched points actually show the same object
        """
        # Define patch size (should be large enough to capture mosquito features)
        patch_size = 21  # Odd number for symmetry
        half_size = patch_size // 2
        
        # Extract patches around detected points
        x1, y1 = int(pt_left[0]), int(pt_left[1])
        x2, y2 = int(pt_right[0]), int(pt_right[1])
        
        # Ensure patches are within image bounds
        if (half_size <= x1 < frame_left.shape[1] - half_size and
            half_size <= y1 < frame_left.shape[0] - half_size and
            half_size <= x2 < frame_right.shape[1] - half_size and
            half_size <= y2 < frame_right.shape[0] - half_size):
            
            # Extract patches
            patch_left = frame_left[y1-half_size:y1+half_size+1, 
                                   x1-half_size:x1+half_size+1]
            patch_right = frame_right[y2-half_size:y2+half_size+1,
                                     x2-half_size:x2+half_size+1]
            
            # Convert to grayscale for matching
            if len(patch_left.shape) == 3:
                patch_left = cv2.cvtColor(patch_left, cv2.COLOR_BGR2GRAY)
            if len(patch_right.shape) == 3:
                patch_right = cv2.cvtColor(patch_right, cv2.COLOR_BGR2GRAY)
            
            # Normalized cross-correlation
            # Returns values between -1 and 1, where 1 is perfect match
            result = cv2.matchTemplate(patch_left, patch_right, cv2.TM_CCOEFF_NORMED)
            return float(result[0][0])
            
        return 0.0
        
    def _extract_patch(self, frame: np.ndarray, center: Tuple[float, float], 
                      size: int = 32) -> np.ndarray:
        """
        Extract square patch around detection for classification
        Handles boundary cases by padding if necessary
        """
        x, y = int(center[0]), int(center[1])
        half_size = size // 2
        
        # Calculate patch boundaries
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(frame.shape[1], x + half_size)
        y2 = min(frame.shape[0], y + half_size)
        
        # Extract patch
        patch = frame[y1:y2, x1:x2]
        
        # Pad if necessary to maintain consistent size
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
        
    def _triangulate(self, pt_left: Tuple[float, float], 
                    pt_right: Tuple[float, float],
                    P1: np.ndarray, P2: np.ndarray) -> Tuple[float, float, float]:
        """
        Triangulate 3D point from stereo correspondence
        
        Uses Direct Linear Transform (DLT) to find 3D point that projects
        to the given 2D points in both cameras
        
        Returns 3D position in millimeters
        """
        # Convert points to homogeneous coordinates
        pts_left = np.array([[pt_left[0], pt_left[1]]], dtype=np.float32)
        pts_right = np.array([[pt_right[0], pt_right[1]]], dtype=np.float32)
        
        # Triangulate using OpenCV
        # Returns 4D homogeneous coordinates
        points_4d = cv2.triangulatePoints(P1, P2, pts_left.T, pts_right.T)
        
        # Convert to 3D by dividing by w coordinate
        points_3d = points_4d[:3] / points_4d[3]
        
        # Return as regular floats in millimeters
        return float(points_3d[0]), float(points_3d[1]), float(points_3d[2])

class MosquitoTracker3D:
    """
    3D multi-object tracker using Kalman filters
    Maintains consistent IDs across frames and predicts future positions
    """
    
    def __init__(self, max_tracks: int = 10, 
                 classification_threshold: float = 0.5):
        """
        Initialize 3D tracker
        
        Args:
            max_tracks: Maximum number of simultaneous tracks
            classification_threshold: Minimum classification score to maintain track
        """
        self.tracks = {}  # Active tracks indexed by ID
        self.next_id = 0  # Next available track ID
        self.max_tracks = max_tracks
        self.classification_threshold = classification_threshold
        
        # Tracking parameters
        self.max_distance_3d = 100  # Maximum distance for matching (mm)
        self.max_missed_frames = 10  # Frames before track is deleted
        self.min_track_length = 5  # Minimum detections to establish track
        
    def update(self, detections_3d: List[Dict]) -> List[MosquitoTrack3D]:
        """
        Update tracks with new detections using Hungarian algorithm
        
        Steps:
        1. Predict next position for all tracks
        2. Associate detections with existing tracks
        3. Update matched tracks
        4. Create new tracks for unmatched detections
        5. Delete old tracks
        """
        current_time = time.time()
        
        # Step 1: Predict next position for all existing tracks
        for track in self.tracks.values():
            track.kalman.predict()
            # Update velocity from Kalman state
            track.velocity_3d = track.kalman.statePost[3:6].flatten()
            
        # Step 2: Data association using nearest neighbor
        # (Could be upgraded to Hungarian algorithm for global optimization)
        unmatched_detections = list(detections_3d)
        matched_tracks = set()
        
        # Build cost matrix for assignment
        for detection in detections_3d:
            best_track = None
            best_distance = float('inf')
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                    
                # Calculate 3D Euclidean distance
                predicted = track.kalman.statePost[:3].flatten()
                det_pos = detection['pos_3d']
                distance = np.linalg.norm(np.array(det_pos) - predicted)
                
                # Consider classification consistency
                classification_diff = abs(
                    detection['classification_score'] - track.classification_score
                )
                
                # Combined cost
                cost = distance + classification_diff * 50  # Weight classification
                
                if cost < best_distance and distance < self.max_distance_3d:
                    best_distance = cost
                    best_track = track_id
                    
            # Step 3: Update matched track
            if best_track is not None:
                track = self.tracks[best_track]
                
                # Create measurement vector
                measurement = np.array([
                    [detection['pos_3d'][0]],
                    [detection['pos_3d'][1]],
                    [detection['pos_3d'][2]]
                ], dtype=np.float32)
                
                # Kalman correction step
                track.kalman.correct(measurement)
                
                # Update track history
                track.positions_3d.append(detection['pos_3d'])
                track.positions_2d_left.append(detection['pos_2d_left'])
                track.positions_2d_right.append(detection['pos_2d_right'])
                track.last_seen = current_time
                track.confidence = detection['confidence']
                
                # Update classification with moving average
                track.classification_history.append(detection['classification_score'])
                track.classification_score = np.mean(track.classification_history)
                
                matched_tracks.add(best_track)
                unmatched_detections.remove(detection)
                
        # Step 4: Create new tracks for unmatched detections
        for detection in unmatched_detections:
            # Only create track if classification is confident
            if (detection['classification_score'] > self.classification_threshold and 
                len(self.tracks) < self.max_tracks):
                self._create_track_3d(detection, current_time)
                
        # Step 5: Remove old or low-confidence tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            # Check if track is too old
            if current_time - track.last_seen > self.max_missed_frames * 0.033:
                tracks_to_remove.append(track_id)
            # Check if classification confidence dropped too low
            elif (len(track.positions_3d) > self.min_track_length and 
                  track.classification_score < self.classification_threshold * 0.7):
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            
        return list(self.tracks.values())
        
    def _create_track_3d(self, detection: Dict, timestamp: float):
        """
        Initialize new 3D Kalman filter track
        
        State vector: [x, y, z, vx, vy, vz]
        Measurement vector: [x, y, z]
        
        Uses constant velocity motion model
        """
        # Create Kalman filter with 6 states and 3 measurements
        kalman = cv2.KalmanFilter(6, 3)
        
        # State transition matrix (constant velocity model)
        # x(t) = x(t-1) + vx * dt
        dt = 0.033  # Assume 30 FPS
        kalman.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],   # x = x + vx*dt
            [0, 1, 0, 0, dt, 0],   # y = y + vy*dt
            [0, 0, 1, 0, 0, dt],   # z = z + vz*dt
            [0, 0, 0, 1, 0, 0],    # vx = vx
            [0, 0, 0, 0, 1, 0],    # vy = vy
            [0, 0, 0, 0, 0, 1]     # vz = vz
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position, not velocity)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0]   # z
        ], dtype=np.float32)
        
        # Process noise covariance
        # Higher values = less trust in motion model
        q = 0.1  # Position noise
        qv = 1.0  # Velocity noise (mosquitos can change direction quickly)
        kalman.processNoiseCov = np.diag([q, q, q, qv, qv, qv]).astype(np.float32)
        
        # Measurement noise covariance
        # Based on stereo reconstruction accuracy
        r = 2.0  # Position measurement noise in mm
        kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * r
        
        # Initial state (position with zero velocity)
        x, y, z = detection['pos_3d']
        kalman.statePre = np.array([x, y, z, 0, 0, 0], dtype=np.float32)
        kalman.statePost = kalman.statePre.copy()
        
        # Error covariance initialization
        kalman.errorCovPre = np.eye(6, dtype=np.float32) * 1000
        kalman.errorCovPost = kalman.errorCovPre.copy()
        
        # Create track object
        track = MosquitoTrack3D(
            id=self.next_id,
            positions_3d=deque([detection['pos_3d']], maxlen=30),  # Keep last second
            positions_2d_left=deque([detection['pos_2d_left']], maxlen=30),
            positions_2d_right=deque([detection['pos_2d_right']], maxlen=30),
            kalman=kalman,
            last_seen=timestamp,
            velocity_3d=np.zeros(3),
            confidence=detection['confidence'],
            classification_score=detection['classification_score'],
            classification_history=deque([detection['classification_score']], maxlen=10)
        )
        
        self.tracks[self.next_id] = track
        self.next_id += 1

class StereoServoController:
    """
    Enhanced servo controller for 3D targeting with predictive aiming
    Includes laser focus control and power modulation
    """
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        """Initialize servo controller with serial connection"""
        self.serial = serial.Serial(port, baudrate, timeout=0.1)
        time.sleep(2)  # Wait for Arduino reset
        
        # Servo configuration (degrees)
        self.pan_min, self.pan_max = -90, 90
        self.tilt_min, self.tilt_max = -45, 45
        
        # Current servo positions
        self.current_pan = 0
        self.current_tilt = 0
        
        # PID controllers for smooth motion
        self.pan_pid = PIDController3D(
            kp=0.8,   # Proportional gain
            ki=0.2,   # Integral gain
            kd=0.3    # Derivative gain
        )
        self.tilt_pid = PIDController3D(
            kp=0.8, 
            ki=0.2, 
            kd=0.3
        )
        
        # Laser configuration
        self.focus_distance = 1000  # Current focus distance in mm
        self.min_power = 10  # Minimum laser power (0-255)
        self.max_power = 100  # Maximum safe laser power
        
    def target_3d_position(self, x_mm: float, y_mm: float, z_mm: float,
                          velocity: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """
        Calculate and move to required angles for 3D target position
        
        Args:
            x_mm: Lateral position (right positive)
            y_mm: Vertical position (up positive)  
            z_mm: Depth position (forward positive)
            velocity: Optional 3D velocity for lead compensation
            
        Returns:
            pan_angle: Required pan angle in degrees
            tilt_angle: Required tilt angle in degrees
            distance: Distance to target in mm
        """
        # Apply velocity-based lead compensation if provided
        if velocity is not None:
            # Estimate total system lag
            servo_lag = 0.05  # 50ms mechanical response
            processing_lag = 0.033  # One frame delay
            total_lag = servo_lag + processing_lag
            
            # Predict future position
            x_mm += velocity[0] * total_lag
            y_mm += velocity[1] * total_lag
            z_mm += velocity[2] * total_lag
        
        # Convert Cartesian to spherical coordinates
        distance = np.sqrt(x_mm**2 + y_mm**2 + z_mm**2)
        
        # Pan angle (horizontal rotation)
        # atan2 handles all quadrants correctly
        pan_angle = np.degrees(np.arctan2(x_mm, z_mm))
        
        # Tilt angle (vertical rotation)
        # Prevent division by zero
        if distance > 0:
            tilt_angle = np.degrees(np.arcsin(y_mm / distance))
        else:
            tilt_angle = 0
        
        # Apply PID control for smooth motion
        dt = 0.033  # Assume 30Hz update rate
        
        # Calculate control signals
        pan_error = pan_angle - self.current_pan
        tilt_error = tilt_angle - self.current_tilt
        
        pan_output = self.pan_pid.update(pan_error, dt)
        tilt_output = self.tilt_pid.update(tilt_error, dt)
        
        # Update positions with rate limiting
        max_rate = 180  # degrees per second
        pan_output = np.clip(pan_output, -max_rate * dt, max_rate * dt)
        tilt_output = np.clip(tilt_output, -max_rate * dt, max_rate * dt)
        
        new_pan = self.current_pan + pan_output
        new_tilt = self.current_tilt + tilt_output
        
        # Command servos
        self.move_to(new_pan, new_tilt)
        
        # Adjust laser focus for target distance
        self._set_focus_distance(distance)
        
        return pan_angle, tilt_angle, distance
        
    def _set_focus_distance(self, distance_mm: float):
        """
        Adjust laser focus based on target distance
        Maps distance to focus servo position
        """
        # Only update if distance changed significantly
        if abs(distance_mm - self.focus_distance) > 50:  # 5cm threshold
            self.focus_distance = distance_mm
            
            # Map distance to servo position
            # Assumes focus servo calibrated for 500-3000mm range
            focus_position = int(np.interp(
                distance_mm, 
                [500, 3000],  # Distance range
                [0, 180]      # Servo range
            ))
            
            command = f"F{focus_position}\n"
            self.serial.write(command.encode())
        
    def move_to(self, pan_angle: float, tilt_angle: float):
        """
        Move servos to specified angles with bounds checking
        """
        # Apply limits
        pan_angle = np.clip(pan_angle, self.pan_min, self.pan_max)
        tilt_angle = np.clip(tilt_angle, self.tilt_min, self.tilt_max)
        
        # Send command to Arduino
        # Format: P<pan>,T<tilt>
        command = f"P{int(pan_angle)},T{int(tilt_angle)}\n"
        self.serial.write(command.encode())
        
        # Update current position
        self.current_pan = pan_angle
        self.current_tilt = tilt_angle
        
    def fire_laser(self, duration_ms: int = 100, power: int = 50):
        """
        Activate laser with specified duration and power
        
        Args:
            duration_ms: Pulse duration in milliseconds
            power: Laser power (0-255 PWM value)
        """
        # Safety limits
        duration_ms = min(duration_ms, 500)  # Max 500ms
        power = np.clip(power, self.min_power, self.max_power)
        
        # Send fire command
        # Format: L<duration>,<power>
        command = f"L{duration_ms},{power}\n"
        self.serial.write(command.encode())
        
    def calculate_laser_power(self, distance_mm: float, 
                            target_size: float = 50) -> int:
        """
        Calculate appropriate laser power based on distance and target size
        
        Uses inverse square law with adjustments for mosquito size
        """
        # Base power for 1 meter distance
        base_power = 50
        
        # Inverse square law compensation
        distance_m = distance_mm / 1000.0
        power = base_power * (distance_m ** 2)
        
        # Adjust for target size (smaller targets need more precision)
        size_factor = 50 / target_size  # Normalized to 50 pixel reference
        power *= size_factor
        
        # Add minimum power to ensure effectiveness
        power = max(power, self.min_power)
        
        return int(np.clip(power, self.min_power, self.max_power))

class PIDController3D:
    """
    PID controller with anti-windup and derivative filtering
    Tuned for 3D servo control with mosquito tracking
    """
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 integral_limit: float = 10.0,
                 derivative_filter: float = 0.1):
        """
        Initialize PID controller
        
        Args:
            kp: Proportional gain
            ki: Integral gain  
            kd: Derivative gain
            integral_limit: Anti-windup limit
            derivative_filter: Low-pass filter coefficient for derivative
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.derivative_filter = derivative_filter
        
        # State variables
        self.prev_error = 0
        self.integral = 0
        self.prev_derivative = 0
        
    def update(self, error: float, dt: float) -> float:
        """
        Calculate PID output with anti-windup and filtered derivative
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term with filtering
        if dt > 0:
            derivative = (error - self.prev_error) / dt
            # Low-pass filter to reduce noise
            derivative = (self.derivative_filter * derivative + 
                         (1 - self.derivative_filter) * self.prev_derivative)
            self.prev_derivative = derivative
        else:
            derivative = 0
            
        d_term = self.kd * derivative
        
        # Calculate total output
        output = p_term + i_term + d_term
        
        # Update state
        self.prev_error = error
        
        return output
        
    def reset(self):
        """Reset controller state"""
        self.prev_error = 0
        self.integral = 0
        self.prev_derivative = 0

class StereoMosquitoLaserSystem:
    """
    Main system controller integrating all components
    Handles initialization, main loop, targeting logic, and visualization
    """
    
    def __init__(self, left_camera_id: int = 0, right_camera_id: int = 1,
                 calibration_file: str = 'stereo_calibration.pkl',
                 serial_port: str = '/dev/ttyUSB0',
                 reference_dir: str = 'mosquito_references/'):
        """
        Initialize complete mosquito targeting system
        
        Args:
            left_camera_id: Camera index for left stereo camera
            right_camera_id: Camera index for right stereo camera
            calibration_file: Path to stereo calibration data
            serial_port: Serial port for Arduino connection
            reference_dir: Directory containing mosquito reference images
        """
        # Initialize stereo cameras
        self.cap_left = cv2.VideoCapture(left_camera_id)
        self.cap_right = cv2.VideoCapture(right_camera_id)
        
        # Configure cameras for high-speed capture
        for cap in [self.cap_left, self.cap_right]:
            cap.set(cv2.CAP_PROP_FPS, 60)  # 60 FPS for better tracking
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
        # Load stereo calibration
        print("Loading stereo calibration...")
        with open(calibration_file, 'rb') as f:
            self.calibration = pickle.load(f)
            
        # Initialize mosquito classifier
        print("Initializing mosquito classifier...")
        self.classifier = MosquitoClassifier(
            reference_dir=reference_dir,
            method='deep'  # Deep learning is now the best default option
        )
        
        # Initialize system components
        self.detector = StereoMosquitoDetector(self.calibration, self.classifier)
        self.tracker = MosquitoTracker3D(max_tracks=10)
        self.servo_controller = StereoServoController(serial_port)
        
        # System state
        self.running = False
        self.targeting_enabled = False
        self.safety_mode = True  # Require manual confirmation to fire
        self.visualization_mode = 'stereo'  # 'stereo', '3d', or 'tracks'
        
        # Performance monitoring
        self.fps_counter = FPSCounter()
        self.detection_count = 0
        self.engagement_count = 0
        
        # Targeting parameters
        self.min_classification_score = 0.6  # Minimum score to engage
        self.min_track_length = 5  # Minimum detections before targeting
        self.engagement_cooldown = 0.5  # Seconds between shots
        self.last_engagement_time = 0
        
    def run(self):
        """
        Main processing loop
        
        Workflow:
        1. Capture synchronized stereo frames
        2. Detect and classify mosquitos in 3D
        3. Update multi-object tracking
        4. Select best target based on multiple criteria
        5. Engage target with predictive aiming
        6. Visualize results
        """
        self.running = True
        print("\nSystem active. Press 'q' to quit.")
        
        while self.running:
            loop_start_time = time.time()
            
            # Capture synchronized frames
            ret_left, frame_left = self.cap_left.read()
            ret_right, frame_right = self.cap_right.read()
            
            if not ret_left or not ret_right:
                print("Camera read failed!")
                continue
                
            # Main processing pipeline
            # 1. Detect mosquitos in 3D with classification
            detections_3d = self.detector.detect_3d(frame_left, frame_right)
            self.detection_count += len(detections_3d)
            
            # 2. Update tracking
            tracks = self.tracker.update(detections_3d)
            
            # 3. Visualization (before targeting for safety)
            vis_left = frame_left.copy()
            vis_right = frame_right.copy()
            self._draw_visualization(vis_left, vis_right, tracks, detections_3d)
            
            # 4. Target selection and engagement
            if self.targeting_enabled and tracks:
                target = self._select_best_target(tracks)
                if target:
                    engaged = self._engage_target(target)
                    if engaged:
                        self.engagement_count += 1
                        
            # 5. Display results
            self._display_output(vis_left, vis_right)
            
            # 6. Handle user input
            if not self._handle_controls():
                break
                
            # Update FPS counter
            self.fps_counter.update()
            
        self.cleanup()
        
    def _select_best_target(self, tracks: List[MosquitoTrack3D]) -> Optional[MosquitoTrack3D]:
        """
        Select optimal target based on multiple weighted criteria
        
        Scoring factors:
        - Classification confidence (is it really a mosquito?)
        - Distance (closer targets are easier)
        - Velocity (slower targets are easier)
        - Track stability (longer tracks are more reliable)
        - Time since last engagement (avoid double-targeting)
        """
        if not tracks:
            return None
            
        current_time = time.time()
        best_score = -1
        best_track = None
        
        for track in tracks:
            # Skip young tracks
            if len(track.positions_3d) < self.min_track_length:
                continue
                
            # Skip if classification confidence too low
            if track.classification_score < self.min_classification_score:
                continue
                
            # Current 3D position
            x, y, z = track.positions_3d[-1]
            distance = np.sqrt(x**2 + y**2 + z**2)
            
            # Skip if too far
            if distance > 2500:  # 2.5 meters
                continue
                
            # Calculate scoring components
            
            # 1. Distance score (inverse, normalized)
            distance_score = 1.0 / (1.0 + distance / 1000.0)
            
            # 2. Classification confidence
            classification_score = track.classification_score
            
            # 3. Velocity score (prefer slower targets)
            velocity = np.linalg.norm(track.velocity_3d)
            velocity_score = 1.0 / (1.0 + velocity / 100.0)
            
            # 4. Track stability (more observations = more stable)
            stability_score = min(len(track.positions_3d) / 30.0, 1.0)
            
            # 5. Centered score (prefer targets near center of view)
            centered_score = 1.0 - (abs(x) + abs(y)) / (distance + 1.0)
            
            # Combined weighted score
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
        
    def _engage_target(self, target: MosquitoTrack3D) -> bool:
        """
        Engage selected target with predictive aiming and safety checks
        
        Returns True if laser was fired
        """
        current_time = time.time()
        
        # Check engagement cooldown
        if current_time - self.last_engagement_time < self.engagement_cooldown:
            return False
            
        # Get current state from Kalman filter
        state = target.kalman.statePost
        x, y, z = state[0], state[1], state[2]
        velocity = state[3:6]
        
        # Command servos with velocity compensation
        pan, tilt, distance = self.servo_controller.target_3d_position(
            x, y, z, velocity
        )
        
        # Check if on target
        if self._is_on_target(target, pan, tilt, distance):
            if not self.safety_mode:  # Only fire if safety is off
                # Calculate appropriate laser power
                power = self.servo_controller.calculate_laser_power(
                    distance, 
                    target_size=50  # Approximate mosquito size
                )
                
                # Fire laser
                self.servo_controller.fire_laser(duration_ms=50, power=power)
                self.last_engagement_time = current_time
                
                print(f"Engaged target {target.id} at {distance:.0f}mm")
                return True
            else:
                print(f"Would engage target {target.id} (safety on)")
                
        return False
        
    def _is_on_target(self, target: MosquitoTrack3D, 
                     pan: float, tilt: float, distance: float) -> bool:
        """
        Verify servo alignment with target position
        Uses tighter tolerance for closer targets
        """
        # Get current position
        x, y, z = target.positions_3d[-1]
        
        # Calculate required angles
        required_pan = np.degrees(np.arctan2(x, z))
        required_tilt = np.degrees(np.arcsin(y / distance)) if distance > 0 else 0
        
        # Calculate errors
        pan_error = abs(required_pan - self.servo_controller.current_pan)
        tilt_error = abs(required_tilt - self.servo_controller.current_tilt)
        
        # Dynamic tolerance based on distance
        # Closer targets need more precision
        base_tolerance = 1.0  # degrees
        distance_factor = distance / 1000.0  # meters
        tolerance = base_tolerance * (1.0 + distance_factor * 0.5)
        
        return pan_error < tolerance and tilt_error < tolerance
        
    def _draw_visualization(self, left_frame: np.ndarray, right_frame: np.ndarray,
                          tracks: List[MosquitoTrack3D], 
                          detections: List[Dict]):
        """
        Draw comprehensive visualization including:
        - Tracked mosquitos with IDs and trails
        - 3D position information
        - Classification confidence
        - Current detections
        """
        # Draw tracks
        for track in tracks:
            if track.positions_2d_left and track.positions_2d_right:
                # Color based on classification confidence
                confidence = track.classification_score
                color = (
                    int(255 * (1 - confidence)),  # Red decreases with confidence
                    int(255 * confidence),         # Green increases with confidence
                    0
                )
                
                # Draw trail on left image
                if len(track.positions_2d_left) > 1:
                    points_left = np.array(list(track.positions_2d_left), dtype=np.int32)
                    cv2.polylines(left_frame, [points_left], False, color, 1)
                
                # Current position on left
                x_l, y_l = track.positions_2d_left[-1]
                cv2.circle(left_frame, (int(x_l), int(y_l)), 5, color, -1)
                
                # Draw on right image
                if len(track.positions_2d_right) > 1:
                    points_right = np.array(list(track.positions_2d_right), dtype=np.int32)
                    cv2.polylines(right_frame, [points_right], False, color, 1)
                
                x_r, y_r = track.positions_2d_right[-1]
                cv2.circle(right_frame, (int(x_r), int(y_r)), 5, color, -1)
                
                # Add information overlay
                if track.positions_3d:
                    x_3d, y_3d, z_3d = track.positions_3d[-1]
                    distance = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
                    velocity = np.linalg.norm(track.velocity_3d)
                    
                    # Create info text
                    info_lines = [
                        f"ID: {track.id}",
                        f"Dist: {distance:.0f}mm",
                        f"Vel: {velocity:.0f}mm/s",
                        f"Conf: {confidence:.2f}"
                    ]
                    
                    # Draw text background for readability
                    y_offset = int(y_l) - 60
                    for i, line in enumerate(info_lines):
                        y_pos = y_offset + i * 15
                        
                        # Get text size
                        (w, h), _ = cv2.getTextSize(
                            line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                        )
                        
                        # Draw background rectangle
                        cv2.rectangle(
                            left_frame,
                            (int(x_l) + 10, y_pos - h - 2),
                            (int(x_l) + 10 + w + 4, y_pos + 2),
                            (0, 0, 0),
                            -1
                        )
                        
                        # Draw text
                        cv2.putText(
                            left_frame, line,
                            (int(x_l) + 12, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1
                        )
                        
        # Draw current detections (before tracking)
        for det in detections:
            x_l, y_l = det['pos_2d_left']
            x_r, y_r = det['pos_2d_right']
            
            # Draw as squares to distinguish from tracked circles
            cv2.rectangle(left_frame, 
                         (int(x_l) - 3, int(y_l) - 3),
                         (int(x_l) + 3, int(y_l) + 3),
                         (255, 255, 0), 1)
            cv2.rectangle(right_frame,
                         (int(x_r) - 3, int(y_r) - 3),
                         (int(x_r) + 3, int(y_r) + 3),
                         (255, 255, 0), 1)
                         
    def _display_output(self, vis_left: np.ndarray, vis_right: np.ndarray):
        """
        Display visualization with system information overlay
        """
        # Add system status overlay
        status_lines = [
            f"FPS: {self.fps_counter.get_fps():.1f}",
            f"Targeting: {'ON' if self.targeting_enabled else 'OFF'}",
            f"Safety: {'ON' if self.safety_mode else 'OFF'}",
            f"Tracks: {len(self.tracker.tracks)}",
            f"Detections: {self.detection_count}",
            f"Engagements: {self.engagement_count}"
        ]
        
        # Draw status on left frame
        for i, line in enumerate(status_lines):
            cv2.putText(vis_left, line,
                       (10, 20 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)
                       
        # Combine stereo views
        if self.visualization_mode == 'stereo':
            combined = np.hstack([vis_left, vis_right])
            cv2.imshow('Mosquito Defense System', combined)
        else:
            cv2.imshow('Left View', vis_left)
            cv2.imshow('Right View', vis_right)
            
    def _handle_controls(self) -> bool:
        """
        Handle keyboard controls
        Returns False if should quit
        """
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('t'):
            self.targeting_enabled = not self.targeting_enabled
            print(f"Targeting: {'ON' if self.targeting_enabled else 'OFF'}")
        elif key == ord('s'):
            self.safety_mode = not self.safety_mode
            print(f"Safety mode: {'ON' if self.safety_mode else 'OFF'}")
        elif key == ord('v'):
            # Cycle visualization modes
            modes = ['stereo', 'separate']
            current_idx = modes.index(self.visualization_mode)
            self.visualization_mode = modes[(current_idx + 1) % len(modes)]
            print(f"Visualization: {self.visualization_mode}")
        elif key == ord('r'):
            # Reset statistics
            self.detection_count = 0
            self.engagement_count = 0
            print("Statistics reset")
            
        return True
        
    def cleanup(self):
        """Clean shutdown of all components"""
        print("\nShutting down...")
        self.running = False
        
        # Release cameras
        self.cap_left.release()
        self.cap_right.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Close serial connection
        self.servo_controller.serial.close()
        
        # Print session statistics
        print(f"\nSession Statistics:")
        print(f"Total detections: {self.detection_count}")
        print(f"Total engagements: {self.engagement_count}")
        if self.fps_counter.get_fps() > 0:
            runtime = len(self.fps_counter.timestamps) / self.fps_counter.get_fps()
            print(f"Runtime: {runtime:.1f} seconds")
            print(f"Average FPS: {self.fps_counter.get_fps():.1f}")

class FPSCounter:
    """Simple FPS counter using sliding window"""
    
    def __init__(self, window_size: int = 30):
        self.timestamps = deque(maxlen=window_size)
        
    def update(self):
        self.timestamps.append(time.time())
        
    def get_fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span > 0:
            return (len(self.timestamps) - 1) / time_span
        return 0.0

# Arduino code for the hardware interface
ARDUINO_CODE_STEREO = """
/*
 * Mosquito Defense System - Arduino Controller
 * Controls pan/tilt servos and laser module
 * 
 * Commands:
 * P<angle>,T<angle> - Set pan and tilt angles
 * F<position> - Set focus servo position
 * L<duration>,<power> - Fire laser with duration and power
 */

#include <Servo.h>

// Pin definitions
const int PAN_PIN = 9;
const int TILT_PIN = 10;
const int FOCUS_PIN = 11;
const int LASER_PIN = 13;
const int LASER_PWM_PIN = 6;

// Servo objects
Servo panServo;
Servo tiltServo;
Servo focusServo;

// State variables
int currentPan = 90;
int currentTilt = 90;
int currentFocus = 90;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.setTimeout(10);
  
  // Attach servos
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  focusServo.attach(FOCUS_PIN);
  
  // Initialize laser pins
  pinMode(LASER_PIN, OUTPUT);
  pinMode(LASER_PWM_PIN, OUTPUT);
  
  // Center all servos
  panServo.write(currentPan);
  tiltServo.write(currentTilt);
  focusServo.write(currentFocus);
  
  // Ensure laser is off
  digitalWrite(LASER_PIN, LOW);
  analogWrite(LASER_PWM_PIN, 0);
  
  // Ready signal
  Serial.println("READY");
}

void loop() {
  // Check for incoming commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\\n');
    processCommand(command);
  }
}

void processCommand(String cmd) {
  cmd.trim(); // Remove whitespace
  
  if (cmd.startsWith("P")) {
    // Pan/Tilt command: P<angle>,T<angle>
    int commaIndex = cmd.indexOf(',');
    if (commaIndex > 0) {
      // Extract pan angle
      int panAngle = cmd.substring(1, commaIndex).toInt();
      
      // Extract tilt angle
      int tiltStart = cmd.indexOf('T', commaIndex);
      if (tiltStart > 0) {
        int tiltAngle = cmd.substring(tiltStart + 1).toInt();
        
        // Apply servo center offset and limits
        currentPan = constrain(90 + panAngle, 0, 180);
        currentTilt = constrain(90 + tiltAngle, 0, 180);
        
        // Move servos
        panServo.write(currentPan);
        tiltServo.write(currentTilt);
        
        // Acknowledge
        Serial.print("OK P:");
        Serial.print(panAngle);
        Serial.print(",T:");
        Serial.println(tiltAngle);
      }
    }
    
  } else if (cmd.startsWith("F")) {
    // Focus command: F<position>
    int focusPos = cmd.substring(1).toInt();
    currentFocus = constrain(focusPos, 0, 180);
    focusServo.write(currentFocus);
    
    Serial.print("OK F:");
    Serial.println(currentFocus);
    
  } else if (cmd.startsWith("L")) {
    // Laser command: L<duration>,<power>
    int commaIndex = cmd.indexOf(',');
    if (commaIndex > 0) {
      int duration = cmd.substring(1, commaIndex).toInt();
      int power = cmd.substring(commaIndex + 1).toInt();
      
      // Safety limits
      duration = constrain(duration, 0, 500);
      power = constrain(power, 0, 100);
      
      // Fire laser
      analogWrite(LASER_PWM_PIN, power);
      digitalWrite(LASER_PIN, HIGH);
      delay(duration);
      digitalWrite(LASER_PIN, LOW);
      analogWrite(LASER_PWM_PIN, 0);
      
      Serial.print("OK L:");
      Serial.print(duration);
      Serial.print(",");
      Serial.println(power);
    }
  }
}
"""

# Utility script for dataset preparation and model training
def prepare_dataset_from_video(video_path: str, output_dir: str, sample_rate: int = 30):
    """
    Extract frames from video for manual labeling
    
    Args:
        video_path: Path to video file containing mosquitos
        output_dir: Directory to save extracted frames
        sample_rate: Extract one frame every N frames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_rate == 0:
            # Save frame
            filename = f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    print("Next steps:")
    print("1. Manually sort images into positive/ and negative/ folders")
    print("2. Crop images to show individual objects")
    print("3. Run train_classifier() to train the model")

def train_classifier(reference_dir: str = "mosquito_references/", 
                    method: str = "deep",
                    output_path: str = "mosquito_model.h5"):
    """
    Train mosquito classifier on prepared dataset
    
    Args:
        reference_dir: Directory with positive/ and negative/ subdirectories
        method: 'deep' (recommended), 'hog_svm', or 'template'
        output_path: Where to save trained model
    """
    print(f"Training {method} classifier...")
    
    # Initialize classifier
    classifier = MosquitoClassifier(
        reference_dir=reference_dir,
        method=method,
        model_path=output_path
    )
    
    # For MobileNet, run fine-tuning
    if method == "deep" and hasattr(classifier, 'model_type') and classifier.model_type == 'mobilenet':
        classifier.train_mobilenet(epochs=15)
        
    print("Training complete!")
    
    # Test the classifier
    test_classifier(classifier, reference_dir)

def test_classifier(classifier: MosquitoClassifier, test_dir: str):
    """Test classifier accuracy on a test set"""
    correct = 0
    total = 0
    
    # Test positive samples
    positive_dir = os.path.join(test_dir, "positive")
    if os.path.exists(positive_dir):
        for img_path in glob.glob(os.path.join(positive_dir, "*.jpg"))[:10]:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                score, method = classifier.classify_patch(img)
                if score > 0.5:
                    correct += 1
                total += 1
                
    # Test negative samples
    negative_dir = os.path.join(test_dir, "negative")
    if os.path.exists(negative_dir):
        for img_path in glob.glob(os.path.join(negative_dir, "*.jpg"))[:10]:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                score, method = classifier.classify_patch(img)
                if score < 0.5:
                    correct += 1
                total += 1
                
    if total > 0:
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.2%} ({correct}/{total})")

if __name__ == "__main__":
    print("=" * 60)
    print("Stereo Vision Mosquito Defense System v3.0")
    print("=" * 60)
    print("\nFeatures:")
    print("  - Stereo 3D tracking")
    print("  - Deep learning mosquito classification (MobileNet/CNN)")
    print("  - Predictive targeting with Kalman filtering")
    print("  - Adaptive laser power control")
    print("  - Multi-target tracking")
    print("\nControls:")
    print("  't' - Toggle targeting")
    print("  's' - Toggle safety mode")
    print("  'v' - Change visualization")
    print("  'r' - Reset statistics")
    print("  'q' - Quit")
    print("\n" + "!" * 60)
    print("WARNING: Laser safety protocols must be followed!")
    print("Never aim at eyes, people, or reflective surfaces!")
    print("!" * 60 + "\n")
    
    # Check for command line arguments
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "prepare":
            # Extract frames from video for dataset preparation
            if len(sys.argv) >= 3:
                prepare_dataset_from_video(sys.argv[2], "dataset_frames")
            else:
                print("Usage: python mosquito_defense.py prepare <video_file>")
        elif sys.argv[1] == "train":
            # Train classifier
            train_classifier()
        else:
            print("Unknown command. Use 'prepare' or 'train'")
        sys.exit(0)
    
    try:
        # Check for required files
        import os
        
        if not os.path.exists('stereo_calibration.pkl'):
            print("ERROR: No stereo calibration found!")
            print("Please run stereo calibration first.")
            print("See StereoCalibrator class for calibration process.")
            exit(1)
            
        if not os.path.exists('mosquito_references/'):
            print("WARNING: No mosquito reference images found!")
            print("Create mosquito_references/ directory with:")
            print("  - positive/ (mosquito images)")
            print("  - negative/ (non-mosquito images)")
            print("\nOr run: python mosquito_defense.py prepare <video_file>")
            print("to extract frames from a video for labeling.")
            print("\nSystem will use shape-based detection as fallback.\n")
            
        # Initialize and run system
        system = StereoMosquitoLaserSystem(
            left_camera_id=0,
            right_camera_id=1,
            calibration_file='stereo_calibration.pkl',
            serial_port='/dev/ttyUSB0',
            reference_dir='mosquito_references/'
        )
        
        system.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
