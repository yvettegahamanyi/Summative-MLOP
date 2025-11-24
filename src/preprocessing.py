"""
Image preprocessing utilities for waste classification.
Handles image transformation for prediction and database uploads for retraining.
"""

import io
import os
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
import tensorflow as tf

from database import save_image_to_db, save_images_batch


# Default image size (should match training configuration)
DEFAULT_IMG_SIZE = (224, 224)

# Valid class names
VALID_CLASSES = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash',
    'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
]


def preprocess_image(image_bytes: bytes, img_size: Tuple[int, int] = DEFAULT_IMG_SIZE) -> np.ndarray:
    """
    Preprocess an image from bytes for EfficientNet model prediction.
    Keeps values in [0, 255] range as the model has EfficientNet preprocessing built-in.
    
    Args:
        image_bytes: Raw image bytes from uploaded file
        img_size: Target image size (height, width)
    
    Returns:
        Preprocessed image array in shape (1, height, width, 3) with values in [0, 255]
    """
    try:
        # Open and convert to RGB (ensures 3 channels)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize to target size
        img = img.resize(img_size)
        
        # Convert to numpy array, keep in [0, 255] range
        # The model has EfficientNet preprocessing built-in
        arr = np.array(img, dtype=np.float32)
        
        # Ensure we have 3 channels (RGB)
        if len(arr.shape) == 2:
            # Grayscale image - convert to RGB by repeating channels
            arr = np.stack([arr, arr, arr], axis=-1)
        elif len(arr.shape) == 3 and arr.shape[2] == 1:
            # Single channel - convert to RGB
            arr = np.repeat(arr, 3, axis=2)
        elif len(arr.shape) == 3 and arr.shape[2] != 3:
            raise ValueError(f"Unexpected number of channels: {arr.shape[2]}. Expected 3 (RGB).")
        
        # Verify shape is (height, width, 3)
        if len(arr.shape) != 3 or arr.shape[2] != 3:
            raise ValueError(f"Invalid image shape after preprocessing: {arr.shape}. Expected (H, W, 3).")
        
        # Add batch dimension: (1, height, width, 3)
        arr = np.expand_dims(arr, axis=0)
        
        return arr
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def preprocess_image_from_path(image_path: str, img_size: Tuple[int, int] = DEFAULT_IMG_SIZE) -> np.ndarray:
    """
    Preprocess an image from file path for prediction.
    
    Args:
        image_path: Path to image file
        img_size: Target image size (height, width)
    
    Returns:
        Preprocessed image array in shape (1, height, width, 3)
    """
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return preprocess_image(image_bytes, img_size)


def validate_class_name(class_name: str) -> bool:
    """
    Validate if class name is in the list of valid classes.
    
    Args:
        class_name: Class name to validate
    
    Returns:
        True if valid, False otherwise
    """
    return class_name in VALID_CLASSES


def upload_image_to_db(image_bytes: bytes, class_name: str, filename: str) -> int:
    """
    Upload a single image to the database for retraining.
    
    Args:
        image_bytes: Image file bytes
        class_name: Waste category class name (must be valid)
        filename: Original filename
    
    Returns:
        ID of the inserted record
    
    Raises:
        ValueError: If class name is invalid
    """
    if not validate_class_name(class_name):
        raise ValueError(
            f"Invalid class name: {class_name}. "
            f"Valid classes are: {', '.join(VALID_CLASSES)}"
        )
    
    # Validate image can be opened
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")
    
    return save_image_to_db(image_bytes, class_name, filename)


def upload_images_batch(images_data: List[Tuple[bytes, str, str]]) -> List[int]:
    """
    Upload multiple images to the database in a batch for retraining.
    
    Args:
        images_data: List of (image_bytes, class_name, filename) tuples
    
    Returns:
        List of inserted record IDs
    
    Raises:
        ValueError: If any class name is invalid or image is invalid
    """
    # Validate all class names
    for _, class_name, filename in images_data:
        if not validate_class_name(class_name):
            raise ValueError(
                f"Invalid class name '{class_name}' for file '{filename}'. "
                f"Valid classes are: {', '.join(VALID_CLASSES)}"
            )
    
    # Validate all images
    for img_bytes, class_name, filename in images_data:
        try:
            Image.open(io.BytesIO(img_bytes)).verify()
        except Exception as e:
            raise ValueError(f"Invalid image file '{filename}': {str(e)}")
    
    return save_images_batch(images_data)


def decode_image_for_training(path: str, label: int, img_size: Tuple[int, int] = DEFAULT_IMG_SIZE):
    """
    Decode and preprocess image for training pipeline.
    Used in TensorFlow data pipeline.
    Keeps in [0, 255] range for EfficientNet preprocessing in model.
    
    Args:
        path: Path to image file
        label: Integer label
        img_size: Target image size (height, width)
    
    Returns:
        Tuple of (image tensor, label)
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # Ensure RGB
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32)  # Keep in [0, 255] range
    return image, label


def augment_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply data augmentation to training images.
    
    Args:
        image: Image tensor in [0, 255] range
        label: Label tensor
    
    Returns:
        Tuple of (augmented image, label)
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Brightness and contrast adjustments work on [0, 255] range
    image = tf.image.random_brightness(image, max_delta=25.5)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    # Clip to ensure values stay in valid range
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label