"""
Model retraining logic for waste classification.
Gets data from PostgreSQL database and retrains the model.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from database import (
    get_images_for_training,
    create_training_run, update_training_run
)
from preprocessing import (
    augment_image, VALID_CLASSES, DEFAULT_IMG_SIZE
)


# Configuration (can be overridden)
DEFAULT_CONFIG = {
    'img_size': DEFAULT_IMG_SIZE,
    'batch_size': 32,
    'val_split': 0.15,
    'test_split': 0.10,
    'epochs': 30,
    'learning_rate': 1e-3,
    'l2_weight': 1e-5,
    'dropout_rate': 0.2,
    'target_f1': 0.90,
}


class SklearnF1Callback(tf.keras.callbacks.Callback):
    """Callback to compute F1 score using sklearn during training."""
    
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true = []
        y_pred = []
        
        for batch_x, batch_y in self.validation_data:
            preds = self.model.predict(batch_x, verbose=0)
            y_true.append(batch_y.numpy())
            y_pred.append(np.argmax(preds, axis=1))
        
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        logs['val_macro_f1'] = macro_f1
        print(f" - val_macro_f1: {macro_f1:.4f}")


def load_existing_model(model_path: str = None) -> tf.keras.Model:
    """
    Load existing model from /models/waste_classifier_final.keras.
    
    Args:
        model_path: Path to model. If None, uses default path.
    
    Returns:
        Loaded Keras model
    """
    if model_path is None:
        possible_paths = [
            Path(__file__).parent.parent / 'models' / 'waste_classifier_final.keras',
            Path('models/waste_classifier_final.keras'),
            Path('/content/drive/MyDrive/waste-classification/models/waste_classifier_final.keras'),
        ]
        
        for path in possible_paths:
            if path.exists():
                model_path = str(path)
                break
        
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                "Could not find waste_classifier_final.keras model. "
                "Please train the model first."
            )
    
    print(f"Loading existing model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model


def load_model_config(config_path: str = None) -> Dict:
    """Load model configuration from JSON file."""
    if config_path is None:
        possible_paths = [
            Path(__file__).parent.parent / 'models' / 'model_config.json',
            Path('models/model_config.json'),
            Path('/content/drive/MyDrive/waste-classification/models/model_config.json'),
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
        
        if config_path is None:
            raise FileNotFoundError("Could not find model_config.json")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def get_data_from_database() -> List[Tuple[bytes, str]]:
    """
    Retrieve images from PostgreSQL database for training.
    
    Returns:
        List of (image_bytes, class_name) tuples
    """
    print("Fetching images from database...")
    images_data = get_images_for_training()
    
    if len(images_data) == 0:
        raise ValueError("No images found in database. Please upload images first.")
    
    # Extract image bytes and class names
    images = [(img_bytes, class_name) for _, img_bytes, class_name in images_data]
    
    # Print statistics
    counts = Counter([class_name for _, class_name in images])
    print(f"Total images: {len(images)}")
    print("Images per class:")
    for cls, count in counts.items():
        print(f"  {cls}: {count}")
    
    return images


def create_tf_dataset_from_bytes(images_data: List[Tuple[bytes, str]], 
                                 class_names: List[str],
                                 img_size: Tuple[int, int],
                                 batch_size: int,
                                 training: bool = False):
    """
    Create TensorFlow dataset from image bytes and labels.
    
    Args:
        images_data: List of (image_bytes, class_name) tuples
        class_names: List of all class names for label encoding
        img_size: Target image size
        batch_size: Batch size
        training: Whether this is for training (applies augmentation)
    
    Returns:
        TensorFlow dataset
    """
    # Save images to temporary files (TensorFlow needs file paths)
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    labels = []
    
    for idx, (img_bytes, class_name) in enumerate(images_data):
        label_idx = class_names.index(class_name)
        temp_file = Path(temp_dir) / f"img_{idx}.jpg"
        temp_file.write_bytes(img_bytes)
        temp_files.append(str(temp_file))
        labels.append(label_idx)
    
    # Create dataset
    def decode_file(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32)
        return image, label
    
    ds = tf.data.Dataset.from_tensor_slices((temp_files, labels))
    ds = ds.map(decode_file, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(2048)
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, temp_dir


def build_model(num_classes: int, img_size: Tuple[int, int], learning_rate: float,
                l2_weight: float, dropout_rate: float):
    """
    Build EfficientNetB0-based waste classification model.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size (height, width)
        learning_rate: Learning rate for optimizer
        l2_weight: L2 regularization weight
        dropout_rate: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(*img_size, 3),
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = tf.keras.layers.Input(shape=(*img_size, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(512, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2_weight))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ],
    )
    
    return model


def retrain_model(model_dir: Path, config: Dict = None) -> tf.keras.Model:
    """
    Retrain the waste classification model using data from database.
    Loads previous model from /models/waste_classifier_final.keras.
    
    Args:
        model_dir: Directory to save retrained model
        config: Training configuration (uses defaults if None)
    
    Returns:
        Retrained model
    """
    # Create training run
    run_id = create_training_run()
    print(f"Created training run ID: {run_id}")
    
    try:
        # Load configuration
        if config is None:
            config = DEFAULT_CONFIG.copy()
            try:
                existing_config = load_model_config()
                class_names = existing_config.get('class_names', VALID_CLASSES)
                config['img_size'] = tuple(existing_config.get('img_size', DEFAULT_IMG_SIZE))
            except Exception:
                class_names = VALID_CLASSES
        else:
            class_names = config.get('class_names', VALID_CLASSES)
        
        # Get data from database
        images_data = get_data_from_database()
        
        # Split dataset
        images, labels = zip(*images_data)
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            list(images), list(labels),
            test_size=(config['val_split'] + config['test_split']),
            stratify=list(labels),
            random_state=42
        )
        
        relative_test_split = config['test_split'] / (config['val_split'] + config['test_split'])
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels,
            test_size=relative_test_split,
            stratify=temp_labels,
            random_state=42
        )
        
        print(f"Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")
        
        # Create datasets
        train_data = list(zip(train_images, train_labels))
        val_data = list(zip(val_images, val_labels))
        test_data = list(zip(test_images, test_labels))
        
        train_ds, temp_dir_train = create_tf_dataset_from_bytes(
            train_data, class_names, config['img_size'], config['batch_size'], training=True
        )
        val_ds, temp_dir_val = create_tf_dataset_from_bytes(
            val_data, class_names, config['img_size'], config['batch_size'], training=False
        )
        test_ds, temp_dir_test = create_tf_dataset_from_bytes(
            test_data, class_names, config['img_size'], config['batch_size'], training=False
        )
        
        # Load existing model or build new one
        try:
            model = load_existing_model('models/efficientnetb0_waste_classifier.h5')
            print("Using existing model as base for retraining")
        except FileNotFoundError:
            print("No existing model found. Building new model...")
            model = build_model(
                num_classes=len(class_names),
                img_size=config['img_size'],
                learning_rate=config['learning_rate'],
                l2_weight=config['l2_weight'],
                dropout_rate=config['dropout_rate']
            )
        
        # Create callbacks
        checkpoint_path = model_dir / 'efficientnetb0_waste_classifier_retrained.h5'
        callbacks = [
            SklearnF1Callback(val_ds),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_macro_f1',
                mode='max',
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_macro_f1',
                mode='max',
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        
        # Train model
        print("Training model...")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config['epochs'],
            callbacks=callbacks,
        )
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = model.evaluate(test_ds, return_dict=True)
        
        probs = model.predict(test_ds)
        y_pred = np.argmax(probs, axis=1)
        y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        test_metrics['macro_f1'] = macro_f1
        
        print('\nTest metrics:')
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print('\nClassification report:')
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Save final model
        final_model_path = model_dir / 'efficientnetb0_waste_classifier_final.h5'
        model.save(str(final_model_path))
        print(f'\nFinal model saved to {final_model_path}')
        
        # Update config
        config_path = model_dir / 'model_config.json'
        config_data = {
            'class_names': class_names,
            'img_size': config['img_size'],
            'model_path': str(final_model_path)
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f'Model configuration saved to {config_path}')
        
        # Update training run
        update_training_run(
            run_id, 'completed', test_metrics, str(final_model_path)
        )
        
        # Clean up temp directories
        import shutil
        for temp_dir in [temp_dir_train, temp_dir_val, temp_dir_test]:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        
        return model
    
    except Exception as e:
        update_training_run(run_id, 'failed')
        raise e
