"""
Ultra-Light Single-Scale SACNN 

"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, regularizers
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.single_scale_config import CONFIG

class SingleScaleSACNN(Model):
    """
    Ultra-light SACNN for edge deployment
    - 0.125x width multiplier (8-16-32-64 channels)
    - Strong regularization to prevent overfitting
    - Processes full 256x256 images (no tiling needed)
    """
    
    def __init__(self, 
                 width_multiplier=0.25,  
                 use_depthwise=True,      
                 dropout_rate=0.15,        # High dropout to prevent overfitting
                 l2_weight=0.001,          # L2 regularization
                 name="SingleScaleSACNN"):
        super(SingleScaleSACNN, self).__init__(name=name)
        
        self.use_depthwise = use_depthwise
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        c1 = 32  
        c2 = 64  
        c3 = 128 
        c4 = 256 
                
        # Adaptation layers 
        p1 = max(32, int(256 * width_multiplier))  # 32
        p2 = max(16, int(128 * width_multiplier))  # 16
        
        # L2 regularizer
        l2_reg = regularizers.l2(l2_weight) if l2_weight > 0 else None
        
        # ========== Conv Block 1: 256→128 ==========
        if use_depthwise:
            self.conv1_1 = self._depthwise_separable_conv(c1, 'conv1_1', l2_reg)
            self.conv1_2 = self._depthwise_separable_conv(c1, 'conv1_2', l2_reg)
        else:
            self.conv1_1 = layers.Conv2D(c1, 3, padding='same', activation='relu', 
                                        kernel_regularizer=l2_reg, name='conv1_1')
            self.conv1_2 = layers.Conv2D(c1, 3, padding='same', activation='relu',
                                        kernel_regularizer=l2_reg, name='conv1_2')
        
        self.pool1 = layers.MaxPooling2D(2, 2, name='pool1')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        
        # ========== Conv Block 2: 128→64 ==========
        if use_depthwise:
            self.conv2_1 = self._depthwise_separable_conv(c2, 'conv2_1', l2_reg)
            self.conv2_2 = self._depthwise_separable_conv(c2, 'conv2_2', l2_reg)
        else:
            self.conv2_1 = layers.Conv2D(c2, 3, padding='same', activation='relu',
                                        kernel_regularizer=l2_reg, name='conv2_1')
            self.conv2_2 = layers.Conv2D(c2, 3, padding='same', activation='relu',
                                        kernel_regularizer=l2_reg, name='conv2_2')
        
        self.pool2 = layers.MaxPooling2D(2, 2, name='pool2')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # ========== Conv Block 3: 64→32 ==========
        if use_depthwise:
            self.conv3_1 = self._depthwise_separable_conv(c3, 'conv3_1', l2_reg)
            self.conv3_2 = self._depthwise_separable_conv(c3, 'conv3_2', l2_reg)
            # Skip conv3_3 
            self.conv3_3 = layers.Lambda(lambda x: x, name='conv3_3')
        else:
            self.conv3_1 = layers.Conv2D(c3, 3, padding='same', activation='relu',
                                        kernel_regularizer=l2_reg, name='conv3_1')
            self.conv3_2 = layers.Conv2D(c3, 3, padding='same', activation='relu',
                                        kernel_regularizer=l2_reg, name='conv3_2')
            self.conv3_3 = layers.Conv2D(c3, 3, padding='same', activation='relu',
                                        kernel_regularizer=l2_reg, name='conv3_3')
        
        self.pool3 = layers.MaxPooling2D(2, 2, name='pool3')
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(dropout_rate * 1.5)  # Higher dropout
        
        # ========== Conv Block 4: Stay at 32×32 ==========
        if use_depthwise:
            self.conv4_1 = self._depthwise_separable_conv(c4, 'conv4_1', l2_reg)
            self.conv4_2 = self._depthwise_separable_conv(c4, 'conv4_2', l2_reg)
            # Skip conv4_3 
            self.conv4_3 = layers.Lambda(lambda x: x, name='conv4_3')
        else:
            self.conv4_1 = layers.Conv2D(c4, 3, padding='same', activation='relu',
                                        kernel_regularizer=l2_reg, name='conv4_1')
            self.conv4_2 = layers.Conv2D(c4, 3, padding='same', activation='relu',
                                        kernel_regularizer=l2_reg, name='conv4_2')
            self.conv4_3 = layers.Conv2D(c4, 3, padding='same', activation='relu',
                                        kernel_regularizer=l2_reg, name='conv4_3')
        
        self.bn4 = layers.BatchNormalization()
        self.dropout4 = layers.Dropout(dropout_rate * 1.5)
        
        # ========== Adaptation Layers ==========
        # Use 1×1 convolutions 
        self.p_conv1 = layers.Conv2D(p1, 1, padding='same', activation='relu',
                                    kernel_regularizer=l2_reg, name='p_conv1')
        self.p_conv2 = layers.Conv2D(p2, 1, padding='same', activation='relu',
                                    kernel_regularizer=l2_reg, name='p_conv2')
        self.p_conv3 = layers.Conv2D(1, 1, padding='same', activation=None, name='density_map')
        
    def _depthwise_separable_conv(self, filters, name, l2_reg=None):
        """Depthwise separable convolution with regularization"""
        return keras.Sequential([
            layers.DepthwiseConv2D(3, padding='same', use_bias=False,
                                  depthwise_regularizer=l2_reg, name=f'{name}_dw'),
            layers.BatchNormalization(),
            layers.ReLU(max_value=6),  
            layers.Conv2D(filters, 1, padding='same', use_bias=False,
                        kernel_regularizer=l2_reg, name=f'{name}_pw'),
            layers.BatchNormalization(),
            layers.ReLU(max_value=6)
        ], name=name)
    
    def call(self, inputs, training=False):
        """Forward pass"""
        
        # Conv Block 1
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        # Conv Block 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        # Conv Block 3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)
        
        # Conv Block 4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.bn4(x, training=training)
        x = self.dropout4(x, training=training)
        
        # Adaptation layers
        x = self.p_conv1(x)
        x = self.p_conv2(x)
        density_map = self.p_conv3(x)
        
        # Calculate count
        count = tf.reduce_sum(density_map, axis=[1, 2, 3], keepdims=True)
        
        return density_map, count
    
    def get_config(self):
        return {
            "name": self.name,
            "width_multiplier": self.width_multiplier,
            "use_depthwise": self.use_depthwise,
            "dropout_rate": self.dropout_rate
        }


def create_single_scale_model(input_shape=(128, 128, 3),
                             width_multiplier=0.75,
                             use_depthwise=False,
                             dropout_rate=0.1,
                             l2_weight=0.0):
    """
    Create light SACNN model 
    
    Args:
        input_shape: Input shape (256, 256, 3)
        width_multiplier: 0.125 
        use_depthwise: True 
        dropout_rate: To prevent overfitting
        l2_weight: L2 regularization weight
    
    Returns:
        Keras model ready for training
    """
    
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    sacnn = SingleScaleSACNN(
        width_multiplier=width_multiplier,
        use_depthwise=use_depthwise,
        dropout_rate=dropout_rate,
        l2_weight=l2_weight
    )
    
    density_map, count = sacnn(inputs)
    
    model = Model(
        inputs=inputs,
        outputs={
            'density_map': density_map,
            'count': count
        },
        name='SingleScaleSACNN'
    )
    
    return model


def count_parameters(model):
    """Count model parameters and estimate size"""
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    float32_size_mb = (total_params * 4) / (1024 * 1024)
    int8_size_kb = (total_params * 1) / 1024
    
    return {
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': total_params,
        'float32_size_mb': float32_size_mb,
        'int8_size_kb': int8_size_kb,
    }


def estimate_memory_usage(input_shape=(256, 256, 3), width_multiplier=0.125):
    """
    Estimate runtime memory for full 256x256 inference
    """
    h, w, _ = input_shape
    
    # Channels with 0.125x multiplier
    c1, c2, c3, c4 = 8, 16, 32, 64
    p1, p2 = 32, 16
    
    print("\nMemory Usage Analysis (INT8):")
    print("="*50)
    
    stages = [
        ("Input", h, w, 3),
        ("After conv1+pool1", h//2, w//2, c1),
        ("After conv2+pool2", h//4, w//4, c2),
        ("After conv3+pool3", h//8, w//8, c3),
        ("After conv4", h//8, w//8, c4),
        ("After p_conv1", h//8, w//8, p1),
        ("After p_conv2", h//8, w//8, p2),
        ("Output", h//8, w//8, 1),
    ]
    
    peak_memory = 0
    for name, sh, sw, sc in stages:
        memory_bytes = sh * sw * sc  # INT8 = 1 byte per value
        memory_kb = memory_bytes / 1024
        print(f"{name:20s}: {sh:3d}×{sw:3d}×{sc:2d} = {memory_kb:6.1f} KB")
        peak_memory = max(peak_memory, memory_kb)
    
    # Need to keep input + first conv output in memory
    total_peak = 192 + 128  # Input (192KB) + conv1 output (128KB)
    print(f"\nPeak memory usage: ~{total_peak} KB")

    
    return total_peak


def get_training_config():
    """
    Training configuration to prevent overfitting
    """
    return {
        'optimizer': tf.keras.optimizers.Adam(learning_rate=5e-5),  # Low LR
        'batch_size': 4,  # Larger batch for stability
        'epochs': 150,  
        'callbacks': [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            ),
        ],
        'data_augmentation': tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomBrightness(0.3),
            layers.RandomContrast(0.3),
        ])
    }


if __name__ == "__main__":

    # Create model
    model = create_single_scale_model(
        input_shape=(256, 256, 3),
        width_multiplier=0.125,
        use_depthwise=True,
        dropout_rate=0.3,
        l2_weight=0.01
    )
    
    # Analyze model
    param_info = count_parameters(model)
    print(f"\nModel Statistics:")
    print(f"Total parameters: {param_info['total_params']:,}")
    print(f"Float32 size: {param_info['float32_size_mb']:.2f} MB")
    print(f"INT8 size (estimated): {param_info['int8_size_kb']:.1f} KB")
    
    # Memory analysis
    memory_kb = estimate_memory_usage()
    
    # Test forward pass
    dummy_input = tf.random.normal((1, 256, 256, 3))
    outputs = model(dummy_input)
    print(f"\nOutput shapes:")
    print(f"Density map: {outputs['density_map'].shape}")
    print(f"Count: {outputs['count'].shape}")
    