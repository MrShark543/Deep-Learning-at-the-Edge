"""
Single-Scale VGG-based architecture for SACNN 

"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import sys
from pathlib import Path
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, UpSampling2D
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.single_scale_config import CONFIG

class SingleScaleSACNN(Model):
    """
    Single-scale SACNN with VGG-like backbone 
    """
    
    def __init__(self, name="SingleScaleSACNN"):
        super(SingleScaleSACNN, self).__init__(name=name)
        
        # ========== Conv Block 1: 64 filters, /2 downsampling ==========
        self.conv1_1 = layers.Conv2D(
            CONFIG.CONV1_FILTERS, 3, padding='same', 
            activation='relu', name='conv1_1'
        )
        self.conv1_2 = layers.Conv2D(
            CONFIG.CONV1_FILTERS, 3, padding='same', 
            activation='relu', name='conv1_2'
        )
        self.pool1 = layers.MaxPooling2D(2, 2, name='pool1') 
        
        # ========== Conv Block 2: 128 filters, /2 downsampling ==========
        self.conv2_1 = layers.Conv2D(
            CONFIG.CONV2_FILTERS, 3, padding='same', 
            activation='relu', name='conv2_1'
        )
        self.conv2_2 = layers.Conv2D(
            CONFIG.CONV2_FILTERS, 3, padding='same', 
            activation='relu', name='conv2_2'
        )
        self.pool2 = layers.MaxPooling2D(2, 2, name='pool2') 
        
        # ========== Conv Block 3: 256 filters, /2 downsampling ==========
        self.conv3_1 = layers.Conv2D(
            CONFIG.CONV3_FILTERS, 3, padding='same', 
            activation='relu', name='conv3_1'
        )
        self.conv3_2 = layers.Conv2D(
            CONFIG.CONV3_FILTERS, 3, padding='same', 
            activation='relu', name='conv3_2'
        )
        self.conv3_3 = layers.Conv2D(
            CONFIG.CONV3_FILTERS, 3, padding='same', 
            activation='relu', name='conv3_3'
        )
        self.pool3 = layers.MaxPooling2D(2, 2, name='pool3') 
        
        # ========== Conv Block 4: 512 filters, NO pooling ==========

        self.conv4_1 = layers.Conv2D(
            CONFIG.CONV4_FILTERS, 3, padding='same', 
            activation='relu', name='conv4_1'
        )
        self.conv4_2 = layers.Conv2D(
            CONFIG.CONV4_FILTERS, 3, padding='same', 
            activation='relu', name='conv4_2'
        )
        self.conv4_3 = layers.Conv2D(
            CONFIG.CONV4_FILTERS, 3, padding='same', 
            activation='relu', name='conv4_3'
        )
        
        # ========== Adaptation Layers (p_conv) ==========
        # These adapt VGG features for density estimation
        self.p_conv1 = layers.Conv2D(
            CONFIG.P_CONV1_FILTERS, 3, padding='same', 
            activation='relu', name='p_conv1'
        )
        self.p_conv2 = layers.Conv2D(
            CONFIG.P_CONV2_FILTERS, 3, padding='same', 
            activation='relu', name='p_conv2'
        )
        self.p_conv3 = layers.Conv2D(
            CONFIG.P_CONV3_FILTERS, 1, padding='same', 
            activation=None, name='density_map'
        )
        
    def call(self, inputs, training=False):
        """Forward pass """
        
        # Conv Block 1
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)  
        
        # Conv Block 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)  
        
        # Conv Block 3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x) 
        x = self.pool3(x)  
        
        # Conv Block 4
  
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)  
       
        
        # Adaptation layers
        x = self.p_conv1(x)  
        x = self.p_conv2(x)  
        density_map = self.p_conv3(x)  
        
        # Calculate count 
        count = tf.reduce_sum(density_map, axis=[1, 2, 3], keepdims=True)
        
        return density_map, count
    
    def get_config(self):
        """Get model configuration"""
        return {"name": self.name}


def create_single_scale_model(input_shape=(256, 256, 3)):
    """
    Create and return the single scale SACNN model
    
    Args:
        input_shape: Input tensor shape (H, W, C)
    
    Returns:
        Compiled Keras model
    """
    
    # Create input layer
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Create model instance
    sacnn = SingleScaleSACNN()
    
    # Get outputs
    density_map, count = sacnn(inputs)
    
    # Create Keras Model
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
    """
    Count model parameters and estimate size
    
    Args:
        model: Keras model
    
    Returns:
        dict: Parameter counts and size estimates
    """
    
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    # Estimate model sizes
    float32_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    
    info = {
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': total_params,
        'float32_size_mb': float32_size_mb,
    }
    
    return info


if __name__ == "__main__":
    # Test model creation
    print("Creating Single Scale SACNN Model")
    print("-" * 50)
    
    # Create model with 256x256 input
    model = create_single_scale_model(input_shape=(256, 256, 3))  
    
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = tf.random.normal((1, 256, 256, 3)) 
    outputs = model(dummy_input)
    
    # Verify output dimensions
    expected_density_shape = (1, 16, 16, 1) 
    expected_count_shape = (1, 1, 1, 1)
    
    assert outputs['density_map'].shape == expected_density_shape, \
        f"Density map shape incorrect! Expected {expected_density_shape}, got {outputs['density_map'].shape}"
    
    assert outputs['count'].shape == expected_count_shape, \
        f"Count shape incorrect! Expected {expected_count_shape}, got {outputs['count'].shape}"
    
