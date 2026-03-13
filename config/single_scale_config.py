"""
Configuration for Single-Scale SACNN 

"""

import os
from pathlib import Path

class SingleScaleConfig:
    """Configuration for single-scale SACNN model"""
    
    # ============= Model Architecture =============
    # Input configuration
    INPUT_HEIGHT = 256
    INPUT_WIDTH = 256
    INPUT_CHANNELS = 3  # RGB
    
    # Output configuration 
    OUTPUT_HEIGHT = 32  # CHANGED from 16 
    OUTPUT_WIDTH = 32   # CHANGED from 16 
    OUTPUT_CHANNELS = 1  # Density map
    
    # Network architecture 
    CONV1_FILTERS = 64   # CHANGED from 32
    CONV2_FILTERS = 128  # CHANGED from 64
    CONV3_FILTERS = 256  # CHANGED from 128
    CONV4_FILTERS = 512  # CHANGED from 256
    
    # Adaptation layers
    P_CONV1_FILTERS = 256  # INCREASED from 128
    P_CONV2_FILTERS = 128  # INCREASED from 64
    P_CONV3_FILTERS = 1    # Final density map
    
    # ============= Training Configuration =============
    # Batch size 
    BATCH_SIZE = 1

    # Learning rate schedule 
    INITIAL_LR = 1e-4     # CHANGED from 1e-4 
    FINAL_LR = 1e-6       # CHANGED from 1e-6
    LR_DECAY_EPOCHS = [150, 200]  # CHANGED from [150, 200]
    LR_DECAY_FACTOR = 0.5
    
    # Optimizer settings 
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    # Training epochs - INCREASED
    EPOCHS = 300  # CHANGED from 250
    QUICK_TEST_EPOCHS = 10 
    
    # Loss weights 
    DENSITY_LOSS_WEIGHT = 1.0
    COUNT_LOSS_WEIGHT = 0.1

    # When to add count loss - LATER
    ADD_COUNT_LOSS_EPOCH = 200  # CHANGED from 50


    # ============= Edge Deployment Configuration =============
    EDGE_DEPLOYMENT = False  # Set to True for Teensy deployment to use single_scale_edge file
    WIDTH_MULTIPLIER = 0.25  # Channel reduction for edge
    USE_DEPTHWISE = True     # Use depthwise separable convs
    DROPOUT_RATE = 0.1       # Dropout to prevent overfitting

    # Tiled inference 
    TILE_SIZE = 128
    TILE_OVERLAP = 16
    
    # ============= Data Configuration =============
    # Dataset paths
    # DATA_ROOT = Path("./datasets/shanghaitech_256x256_rgb")
    DATA_ROOT = Path("/kaggle/input/datasets/shashanksathyan/modified-shagitech-part-b-dataset/shanghaitech_256x256_rgb")
    PART_A_PATH = DATA_ROOT / "part_A"
    PART_B_PATH = DATA_ROOT / "part_B"
    PART_MIXED_PATH = DATA_ROOT / "part_mixed"
    
    # Data splits
    TRAIN_SPLIT = "train"
    TEST_SPLIT = "test"
    
    # Data augmentation
    USE_AUGMENTATION = True 
    AUGMENTATION_FACTOR = 3  
    
    # ============= Paths and Directories =============
    # Model checkpoints
    CHECKPOINT_DIR = Path("./checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # Logs
    LOG_DIR = Path("./logs")
    LOG_DIR.mkdir(exist_ok=True)
    
    # Results
    RESULTS_DIR = Path("./results")
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Saved models
    SAVED_MODELS_DIR = Path("./saved_models")
    SAVED_MODELS_DIR.mkdir(exist_ok=True)
    
    # ============= Quantization =============
    MAX_MODEL_SIZE_MB = 1.0  # Target 
    QUANTIZATION_SAMPLES = 100  # Samples for quantization calibration
    
    # ============= Evaluation Metrics =============
    # Metrics to track
    TRACK_MAE = True
    TRACK_MSE = True
    TRACK_RMSE = True
    
    # Paper target performance
    PAPER_TARGETS = {
        'A': {'MAE': 90.7, 'MSE': 152.6},
        'B': {'MAE': 33.2, 'MSE': 53.2},
        'mixed': {'MAE': 62.0, 'MSE': 102.9} #random
    }
    
    # ============= Visualization =============
    SAVE_PREDICTIONS = True
    PREDICTION_SAMPLES = 10
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as dictionary"""
        config = {}
        for key in dir(cls):
            if not key.startswith('_') and not callable(getattr(cls, key)):
                config[key] = getattr(cls, key)
        return config
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("\n" + "="*50)
        print("Single-Scale SACNN Configuration")
        print("="*50)
        config = cls.get_config_dict()
        for key, value in config.items():
            print(f"{key}: {value}")
        print("="*50 + "\n")

# Global config instance
CONFIG = SingleScaleConfig()