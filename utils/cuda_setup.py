"""
CUDA setup 
"""

import tensorflow as tf
import os

def setup_gpu(memory_growth=True):
    """
    Setup single GPU for TensorFlow
    
    Args:
        memory_growth: If True, allocate GPU memory as needed
    
    Returns:
        bool: True if GPU is available and configured
    """
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for the GPU
            tf.config.experimental.set_memory_growth(gpus[0], memory_growth)
            
            # Test GPU availability
            with tf.device('/GPU:0'):
                test = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                test = tf.matmul(test, test)
            
            print(f"GPU setup successful: {gpus[0].name}")
            return True
            
        except RuntimeError as e:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU")
            return False
    else:
        print("No GPU detected - using CPU")
        return False


def check_gpu_info():
    """
    Check GPU availability and info
    
    """
    
    gpus = tf.config.list_physical_devices('GPU')
    
    info = {
        'tensorflow_version': tf.__version__,
        'gpu_available': len(gpus) > 0,
        'gpu_name': gpus[0].name if gpus else 'None',
        'num_gpus': len(gpus)
    }
    
    return info


def print_device_info():
    """Print device information"""
    
    print("\n" + "="*50)
    print("Device Information")
    print("="*50)
    
    info = check_gpu_info()
    
    print(f"TensorFlow Version: {info['tensorflow_version']}")
    print(f"GPU Available: {info['gpu_available']}")
    
    if info['gpu_available']:
        print(f"GPU Device: {info['gpu_name']}")
    else:
        print("No GPU detected - will use CPU")
    
    print("="*50 + "\n")


def estimate_training_time(num_samples, batch_size, epochs, use_gpu=False):
    """
    Estimate training time based on device
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        epochs: Number of epochs
        use_gpu: Whether GPU is being used
    
    Returns:
        str: Estimated time
    """
    
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * epochs
    
    # Rough estimates
    if use_gpu:
        time_per_step = 0.1  
    else:
        time_per_step = 0.5  
    
    total_seconds = total_steps * time_per_step
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    if hours > 0:
        return f"~{hours}h {minutes}m"
    else:
        return f"~{minutes}m"


def auto_setup():
    """
    Automatic setup for training
    Simple single GPU setup
    
    Returns:
        bool: True if GPU is available
    """

    
    # Setup GPU
    gpu_available = setup_gpu()
    
    if not gpu_available:
        print("Tips for CPU training:")
        print("  - Use smaller batch size")
        print("  - Consider fewer epochs for testing")
    
    print("-" * 40 + "\n")
    
    return gpu_available


if __name__ == "__main__":
    # Test the setup
    print("Testing CUDA Setup...")
    
    # Print device info
    print_device_info()
    
    # Try auto setup
    gpu_available = auto_setup()
    
    # Test computation
    print("\nTesting computation...")
    
    device = '/GPU:0' if gpu_available else '/CPU:0'
    with tf.device(device):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"Test computation result:\n{c.numpy()}")
    
    print("\nSetup complete!")
    
