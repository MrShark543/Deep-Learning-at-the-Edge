# """
# Data loader for preprocessed dataset
# """

# import tensorflow as tf
# import numpy as np
# from pathlib import Path
# import cv2
# import sys

# # Add project root to path
# sys.path.append(str(Path(__file__).parent.parent))
# from config.single_scale_config import CONFIG

# class SimpleDataLoader:
#     """
#     Simple loader for preprocessed ShanghaiTech dataset
#     """
    
#     def __init__(self, data_root=None):
#         self.data_root = Path(data_root) if data_root else CONFIG.DATA_ROOT
        
#     def get_file_pairs(self, part='A', split='train'):
#         """
#         Get pairs of (image_path, density_map_path)
        
#         Args:
#             part: 'A' or 'B'
#             split: 'train' or 'test'
        
#         Returns:
#             List of (image_path, density_map_path) tuples
#         """
        
#         # Construct paths
#         if part.lower() == 'mixed':
#             base_path = self.data_root / "part_mixed"
#         elif part == 'A':
#             base_path = self.data_root / "part_A"
#         else:  # part == 'B'
#             base_path = self.data_root / "part_B"
        
#         img_dir = base_path / split / "images"
#         density_dir = base_path / split / "density_maps"
        
#         # Check if directories exist
#         if not img_dir.exists():
#             raise ValueError(f"Image directory not found: {img_dir}")
#         if not density_dir.exists():
#             raise ValueError(f"Density directory not found: {density_dir}")
        
#         # Get all image files
#         img_files = sorted(list(img_dir.glob("*.png")))
#         if len(img_files) == 0:
#             img_files = sorted(list(img_dir.glob("*.jpg")))
        
#         # Match with density maps
#         file_pairs = []
#         for img_path in img_files:
#             density_path = density_dir / f"{img_path.stem}.npy"
            
#             if density_path.exists():
#                 file_pairs.append((str(img_path), str(density_path)))
#             else:
#                 print(f"Warning: No density map for {img_path.name}")
        
#         print(f"Found {len(file_pairs)} image-density pairs for Part {part} {split}")
#         return file_pairs
    
   

#     def load_and_preprocess(self, img_path, density_path, augment=False):
#         """
#         Load and preprocess a single image-density pair with optional augmentation
        
#         Args:
#             img_path: Path to image file
#             density_path: Path to density map file
#             augment: Whether to apply data augmentation
        
#         Returns:
#             Tuple of (image, density_map, count)
#         """
        
#         # Load image 
#         image = cv2.imread(img_path)
#         if image is None:
#             raise ValueError(f"Could not load image: {img_path}")
        
#         # Convert BGR to RGB
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # Resize if needed
#         if image.shape[0] != CONFIG.INPUT_HEIGHT or image.shape[1] != CONFIG.INPUT_WIDTH:
#             image = cv2.resize(image, (CONFIG.INPUT_WIDTH, CONFIG.INPUT_HEIGHT), 
#                         interpolation=cv2.INTER_LINEAR)
        
#         # Load density map 
#         density_map = np.load(density_path).astype(np.float32)
        
#         # Resize density map to correct size
#         target_height = CONFIG.OUTPUT_HEIGHT
#         target_width = CONFIG.OUTPUT_WIDTH
        
#         if density_map.shape != (target_height, target_width):
#             original_sum = np.sum(density_map)
#             density_map = cv2.resize(
#                 density_map, 
#                 (target_width, target_height),
#                 interpolation=cv2.INTER_LINEAR
#             )
#             new_sum = np.sum(density_map)
#             if new_sum > 0:
#                 density_map = density_map * (original_sum / new_sum)
        
#         # AUGMENTATION SECTION 
#         if augment:
#             # Horizontal flip (50% chance)
#             if np.random.random() > 0.5:
#                 image = cv2.flip(image, 1)
#                 density_map = cv2.flip(density_map, 1)
            
#             # Brightness adjustment (30% chance)
#             if np.random.random() > 0.7:
#                 factor = np.random.uniform(0.8, 1.2)
#                 image = np.clip(image * factor, 0, 255).astype(np.uint8)
            
#             # Slight rotation (-5 to +5 degrees, 20% chance)
#             if np.random.random() > 0.8:
#                 angle = np.random.uniform(-5, 5)
#                 h, w = image.shape[:2]
#                 M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
#                 image = cv2.warpAffine(image, M, (w, h))
#                 density_map = cv2.warpAffine(density_map, M, (target_width, target_height))
        
#         # Normalize image to [0, 1]
#         image = image.astype(np.float32) / 255.0
        
#         # Add channel dimension to density
#         density_map = np.expand_dims(density_map, axis=-1)
        
#         # Calculate count
#         count = np.sum(density_map).astype(np.float32)
        
#         return image, density_map, count
    
#     def create_dataset(self, part='A', split='train', batch_size=1, 
#                       shuffle=True, cache=True):
#         """
#         Create TensorFlow dataset
        
#         Args:
#             part: 'A' or 'B'
#             split: 'train' or 'test'
#             batch_size: Batch size
#             shuffle: Whether to shuffle
#             cache: Whether to cache dataset in memory
        
#         Returns:
#             tf.data.Dataset
#         """
        
#         # Get file pairs
#         file_pairs = self.get_file_pairs(part, split)
        
#         if len(file_pairs) == 0:
#             raise ValueError(f"No data found for Part {part} {split}")
        
#         # Create dataset from file paths
#         img_paths = [p[0] for p in file_pairs]
#         density_paths = [p[1] for p in file_pairs]
        
#         def _load_and_preprocess_tf(img_path, density_path):
#             """TensorFlow wrapper for loading"""
            
#             def _load_py(img_p, den_p):
#                 img_p = img_p.numpy().decode('utf-8')
#                 den_p = den_p.numpy().decode('utf-8')
#                 return self.load_and_preprocess(img_p, den_p, augment=False)
            
#             image, density, count = tf.py_function(
#                 func=_load_py,
#                 inp=[img_path, density_path],
#                 Tout=[tf.float32, tf.float32, tf.float32]
#             )
            
#             # Set shapes 
#             image.set_shape([CONFIG.INPUT_HEIGHT, CONFIG.INPUT_WIDTH, CONFIG.INPUT_CHANNELS])
#             density.set_shape([CONFIG.OUTPUT_HEIGHT, CONFIG.OUTPUT_WIDTH, CONFIG.OUTPUT_CHANNELS])
#             count.set_shape([])
            
#             return image, {'density_map': density, 'count': count}
        
#         # Create dataset
#         dataset = tf.data.Dataset.from_tensor_slices((img_paths, density_paths))
        
#         # Map loading 
#         dataset = dataset.map(
#             _load_and_preprocess_tf,
#             num_parallel_calls=tf.data.AUTOTUNE
#         )
        
#         # Cache 
#         if cache:
#             dataset = dataset.cache()
        
#         # Shuffle if training
#         if shuffle and split == 'train':
#             dataset = dataset.shuffle(buffer_size=min(100, len(file_pairs)))
        
#         # Batch
#         dataset = dataset.batch(batch_size)
        
#         # Prefetch for performance
#         dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
#         return dataset
    
#     def get_dataset_info(self, part='A'):
#         """
#         Get information about the dataset
        
#         Args:
#             part: 'A' or 'B'
        
#         Returns:
#             Dictionary with dataset information
#         """
        
#         train_pairs = self.get_file_pairs(part, 'train')
#         test_pairs = self.get_file_pairs(part, 'test')
        
#         info = {
#             'part': part,
#             'train_samples': len(train_pairs),
#             'test_samples': len(test_pairs),
#             'total_samples': len(train_pairs) + len(test_pairs),
#             'input_shape': (CONFIG.INPUT_HEIGHT, CONFIG.INPUT_WIDTH, CONFIG.INPUT_CHANNELS),
#             'output_shape': (CONFIG.OUTPUT_HEIGHT, CONFIG.OUTPUT_WIDTH, CONFIG.OUTPUT_CHANNELS)
#         }
        
#         return info


# def create_train_val_datasets(part='A', batch_size=1, val_split=0.1):
#     """
#     Create training and validation datasets
    
#     Args:
#         part: 'A' or 'B'
#         batch_size: Batch size
#         val_split: Fraction of training data to use for validation
    
#     Returns:
#         Tuple of (train_dataset, val_dataset, test_dataset)
#     """
    
#     loader = SimpleDataLoader()
    
#     # Get training file pairs
#     all_train_pairs = loader.get_file_pairs(part, 'train')
    
#     # Split into train/val 
#     if val_split > 0:
#         n_val = int(len(all_train_pairs) * val_split)
#         val_pairs = all_train_pairs[:n_val]
#         train_pairs = all_train_pairs[n_val:]
        
#         print(f"Split: {len(train_pairs)} train, {len(val_pairs)} validation")
        
#         # Create validation dataset 
#         val_img_paths = [p[0] for p in val_pairs]
#         val_density_paths = [p[1] for p in val_pairs]
        
#         # Loading function
#         def _load_and_preprocess_tf(img_path, density_path):
#             """TensorFlow wrapper for loading"""
            
#             def _load_py(img_p, den_p):
#                 """Python function for loading"""
#                 img_p = img_p.numpy().decode('utf-8')
#                 den_p = den_p.numpy().decode('utf-8')
#                 return loader.load_and_preprocess(img_p, den_p)
            
            
#             image, density, count = tf.py_function(
#                 func=_load_py,
#                 inp=[img_path, density_path],
#                 Tout=[tf.float32, tf.float32, tf.float32]
#             )
            
#             # Set shapes
#             image.set_shape([CONFIG.INPUT_HEIGHT, CONFIG.INPUT_WIDTH, CONFIG.INPUT_CHANNELS])
#             density.set_shape([CONFIG.OUTPUT_HEIGHT, CONFIG.OUTPUT_WIDTH, CONFIG.OUTPUT_CHANNELS])
#             count.set_shape([])
            
#             return image, {'density_map': density, 'count': count}
        
#         val_dataset = tf.data.Dataset.from_tensor_slices((val_img_paths, val_density_paths))
#         val_dataset = val_dataset.map(
#             _load_and_preprocess_tf,
#             num_parallel_calls=tf.data.AUTOTUNE
#         )
#         val_dataset = val_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        
       
#         train_img_paths = [p[0] for p in train_pairs]
#         train_density_paths = [p[1] for p in train_pairs]
        
#         train_dataset = tf.data.Dataset.from_tensor_slices((train_img_paths, train_density_paths))
#         train_dataset = train_dataset.map(
#             _load_and_preprocess_tf,
#             num_parallel_calls=tf.data.AUTOTUNE
#         )
#         train_dataset = train_dataset.shuffle(buffer_size=min(100, len(train_pairs)))
#         train_dataset = train_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
#     else:
#         val_dataset = None
#         # Create training dataset with all pairs
#         train_dataset = loader.create_dataset(part, 'train', batch_size, shuffle=True)
    
#     # Create test dataset
#     test_dataset = loader.create_dataset(part, 'test', batch_size, shuffle=False)
    
#     return train_dataset, val_dataset, test_dataset


# def test_data_loading():
#     """Test data loading functionality"""
    
#     loader = SimpleDataLoader()
    
#     # Test getting dataset info
#     for part in ['A', 'B']:
#         try:
#             info = loader.get_dataset_info(part)
#             print(f"\nPart {part} Dataset Info:")
#             for key, value in info.items():
#                 print(f"  {key}: {value}")
#         except Exception as e:
#             print(f"Error loading Part {part}: {e}")
    
#     # Test loading one batch
#     print("\nTesting batch loading...")
#     try:
#         dataset = loader.create_dataset('B', 'train', batch_size=1)
        
#         for batch in dataset.take(1):
#             images, targets = batch
#             print(f"Image shape: {images.shape}")
#             print(f"Density map shape: {targets['density_map'].shape}")
#             print(f"Count: {targets['count'].numpy()[0]:.2f}")
#             print(f"Density sum: {tf.reduce_sum(targets['density_map']).numpy():.2f}")
            
#             # Check value ranges
#             print(f"Image range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
#             print(f"Density range: [{tf.reduce_min(targets['density_map']):.6f}, "
#                   f"{tf.reduce_max(targets['density_map']):.6f}]")
            
#             # Verify shapes
#             assert images.shape == (1, 256, 256, 3), f"Wrong image shape: {images.shape}"
#             assert targets['density_map'].shape == (1, 32, 32, 1), f"Wrong density shape: {targets['density_map'].shape}"
            

        
#     except Exception as e:
#         print(f"\nData loading test failed: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     test_data_loading()



"""
Data loader for preprocessed dataset
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.single_scale_config import CONFIG

class SimpleDataLoader:
    """
    Simple loader for preprocessed ShanghaiTech dataset
    """
    
    def __init__(self, data_root=None):
        self.data_root = Path(data_root) if data_root else CONFIG.DATA_ROOT
        
    def get_file_pairs(self, part='A', split='train'):
        """
        Get pairs of (image_path, density_map_path)
        
        Args:
            part: 'A' or 'B'
            split: 'train' or 'test'
        
        Returns:
            List of (image_path, density_map_path) tuples
        """
        
        # Construct paths
        if part.lower() == 'mixed':
            base_path = self.data_root / "part_mixed"
        elif part == 'A':
            base_path = self.data_root / "part_A"
        else:  # part == 'B'
            base_path = self.data_root / "part_B"
        
        img_dir = base_path / split / "images"
        density_dir = base_path / split / "density_maps"
        
        # Check if directories exist
        if not img_dir.exists():
            raise ValueError(f"Image directory not found: {img_dir}")
        if not density_dir.exists():
            raise ValueError(f"Density directory not found: {density_dir}")
        
        # Get all image files
        img_files = sorted(list(img_dir.glob("*.png")))
        if len(img_files) == 0:
            img_files = sorted(list(img_dir.glob("*.jpg")))
        
        # Match with density maps
        file_pairs = []
        for img_path in img_files:
            density_path = density_dir / f"{img_path.stem}.npy"
            
            if density_path.exists():
                file_pairs.append((str(img_path), str(density_path)))
            else:
                print(f"Warning: No density map for {img_path.name}")
        
        print(f"Found {len(file_pairs)} image-density pairs for Part {part} {split}")
        return file_pairs
    
   

    def load_and_preprocess(self, img_path, density_path, augment=False):
        """
        Load and preprocess a single image-density pair with optional augmentation
        
        Args:
            img_path: Path to image file
            density_path: Path to density map file
            augment: Whether to apply data augmentation
        
        Returns:
            Tuple of (image, density_map, count)
        """
        
        # Load image 
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if image.shape[0] != CONFIG.INPUT_HEIGHT or image.shape[1] != CONFIG.INPUT_WIDTH:
            image = cv2.resize(image, (CONFIG.INPUT_WIDTH, CONFIG.INPUT_HEIGHT), 
                        interpolation=cv2.INTER_LINEAR)
        
        # Load density map 
        density_map = np.load(density_path).astype(np.float32)
        
        # Resize density map to correct size
        target_height = CONFIG.OUTPUT_HEIGHT
        target_width = CONFIG.OUTPUT_WIDTH
        
        if density_map.shape != (target_height, target_width):
            original_sum = np.sum(density_map)
            density_map = cv2.resize(
                density_map, 
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR
            )
            new_sum = np.sum(density_map)
            if new_sum > 0:
                density_map = density_map * (original_sum / new_sum)
        
        # AUGMENTATION SECTION 
        if augment:
            # Horizontal flip (50% chance)
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
                density_map = cv2.flip(density_map, 1)
            
            # Brightness adjustment (30% chance)
            if np.random.random() > 0.7:
                factor = np.random.uniform(0.8, 1.2)
                image = np.clip(image * factor, 0, 255).astype(np.uint8)
            
            # Slight rotation (-5 to +5 degrees, 20% chance)
            if np.random.random() > 0.8:
                angle = np.random.uniform(-5, 5)
                h, w = image.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h))
                density_map = cv2.warpAffine(density_map, M, (target_width, target_height))
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension to density
        density_map = np.expand_dims(density_map, axis=-1)
        
        # Calculate count
        count = np.sum(density_map).astype(np.float32)
        
        return image, density_map, count
    
    def create_dataset(self, part='A', split='train', batch_size=1, 
                      shuffle=True, cache=True):
        """
        Create TensorFlow dataset
        
        Args:
            part: 'A' or 'B'
            split: 'train' or 'test'
            batch_size: Batch size
            shuffle: Whether to shuffle
            cache: Whether to cache dataset in memory
        
        Returns:
            tf.data.Dataset
        """
        
        # Get file pairs
        file_pairs = self.get_file_pairs(part, split)
        
        if len(file_pairs) == 0:
            raise ValueError(f"No data found for Part {part} {split}")
        
        # Create dataset from file paths
        img_paths = [p[0] for p in file_pairs]
        density_paths = [p[1] for p in file_pairs]
        
        def _load_and_preprocess_tf(img_path, density_path):
            """TensorFlow wrapper for loading"""
            
            def _load_py(img_p, den_p):
                img_p = img_p.numpy().decode('utf-8')
                den_p = den_p.numpy().decode('utf-8')
                return self.load_and_preprocess(img_p, den_p, augment=CONFIG.USE_AUGMENTATION and split == 'train')
            
            image, density, count = tf.py_function(
                func=_load_py,
                inp=[img_path, density_path],
                Tout=[tf.float32, tf.float32, tf.float32]
            )
            
            # Set shapes 
            image.set_shape([CONFIG.INPUT_HEIGHT, CONFIG.INPUT_WIDTH, CONFIG.INPUT_CHANNELS])
            density.set_shape([CONFIG.OUTPUT_HEIGHT, CONFIG.OUTPUT_WIDTH, CONFIG.OUTPUT_CHANNELS])
            count.set_shape([])
            
            return image, {'density_map': density, 'count': count}
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, density_paths))
        
        # Map loading 
        dataset = dataset.map(
            _load_and_preprocess_tf,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Cache 
        if cache:
            dataset = dataset.cache()
        
        # Shuffle if training
        if shuffle and split == 'train':
            dataset = dataset.shuffle(buffer_size=min(100, len(file_pairs)))
        
        # Batch
        dataset = dataset.batch(batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_dataset_info(self, part='A'):
        """
        Get information about the dataset
        
        Args:
            part: 'A' or 'B'
        
        Returns:
            Dictionary with dataset information
        """
        
        train_pairs = self.get_file_pairs(part, 'train')
        test_pairs = self.get_file_pairs(part, 'test')
        
        info = {
            'part': part,
            'train_samples': len(train_pairs),
            'test_samples': len(test_pairs),
            'total_samples': len(train_pairs) + len(test_pairs),
            'input_shape': (CONFIG.INPUT_HEIGHT, CONFIG.INPUT_WIDTH, CONFIG.INPUT_CHANNELS),
            'output_shape': (CONFIG.OUTPUT_HEIGHT, CONFIG.OUTPUT_WIDTH, CONFIG.OUTPUT_CHANNELS)
        }
        
        return info


def create_train_val_datasets(part='A', batch_size=1, val_split=0.1):
    """
    Create training and validation datasets
    
    Args:
        part: 'A' or 'B'
        batch_size: Batch size
        val_split: Fraction of training data to use for validation
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    
    loader = SimpleDataLoader()
    
    # Get training file pairs
    all_train_pairs = loader.get_file_pairs(part, 'train')
    
    # Split into train/val 
    if val_split > 0:
        n_val = int(len(all_train_pairs) * val_split)
        val_pairs = all_train_pairs[:n_val]
        train_pairs = all_train_pairs[n_val:]
        
        print(f"Split: {len(train_pairs)} train, {len(val_pairs)} validation")
        
        # Create validation dataset 
        val_img_paths = [p[0] for p in val_pairs]
        val_density_paths = [p[1] for p in val_pairs]
        
        # Loading function
        def _load_and_preprocess_tf(img_path, density_path):
            """TensorFlow wrapper for loading"""
            
            def _load_py(img_p, den_p):
                """Python function for loading"""
                img_p = img_p.numpy().decode('utf-8')
                den_p = den_p.numpy().decode('utf-8')
                return loader.load_and_preprocess(img_p, den_p, augment=CONFIG.USE_AUGMENTATION)
            
            
            image, density, count = tf.py_function(
                func=_load_py,
                inp=[img_path, density_path],
                Tout=[tf.float32, tf.float32, tf.float32]
            )
            
            # Set shapes
            image.set_shape([CONFIG.INPUT_HEIGHT, CONFIG.INPUT_WIDTH, CONFIG.INPUT_CHANNELS])
            density.set_shape([CONFIG.OUTPUT_HEIGHT, CONFIG.OUTPUT_WIDTH, CONFIG.OUTPUT_CHANNELS])
            count.set_shape([])
            
            return image, {'density_map': density, 'count': count}
        
        val_dataset = tf.data.Dataset.from_tensor_slices((val_img_paths, val_density_paths))
        val_dataset = val_dataset.map(
            _load_and_preprocess_tf,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_dataset = val_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        
       
        train_img_paths = [p[0] for p in train_pairs]
        train_density_paths = [p[1] for p in train_pairs]
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_img_paths, train_density_paths))
        train_dataset = train_dataset.map(
            _load_and_preprocess_tf,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.shuffle(buffer_size=min(100, len(train_pairs)))
        train_dataset = train_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        val_dataset = None
        # Create training dataset with all pairs
        train_dataset = loader.create_dataset(part, 'train', batch_size, shuffle=True)
    
    # Create test dataset
    test_dataset = loader.create_dataset(part, 'test', batch_size, shuffle=False)
    
    return train_dataset, val_dataset, test_dataset


def test_data_loading():
    """Test data loading functionality"""
    
    loader = SimpleDataLoader()
    
    # Test getting dataset info
    for part in ['A', 'B']:
        try:
            info = loader.get_dataset_info(part)
            print(f"\nPart {part} Dataset Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error loading Part {part}: {e}")
    
    # Test loading one batch
    print("\nTesting batch loading...")
    try:
        dataset = loader.create_dataset('B', 'train', batch_size=1)
        
        for batch in dataset.take(1):
            images, targets = batch
            print(f"Image shape: {images.shape}")
            print(f"Density map shape: {targets['density_map'].shape}")
            print(f"Count: {targets['count'].numpy()[0]:.2f}")
            print(f"Density sum: {tf.reduce_sum(targets['density_map']).numpy():.2f}")
            
            # Check value ranges
            print(f"Image range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
            print(f"Density range: [{tf.reduce_min(targets['density_map']):.6f}, "
                  f"{tf.reduce_max(targets['density_map']):.6f}]")
            
            # Verify shapes
            assert images.shape == (1, 256, 256, 3), f"Wrong image shape: {images.shape}"
            assert targets['density_map'].shape == (1, 32, 32, 1), f"Wrong density shape: {targets['density_map'].shape}"
            

        
    except Exception as e:
        print(f"\nData loading test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_loading()