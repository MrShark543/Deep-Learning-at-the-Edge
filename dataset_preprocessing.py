
"""
Creates scaled RGB images, ground truth coordinates, and density maps
"""

import numpy as np
import cv2
import json
import scipy.io
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import shutil
from datetime import datetime

class DatasetPreprocessor:
    """Preprocess ShanghaiTech dataset"""
    
    def __init__(self, source_root="./datasets/shanghaitech", 
                 target_root="./datasets/shanghaitech_256x256_rgb",
                 target_size=(256, 256)):
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.target_size = target_size
        self.density_size = (target_size[0] // 8, target_size[1] // 8)  
        
    def create_directory_structure(self):
        """Create the target directory structure"""
        for part in ['part_A', 'part_B', 'part_mixed']:
            for split in ['train', 'test']:
                for folder in ['images', 'ground_truth', 'density_maps']:
                    dir_path = self.target_root / part / split / folder
                    dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Created directory structure at {self.target_root}")
    
    def load_original_annotations(self, mat_path):
        """Load annotations from original .mat file"""
        try:
            mat_data = scipy.io.loadmat(str(mat_path))
            
            if 'image_info' in mat_data:
                image_info = mat_data['image_info']
                
                if image_info.size > 0:
                    content = image_info[0, 0]
                    
                    if hasattr(content, 'dtype') and content.dtype.names:
                        if 'location' in content.dtype.names:
                            locations = content['location'][0, 0]
                            
                            if len(locations.shape) == 2 and locations.shape[1] >= 2:
                                return locations[:, :2].astype(np.float32)
            
            return np.empty((0, 2), dtype=np.float32)
            
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            return np.empty((0, 2), dtype=np.float32)
    
    def apply_padding_and_resize(self, image, points):
        """Apply padding strategy and resize to target size """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate padding 
        max_dim = max(h, w)
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        
        # Ensure image is RGB 
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # Pad image with black borders
        padded_image = cv2.copyMakeBorder(
            image, pad_h, max_dim - h - pad_h,
            pad_w, max_dim - w - pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        # Resize to target size
        resized_image = cv2.resize(padded_image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Scale and adjust points
        scale = target_h / max_dim
        scaled_points = []
        
        for point in points:
            new_x = (point[0] + pad_w) * scale
            new_y = (point[1] + pad_h) * scale
            
            new_x = np.clip(new_x, 0, target_w - 1)
            new_y = np.clip(new_y, 0, target_h - 1)
            
            scaled_points.append([new_x, new_y])
        
        scaled_points = np.array(scaled_points, dtype=np.float32) if scaled_points else np.empty((0, 2), dtype=np.float32)
        
        return resized_image, scaled_points, (h, w)
    
    def generate_density_map_adaptive(self, points, beta=0.3, k=3):
        """Generate adaptive density map for Part A"""
        density_map = np.zeros(self.density_size, dtype=np.float32)
        
        if len(points) == 0:
            return density_map
        
        # Scale points to density map resolution (32x32 from 256x256)
        scale_factor = self.density_size[0] / self.target_size[0]  
        scaled_points = points * scale_factor
        
        # Build KDTree for k-NN
        tree = KDTree(scaled_points)
        
        for i, point in enumerate(scaled_points):
            x, y = point
            
            # Calculate adaptive sigma
            if len(scaled_points) > k:
                distances, _ = tree.query(point, k=k+1)
                avg_distance = np.mean(distances[1:k+1])
                sigma = avg_distance * beta
            else:
                sigma = 1.0
            
            # Clamp sigma to reasonable range
            sigma = max(0.5, min(sigma, 4.0))
            
            # Generate Gaussian
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            kernel = self.gaussian_kernel_2d(kernel_size, sigma)
            
            # Add to density map
            x_int, y_int = int(x), int(y)
            half_size = kernel_size // 2
            
            y_min = max(0, y_int - half_size)
            y_max = min(self.density_size[0], y_int + half_size + 1)
            x_min = max(0, x_int - half_size)
            x_max = min(self.density_size[1], x_int + half_size + 1)
            
            ky_min = y_min - (y_int - half_size)
            ky_max = kernel_size - ((y_int + half_size + 1) - y_max)
            kx_min = x_min - (x_int - half_size)
            kx_max = kernel_size - ((x_int + half_size + 1) - x_max)
            
            density_map[y_min:y_max, x_min:x_max] += kernel[ky_min:ky_max, kx_min:kx_max]

        # Normalize to preserve count
        if len(points) > 0:
            current_sum = np.sum(density_map)
            if current_sum > 0:
                density_map = density_map * (len(points) / current_sum)
        
        return density_map
    
    def generate_density_map_fixed(self, points, sigma=4.0):
        """Generate fixed sigma density map for Part B"""
        density_map = np.zeros(self.density_size, dtype=np.float32)
        
        if len(points) == 0:
            return density_map
        
        # Scale points to density map resolution
        scale_factor = self.density_size[0] / self.target_size[0]
        scaled_points = points * scale_factor
        
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = self.gaussian_kernel_2d(kernel_size, sigma)
        
        for point in scaled_points:
            x, y = point
            x_int, y_int = int(x), int(y)
            half_size = kernel_size // 2
            
            y_min = max(0, y_int - half_size)
            y_max = min(self.density_size[0], y_int + half_size + 1)
            x_min = max(0, x_int - half_size)
            x_max = min(self.density_size[1], x_int + half_size + 1)
            
            ky_min = y_min - (y_int - half_size)
            ky_max = kernel_size - ((y_int + half_size + 1) - y_max)
            kx_min = x_min - (x_int - half_size)
            kx_max = kernel_size - ((x_int + half_size + 1) - x_max)
            
            density_map[y_min:y_max, x_min:x_max] += kernel[ky_min:ky_max, kx_min:kx_max]

        # Normalize
        if len(points) > 0:
            current_sum = np.sum(density_map)
            if current_sum > 0:
                density_map = density_map * (len(points) / current_sum)
                
        return density_map
    
    def gaussian_kernel_2d(self, size, sigma):
        """Create a 2D Gaussian kernel"""
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                distance_sq = (i - center) ** 2 + (j - center) ** 2
                kernel[i, j] = np.exp(-distance_sq / (2 * sigma ** 2))
        
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def process_single_image(self, img_path, gt_path, target_img_path, 
                           target_gt_path, target_den_path, part='A'):
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not load {img_path}")
            return False
        
        original_points = self.load_original_annotations(gt_path)
        
        scaled_image, scaled_points, original_size = self.apply_padding_and_resize(
            image, original_points
        )
        
        if part == 'A':
            density_map = self.generate_density_map_adaptive(scaled_points)
        else:
            density_map = self.generate_density_map_fixed(scaled_points)
        
        original_count = len(original_points)
        scaled_count = len(scaled_points)
        density_sum = np.sum(density_map)
        
        cv2.imwrite(str(target_img_path), scaled_image)
        
        gt_data = {
            'image_name': target_img_path.name,
            'image_size': list(self.target_size) + [3],
            'count': int(scaled_count),
            'points': scaled_points.tolist(),
            'original_count': int(original_count),
            'original_size': list(original_size),
            'density_sum': float(density_sum),
            'density_error': float(abs(density_sum - scaled_count) / max(scaled_count, 1) * 100)
        }
        
        with open(target_gt_path, 'w') as f:
            json.dump(gt_data, f, indent=2)
        
        np.save(target_den_path, density_map)
        
        return True
    
    def process_dataset(self, process_mixed=True):
        """Process the entire dataset"""
        
        print("\n" + "="*60)
        print("STARTING DATASET PREPROCESSING")
        print(f"Source: {self.source_root}")
        print(f"Target: {self.target_root}")
        print(f"Target size: {self.target_size} RGB")
        print(f"Density size: {self.density_size}")
        print("="*60)
        
        self.create_directory_structure()
        
        stats = {
            'part_A': {'train': 0, 'test': 0, 'errors': []},
            'part_B': {'train': 0, 'test': 0, 'errors': []},
            'part_mixed': {'train': 0, 'test': 0}
        }
        
        mixed_train_images = []
        mixed_test_images = []
        
        for part in ['A', 'B']:
            print(f"\nProcessing Part {part}")
            
            source_part = f"part_{part}_final"
            target_part = f"part_{part}"
            
            for split in ['train', 'test']:
                print(f"\n{split.upper()} set:")
                
                source_img_dir = self.source_root / source_part / f"{split}_data" / "images"
                source_gt_dir = self.source_root / source_part / f"{split}_data" / "ground_truth"
                
                target_img_dir = self.target_root / target_part / split / "images"
                target_gt_dir = self.target_root / target_part / split / "ground_truth"
                target_den_dir = self.target_root / target_part / split / "density_maps"
                
                img_files = sorted(list(source_img_dir.glob("*.jpg")))
                
                if not img_files:
                    print(f"  No images found in {source_img_dir}")
                    continue
                
                for img_path in tqdm(img_files, desc=f"Processing Part {part} {split}"):
                    gt_name = "GT_" + img_path.name.replace('.jpg', '.mat')
                    gt_path = source_gt_dir / gt_name
                    
                    if not gt_path.exists():
                        stats[f'part_{part}']['errors'].append(img_path.name)
                        continue
                    
                    target_img_path = target_img_dir / img_path.name.replace('.jpg', '.png')
                    target_gt_path = target_gt_dir / img_path.name.replace('.jpg', '.json')
                    target_den_path = target_den_dir / img_path.name.replace('.jpg', '.npy')
                    
                    success = self.process_single_image(
                        img_path, gt_path, 
                        target_img_path, target_gt_path, target_den_path,
                        part=part
                    )
                    
                    if success:
                        stats[f'part_{part}'][split] += 1
                        
                        if process_mixed:
                            if split == 'train':
                                mixed_train_images.append((target_img_path, target_gt_path, target_den_path, part))
                            else:
                                mixed_test_images.append((target_img_path, target_gt_path, target_den_path, part))
                
                print(f"  Processed: {stats[f'part_{part}'][split]} images")
        
        if process_mixed:
            print("\nCreating Mixed Dataset")
            self.create_mixed_dataset(mixed_train_images, mixed_test_images, stats)
        
        self.save_preprocessing_info(stats)
        self.print_summary(stats)
        
        return stats
    
    def create_mixed_dataset(self, train_images, test_images, stats):
        """Create a mixed dataset combining Part A and Part B"""
        
        mixed_train_img = self.target_root / "part_mixed" / "train" / "images"
        mixed_train_gt = self.target_root / "part_mixed" / "train" / "ground_truth"
        mixed_train_den = self.target_root / "part_mixed" / "train" / "density_maps"
        
        mixed_test_img = self.target_root / "part_mixed" / "test" / "images"
        mixed_test_gt = self.target_root / "part_mixed" / "test" / "ground_truth"
        mixed_test_den = self.target_root / "part_mixed" / "test" / "density_maps"
        
        print("Creating mixed training set")
        for img_path, gt_path, den_path, part in tqdm(train_images):
            new_name = f"{part}_{img_path.name}"
            
            shutil.copy2(img_path, mixed_train_img / new_name)
            shutil.copy2(gt_path, mixed_train_gt / new_name.replace('.png', '.json'))
            shutil.copy2(den_path, mixed_train_den / new_name.replace('.png', '.npy'))
            
            stats['part_mixed']['train'] += 1
        
        print("Creating mixed test set")
        for img_path, gt_path, den_path, part in tqdm(test_images):
            new_name = f"{part}_{img_path.name}"
            
            shutil.copy2(img_path, mixed_test_img / new_name)
            shutil.copy2(gt_path, mixed_test_gt / new_name.replace('.png', '.json'))
            shutil.copy2(den_path, mixed_test_den / new_name.replace('.png', '.npy'))
            
            stats['part_mixed']['test'] += 1
        
        print(f"Mixed dataset created: {stats['part_mixed']['train']} train, {stats['part_mixed']['test']} test")
    
    def save_preprocessing_info(self, stats):
        """Save preprocessing metadata"""
        info = {
            'date': datetime.now().isoformat(),
            'source_root': str(self.source_root),
            'target_root': str(self.target_root),
            'target_size': list(self.target_size) + [3],
            'density_size': list(self.density_size),
            'preprocessing': 'padding + RGB (256x256)',
            'statistics': stats,
            'notes': 'Density maps are 32x32 (1/8 of input), normalized to preserve count'
        }
        
        info_path = self.target_root / 'preprocessing_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def print_summary(self, stats):
        """Print preprocessing summary"""
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        total = 0
        for part in ['part_A', 'part_B', 'part_mixed']:
            if part in stats:
                train_count = stats[part]['train']
                test_count = stats[part]['test']
                error_count = len(stats[part].get('errors', []))
                part_total = train_count + test_count
                total += part_total if part != 'part_mixed' else 0
                
                print(f"\n{part}:")
                print(f"  Train: {train_count} images")
                print(f"  Test: {test_count} images")
                print(f"  Total: {part_total} images")
                
                if error_count > 0:
                    print(f"  Errors: {error_count} images skipped")
        
        print(f"\nTotal unique images processed: {total}")
        print(f"Dataset saved to: {self.target_root}")
    
    def verify_dataset(self, num_samples=3):
        """Verify the preprocessed dataset"""
        print("\n" + "="*60)
        print("VERIFYING PREPROCESSED DATASET")
        print("="*60)
        
        for part in ['A', 'B']:
            print(f"\nPart {part}")
            
            img_dir = self.target_root / f"part_{part}" / "train" / "images"
            gt_dir = self.target_root / f"part_{part}" / "train" / "ground_truth"
            den_dir = self.target_root / f"part_{part}" / "train" / "density_maps"
            
            if not img_dir.exists():
                print(f"  Directory not found: {img_dir}")
                continue
            
            img_files = sorted(list(img_dir.glob("*.png")))[:num_samples]
            
            for img_path in img_files:
                image = cv2.imread(str(img_path))
                
                gt_path = gt_dir / img_path.name.replace('.png', '.json')
                with open(gt_path, 'r') as f:
                    gt_data = json.load(f)
                
                den_path = den_dir / img_path.name.replace('.png', '.npy')
                density_map = np.load(den_path)
                
                print(f"\n{img_path.name}:")
                print(f"  Image shape: {image.shape} (should be 256×256×3)")
                print(f"  Point count: {gt_data['count']}")
                print(f"  Density shape: {density_map.shape} (should be 32×32)")
                print(f"  Density sum: {np.sum(density_map):.2f}")
                print(f"  Count vs sum diff: {abs(gt_data['count'] - np.sum(density_map)):.4f}")
                
                assert image.shape == (256, 256, 3), f"Image shape mismatch!"
                assert density_map.shape == (32, 32), f"Density shape mismatch!"
                assert abs(np.sum(density_map) - gt_data['count']) < 0.1, "Density sum doesn't match count!"
        
        print("\nVerification complete!")
def verify_no_data_leakage():
    """To verify there's no overlap between train and test sets"""
    
    target_root = Path("./datasets/shanghaitech_256x256_rgb")
    
    for part in ['A', 'B']:
        print(f"\nVerifying Part {part} data split:")
        
        train_dir = target_root / f"part_{part}" / "train" / "images"
        test_dir = target_root / f"part_{part}" / "test" / "images"
        
        if not train_dir.exists() or not test_dir.exists():
            print(f"  Directories not found for Part {part}")
            continue
            
        train_files = set([f.stem for f in train_dir.glob("*.png")])
        test_files = set([f.stem for f in test_dir.glob("*.png")])
        
        # Check for overlap
        overlap = train_files.intersection(test_files)
        
        print(f"  Train samples: {len(train_files)}")
        print(f"  Test samples: {len(test_files)}")
        
        if overlap:
            print(f"  WARNING: {len(overlap)} files appear in both sets!")
            print(f"  Overlapping files: {list(overlap)[:5]}")
        else:
            print(f"  No overlap detected")
        
        # Verify against expected counts
        if part == 'A':
            if len(train_files) != 300:
                print(f"  Expected 300 train images, got {len(train_files)}")
            if len(test_files) != 182:
                print(f"  Expected 182 test images, got {len(test_files)}")
        elif part == 'B':
            if len(train_files) != 400:
                print(f"  Expected 400 train images, got {len(train_files)}")
            if len(test_files) != 316:
                print(f"  Expected 316 test images, got {len(test_files)}")


def preprocess_shanghaitech_dataset_256():
    """Main function to preprocess the entire dataset for 256x256"""
    
    preprocessor = DatasetPreprocessor(
        source_root="./datasets/shanghaitech",
        target_root="./datasets/shanghaitech_256x256_rgb",
        target_size=(256, 256)
    )
    
    stats = preprocessor.process_dataset(process_mixed=True)
    preprocessor.verify_dataset(num_samples=2)
    
    print("\nDataset preprocessing complete!")
    print("256×256 RGB dataset ready!")
    print("Density maps are 32×32 (1/8 scale)")
    
    return preprocessor

if __name__ == "__main__":
    preprocessor = preprocess_shanghaitech_dataset_256()

    verify_no_data_leakage()