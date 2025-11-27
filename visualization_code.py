"""
INT8 Model Visualization for Pruned Models
comparing actual vs predicted density maps
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class INT8ModelVisualizer:
    """Visualizer for INT8 quantized models"""
    
    def __init__(self, models_dir: str, test_data_path: str, verbose: bool = False):
        self.models_dir = Path(models_dir)
        self.test_data_path = Path(test_data_path)
        self.verbose = verbose
        self.test_images = None
        self.ground_truth = None
        self.ground_truth_maps = None
        self.models_by_sparsity = defaultdict(dict)
        self._load_test_data()
        self._organize_models()
    
    def _load_test_data(self):
        """Load test data from npz file"""
        data = np.load(self.test_data_path)
        self.test_images = data['images']
        
        # Handle 5D arrays
        if len(self.test_images.shape) == 5:
            self.test_images = self.test_images.squeeze(1)
        
        self.ground_truth = data['counts'].squeeze()
        
        # Look for density maps
        density_field_names = ['density_maps',  'density',]
        self.ground_truth_maps = None
        
        for field_name in density_field_names:
            if field_name in data:
                self.ground_truth_maps = data[field_name]
                if len(self.ground_truth_maps.shape) == 5:
                    self.ground_truth_maps = self.ground_truth_maps.squeeze(1)
                print(f"Loaded {len(self.test_images)} test images with density maps (field: '{field_name}')")
                break
        
        if self.ground_truth_maps is None:
            print(f"Loaded {len(self.test_images)} test images (no density maps found)")
    
    def _parse_model_filename(self, filename: str) -> Dict:
        """Parse model filename to extract info"""
        info = {
            'sparsity': 0.0,
            'quantization': 'unknown',
            'is_baseline': False,
            'is_int8': False
        }
        
        filename_lower = filename.lower()
        
        # Check if INT8
        if 'int8' not in filename_lower:
            return info
        
        info['is_int8'] = True
        
        # Determine INT8 type
        if 'int8_full' in filename_lower:
            info['quantization'] = 'int8_full'
        elif 'int8_hybrid' in filename_lower:
            info['quantization'] = 'int8_hybrid'
        elif 'int8_fallback' in filename_lower:
            info['quantization'] = 'int8_fallback'
        else:
            info['quantization'] = 'int8'
        
        # Check if baseline
        if 'baseline' in filename_lower:
            info['is_baseline'] = True
            return info
        
        # Extract sparsity
        sparsity_match = re.search(r'pruned_(\d+)pct', filename_lower)
        if sparsity_match:
            info['sparsity'] = float(sparsity_match.group(1)) / 100
        
        return info
    
    def _organize_models(self):
        """Organize INT8 models by sparsity level"""
        all_models = list(self.models_dir.glob("*.tflite"))
        print(f"\nFound {len(all_models)} total TFLite models")
        
        int8_count = 0
        for model_path in all_models:
            info = self._parse_model_filename(model_path.stem)
            
            if not info['is_int8']:
                continue
            
            int8_count += 1
            
            # Determine sparsity key
            if info['is_baseline']:
                sparsity_key = 'baseline'
            elif info['sparsity'] > 0:
                sparsity_key = f"{int(info['sparsity'] * 100)}pct"
            else:
                continue
            
            # Store model
            quant_type = info['quantization']
            self.models_by_sparsity[sparsity_key][quant_type] = model_path
        
        print(f"Found {int8_count} INT8 models")
        print("\nINT8 models by sparsity:")
        for sparsity in sorted(self.models_by_sparsity.keys()):
            models = self.models_by_sparsity[sparsity]
            print(f"  {sparsity}: {len(models)} variants")
    
    def _run_inference(self, model_path: Path, input_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Run inference on a model"""
        try:
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            
            # Prepare input
            input_data = input_image[np.newaxis, ...].astype(np.float32)
            
            # Handle INT8 quantization
            if input_details['dtype'] == np.int8:
                if input_details.get('quantization'):
                    scale, zero_point = input_details['quantization'][:2]
                    if scale != 0:
                        input_data = input_data / scale + zero_point
                        input_data = np.clip(input_data, -128, 127).astype(np.int8)
            
            # Run inference
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details['index'])
            
            # Dequantize if needed
            if output_details['dtype'] == np.int8:
                if output_details.get('quantization'):
                    scale, zero_point = output_details['quantization'][:2]
                    if scale != 0:
                        output_data = (output_data.astype(np.float32) - zero_point) * scale
            
            # Process output
            density_map = output_data.squeeze()
            
            if density_map.ndim != 2:
                return None, np.sum(density_map) if density_map.size > 0 else None
            
            return density_map, np.sum(density_map)
            
        except Exception as e:
            if self.verbose:
                print(f"Error with {model_path.name}: {e}")
            return None, None
    
    def visualize_models(self, sample_indices: Optional[List[int]] = None):
        """Create visualizations for INT8 models"""
        if sample_indices is None:
            sample_indices = [0, 1, 2]
        
        # Get sparsity levels
        all_levels = list(self.models_by_sparsity.keys())
        baseline = [l for l in all_levels if l == 'baseline']
        percentages = [l for l in all_levels if 'pct' in l]
        percentages.sort(key=lambda x: int(x.replace('pct', '')))
        sparsity_levels = baseline + percentages
        
        print(f"\nProcessing sparsity levels: {sparsity_levels}")
        
        # Process each sparsity level
        for sparsity in sparsity_levels:
            if sparsity not in self.models_by_sparsity:
                continue
            
            models = self.models_by_sparsity[sparsity]
            print(f"\nProcessing {sparsity} with {len(models)} INT8 variants...")
            
            for sample_idx in sample_indices:
                self._create_figure(sparsity, models, sample_idx)
    
    def _create_figure(self, sparsity: str, models: Dict[str, Path], sample_idx: int):
        """Create visualization figure"""
        # Get INT8 types in order
        int8_order = ['int8_full', 'int8_hybrid', 'int8_fallback']
        available_int8 = [q for q in int8_order if q in models]
        if not available_int8:
            available_int8 = sorted(models.keys())
        
        num_models = len(available_int8)
        if num_models == 0:
            return
        
        # Create figure
        fig, axes = plt.subplots(num_models, 4, figsize=(16, 4*num_models))
        if num_models == 1:
            axes = axes.reshape(1, -1)
        
        # Get test data
        test_image = self.test_images[sample_idx]
        true_count = self.ground_truth[sample_idx]
        
        # Get ground truth density map
        if self.ground_truth_maps is not None and sample_idx < len(self.ground_truth_maps):
            gt_density = self.ground_truth_maps[sample_idx].squeeze()
        else:
            # Create placeholder
            gt_density = np.zeros((test_image.shape[0]//8, test_image.shape[1]//8))
        
        # Column titles
        col_titles = ['Input Image', 'Actual Density Map', 'Predicted Density Map', 'Count Comparison']
        
        # Process each model
        for row_idx, int8_type in enumerate(available_int8):
            model_path = models[int8_type]
            pred_density, pred_count = self._run_inference(model_path, test_image)
            
            # Column 0: Input Image
            axes[row_idx, 0].imshow(test_image)
            axes[row_idx, 0].axis('off')
            axes[row_idx, 0].text(0.5, -0.08, f'True Count: {true_count:.0f}',
                                transform=axes[row_idx, 0].transAxes,
                                ha='center', fontsize=12, fontweight='bold')
            if row_idx == 0:
                axes[row_idx, 0].set_title(col_titles[0], fontsize=12, fontweight='bold', pad=10)
            
            # Column 1: Actual Density Map
            if gt_density.ndim == 2:
                vmax_gt = np.max(gt_density) if np.max(gt_density) > 0 else 1
                im1 = axes[row_idx, 1].imshow(gt_density, cmap='jet', vmin=0, vmax=vmax_gt)
                axes[row_idx, 1].axis('off')
                
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider1 = make_axes_locatable(axes[row_idx, 1])
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1)
            else:
                axes[row_idx, 1].text(0.5, 0.5, 'Not Available',
                                     ha='center', va='center',
                                     transform=axes[row_idx, 1].transAxes)
                axes[row_idx, 1].axis('off')
            
            if row_idx == 0:
                axes[row_idx, 1].set_title(col_titles[1], fontsize=12, fontweight='bold', pad=10)
            
            # Column 2: Predicted Density Map
            if pred_density is not None and pred_density.ndim == 2:
                vmax_pred = np.max(pred_density) if np.max(pred_density) > 0 else 1
                im2 = axes[row_idx, 2].imshow(pred_density, cmap='jet', vmin=0, vmax=vmax_pred)
                axes[row_idx, 2].axis('off')
                
                divider2 = make_axes_locatable(axes[row_idx, 2])
                cax2 = divider2.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im2, cax=cax2)
            else:
                axes[row_idx, 2].text(0.5, 0.5, 'No Density Map',
                                     ha='center', va='center',
                                     transform=axes[row_idx, 2].transAxes)
                axes[row_idx, 2].axis('off')
            
            if row_idx == 0:
                axes[row_idx, 2].set_title(col_titles[2], fontsize=12, fontweight='bold', pad=10)
            
            # Column 3: Count Comparison
            if pred_count is not None:
                error = pred_count - true_count
                model_size_mb = model_path.stat().st_size / (1024 * 1024)
                
                # Display metrics
                axes[row_idx, 3].text(0.5, 0.75, int8_type.replace('_', ' ').upper(),
                                    ha='center', va='center', fontsize=12, fontweight='bold',
                                    transform=axes[row_idx, 3].transAxes)
                
                axes[row_idx, 3].text(0.5, 0.50, f'Predicted: {pred_count:.1f}',
                                    ha='center', va='center', fontsize=13,
                                    transform=axes[row_idx, 3].transAxes)
                
                axes[row_idx, 3].text(0.5, 0.35, f'True: {true_count:.0f}',
                                    ha='center', va='center', fontsize=13,
                                    transform=axes[row_idx, 3].transAxes)
                
                axes[row_idx, 3].text(0.5, 0.20, f'Error: {error:+.1f}',
                                    ha='center', va='center', fontsize=13,
                                    transform=axes[row_idx, 3].transAxes)
                
                axes[row_idx, 3].text(0.5, 0.05, f'{model_size_mb:.2f} MB',
                                    ha='center', va='center', fontsize=10, color='gray',
                                    transform=axes[row_idx, 3].transAxes)
            else:
                axes[row_idx, 3].text(0.5, 0.5, 'Failed',
                                    ha='center', va='center', color='red', fontsize=14,
                                    transform=axes[row_idx, 3].transAxes)
            
            axes[row_idx, 3].axis('off')
            
            if row_idx == 0:
                axes[row_idx, 3].set_title(col_titles[3], fontsize=12, fontweight='bold', pad=10)
            
            # Row label
            row_label = int8_type.replace('int8_', '').upper()
            axes[row_idx, 0].text(-0.12, 0.5, row_label,
                                transform=axes[row_idx, 0].transAxes,
                                fontsize=11, fontweight='bold',
                                rotation=90, va='center', ha='center')
        
        # Main title
        sparsity_label = 'Baseline (No Pruning)' if sparsity == 'baseline' else f'{sparsity} Sparsity'
        plt.suptitle(f'{sparsity_label} - Sample {sample_idx}\nINT8 Quantization Comparison',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save
        output_filename = f'int8_{sparsity}_sample_{sample_idx}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Saved: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="INT8 model visualization")
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--samples', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    visualizer = INT8ModelVisualizer(args.models_dir, args.test_data, args.verbose)
    visualizer.visualize_models(args.samples)

if __name__ == "__main__":
    main()