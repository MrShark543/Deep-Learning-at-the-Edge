# """
# Post-Training Quantization Pipeline for Crowd Counting Models
# Note:Evaluation takes time to run and gives incorrect results if run on a windows computer however quantization will happen normally

# """

# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import json
# import time
# import sys
# from datetime import datetime
# from tqdm import tqdm  

# # Add project paths
# sys.path.append(str(Path(__file__).parent.parent))

# from config.single_scale_config import CONFIG
# from models.single_scale_vgg import SingleScaleSACNN, create_single_scale_model
# from models.losses import euclidean_loss, relative_count_loss, mae_count, mse_count, rmse_count
# from data.simple_loader import SimpleDataLoader, create_train_val_datasets

# class PostTrainingQuantizer:
#     """Post-training quantization pipeline """
    
#     def __init__(self, experiment_name=None):
#         self.experiment_name = experiment_name or f"quantization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         self.output_dir = Path("quantization_results") / self.experiment_name
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Custom objects for model loading
#         self.custom_objects = {
#             'SingleScaleSACNN': SingleScaleSACNN,
#             'euclidean_loss': euclidean_loss,
#             'relative_count_loss': relative_count_loss,
#             'mae_count': mae_count,
#             'mse_count': mse_count,
#             'rmse_count': rmse_count
#         }
        
#         self.results = {}
        
#     def load_model(self, model_path):
#         """Load model """
#         try:
#             model = tf.keras.models.load_model(
#                 model_path, 
#                 custom_objects=self.custom_objects,
#                 compile=False
#             )
#             print(f"Loaded model: {model_path}")
#             return model
#         except Exception as e:
#             print(f"Failed to load as saved model: {e}")
#             print("Attempting to load weights into architecture...")
#             model = create_single_scale_model(input_shape=(128, 128, 3))
#             model.load_weights(model_path)
#             return model
    

#     def get_calibration_dataset(self, part='B', num_samples=100):
#         """Get representative calibration data"""
#         loader = SimpleDataLoader()
#         train_dataset = loader.create_dataset(part, 'train', batch_size=1, shuffle=True)
       
#         def representative_dataset():
#             """Generator with better coverage"""
#             count = 0
#             for dataset in [train_dataset]:
#                 for images, _ in dataset:
#                     if count >= num_samples:
#                         break
#                     # Ensure proper normalization
#                     img = images.numpy()
#                     # Verify the range is [0, 1]
#                     img = np.clip(img, 0.0, 1.0)
#                     yield [img.astype(np.float32)]
#                     count += 1
#                     if count >= num_samples:
#                         break
    
#         return representative_dataset, train_dataset
    
#     def quantize_model(self, model, method, calibration_fn=None):
#         """Apply specific quantization method"""
#         converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
#         if method == 'float32':
#             pass
            
#         elif method == 'dynamic_range':
#             converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
#         elif method == 'float16':
#             converter.optimizations = [tf.lite.Optimize.DEFAULT]
#             converter.target_spec.supported_types = [tf.float16]
            
#         if method == 'int8_full':
#             if calibration_fn is None:
#                 raise ValueError("INT8 quantization requires calibration data")
#             converter.optimizations = [tf.lite.Optimize.DEFAULT]
#             converter.representative_dataset = calibration_fn
#             converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#             converter.inference_input_type = tf.int8
#             converter.inference_output_type = tf.int8
            
#         elif method == 'int8_hybrid':
#             if calibration_fn is None:
#                 raise ValueError("INT8 quantization requires calibration data")
#             converter.optimizations = [tf.lite.Optimize.DEFAULT]
#             converter.representative_dataset = calibration_fn
#             # Float I/O for better accuracy
            
#         elif method == 'int8_fallback':
#             if calibration_fn is None:
#                 raise ValueError("INT8 quantization requires calibration data")
#             converter.optimizations = [tf.lite.Optimize.DEFAULT]
#             converter.representative_dataset = calibration_fn
#             converter.target_spec.supported_ops = [
#                 tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
#                 tf.lite.OpsSet.SELECT_TF_OPS  # Allow TF ops as fallback
#             ]
#             converter.allow_custom_ops = True
            
#         else:
#             raise ValueError(f"Unknown quantization method: {method}")
        
#         return converter.convert()
    

#     def evaluate_tflite_model(self, tflite_model, test_dataset, num_samples=50, model_name="", method=""):
#         """Evaluate quantized model accuracy and speed"""
#         interpreter = tf.lite.Interpreter(model_content=tflite_model)
#         interpreter.allocate_tensors()
        
#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()
        
#         predictions = []
#         ground_truths = []
#         inference_times = []
        
#         # Create progress bar with descriptive name
#         desc = f"Eval {model_name}/{method}" if model_name else "Evaluating"
        
      
#         sample_count = 0
#         skipped_samples = 0
        
#         for images, targets in tqdm(test_dataset.take(num_samples), 
#                                 total=num_samples, 
#                                 desc=desc,
#                                 leave=False):
#             try:
#                 # Prepare input
#                 input_data = images.numpy()
                
#                 # Handle INT8 input if needed
#                 if input_details[0]['dtype'] == np.int8:
#                     input_scale = input_details[0]['quantization'][0]
#                     input_zero_point = input_details[0]['quantization'][1]
#                     input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
                
#                 # Run inference
#                 start_time = time.time()
#                 interpreter.set_tensor(input_details[0]['index'], input_data)
#                 interpreter.invoke()
#                 inference_time = (time.time() - start_time) * 1000  # ms
                
               
#                 count_value = None
                
#                 if len(output_details) == 2:
#                     # Model has density_map and count outputs
#                     count_output = interpreter.get_tensor(output_details[1]['index'])
#                 else:
#                     density_output = interpreter.get_tensor(output_details[0]['index'])
#                     count_output = np.sum(density_output)
                
#                 # Handle different output shapes and types
#                 if isinstance(count_output, np.ndarray):
#                     # Flatten the array first
#                     count_flat = count_output.flatten()
                    
#                     if count_flat.size == 1:
#                         count_value = float(count_flat[0])
#                     elif count_flat.size == 0:
#                         # Empty output
#                         skipped_samples += 1
#                         continue
#                     else:
#                         # Multiple values 
#                         count_value = float(np.sum(count_flat))
#                 elif np.isscalar(count_output):
#                     # Already a scalar
#                     count_value = float(count_output)
#                 else:
#                     # Unknown type 
#                     try:
#                         count_value = float(count_output)
#                     except:
#                         skipped_samples += 1
#                         continue
                
#                 # Get ground truth
#                 gt_value = float(targets['count'].numpy()[0])
                
#                 predictions.append(count_value)
#                 ground_truths.append(gt_value)
#                 inference_times.append(inference_time)
#                 sample_count += 1
                
#             except Exception as e:
#                 # Log the error but continue
#                 if sample_count == 0:  # Only print first error to avoid spam
#                     print(f"    Warning: Error processing sample: {str(e)[:100]}")
#                 skipped_samples += 1
#                 continue
        
#         # Check if to see any valid predictions
#         if len(predictions) == 0:
#             print(f"    WARNING: All {num_samples} samples failed for {model_name}/{method}")
#             return {
#                 'mae': float('inf'),
#                 'mse': float('inf'),
#                 'rmse': float('inf'),
#                 'inference_ms': 0,
#                 'predictions': [],
#                 'ground_truths': []
#             }
        
#         if skipped_samples > 0:
#             print(f"     Skipped {skipped_samples}/{num_samples} samples due to errors")
        
#         # Calculate metrics
#         predictions = np.array(predictions)
#         ground_truths = np.array(ground_truths)
        
#         mae = np.mean(np.abs(predictions - ground_truths))
#         mse = np.mean((predictions - ground_truths) ** 2)
#         rmse = np.sqrt(mse)
#         avg_inference_time = np.mean(inference_times) if inference_times else 0
        
#         return {
#             'mae': mae,
#             'mse': mse,
#             'rmse': rmse,
#             'inference_ms': avg_inference_time,
#             'predictions': predictions.tolist(),
#             'ground_truths': ground_truths.tolist()
#         }
        
#     def run_quantization_pipeline(self, model_paths, part='B'):
#         """Run complete quantization pipeline on multiple models"""
        
#         print("="*60)
#         print("Post-Training Quantization Pipeline")
#         print("="*60)
        
#         # Get calibration and test data
#         print("\nLoading calibration dataset...")
#         calibration_fn, _ = self.get_calibration_dataset(part=part, num_samples=100)
        
#         print("Loading test dataset...")
#         _, _, test_dataset = create_train_val_datasets(part=part, batch_size=1, val_split=0)
        
#         # Quantization methods to test
#         methods = [
#             # 'float32',
#             # 'dynamic_range', 
#             # 'float16',
#             'int8_full'
#             # 'int8_hybrid',
#             # 'int8_fallback'
#         ]
        
#         # Calculate total operations for main progress bar
#         total_operations = len(model_paths) * len(methods)
        
#         # Main progress bar
#         with tqdm(total=total_operations, desc="Overall Progress") as pbar:
#             for model_name, model_path in model_paths.items():
#                 print(f"\n{'='*40}")
#                 print(f"Processing: {model_name}")
#                 print(f"{'='*40}")
                
#                 # Load model
#                 model = self.load_model(model_path)
#                 self.results[model_name] = {}
                
#                 # Progress bar for methods within each model
#                 for method in tqdm(methods, desc=f"{model_name} methods", leave=False):
#                     print(f"\nApplying {method} quantization...")
                    
#                     try:
#                         # Quantize model
#                         print(f"  Converting to {method}...")
#                         tflite_model = self.quantize_model(
#                             model, 
#                             method,
#                             calibration_fn if 'int8' in method else None
#                         )
                        
#                         # Save quantized model
#                         output_path = self.output_dir / f"{model_name}_{method}.tflite"
#                         with open(output_path, 'wb') as f:
#                             f.write(tflite_model)
                        
#                         size_mb = len(tflite_model) / (1024 * 1024)
#                         print(f"  Size: {size_mb:.2f} MB")
                        
#                         # Evaluate model with progress tracking
#                         print(f"  Evaluating {model_name}/{method}...")
#                         metrics = self.evaluate_tflite_model(
#                             tflite_model, 
#                             test_dataset,
#                             num_samples=10,
#                             model_name=model_name,
#                             method=method
#                         )
                        
#                         # Store results
#                         self.results[model_name][method] = {
#                             'size_mb': size_mb,
#                             'mae': metrics['mae'],
#                             'mse': metrics['mse'],
#                             'rmse': metrics['rmse'],
#                             'inference_ms': metrics['inference_ms'],
#                             'path': str(output_path)
#                         }
                        
#                         print(f"  MAE: {metrics['mae']:.2f}")
#                         print(f"  Inference: {metrics['inference_ms']:.2f} ms")
                        
#                     except Exception as e:
#                         print(f"  Failed: {e}")
#                         self.results[model_name][method] = {'error': str(e)}
                    
#                     # Update main progress bar
#                     pbar.update(1)
        
#         # Save results
#         results_file = self.output_dir / "quantization_results.json"
#         with open(results_file, 'w') as f:
#             json.dump(self.results, f, indent=2)
        
#         print(f"\nResults saved to: {results_file}")
        
#         # Generate visualizations
#         print("\nGenerating visualizations...")
#         self.visualize_results()
        
#         return self.results

    
#     def visualize_results(self):
#         """Create comprehensive visualization of quantization results"""
        
#         if not self.results:
#             print("No results to visualize")
#             return
        
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
#         # Prepare data
#         models = list(self.results.keys())
#         methods = ['float32', 'dynamic_range', 'float16', 'int8_full', 'int8_hybrid', 'int8_fallback']
        
#         # 1. Size Comparison
#         ax = axes[0, 0]
#         x = np.arange(len(models))
#         width = 0.13
        
#         for i, method in enumerate(methods):
#             sizes = []
#             for model in models:
#                 if method in self.results[model] and 'size_mb' in self.results[model][method]:
#                     sizes.append(self.results[model][method]['size_mb'])
#                 else:
#                     sizes.append(0)
            
#             if any(sizes):
#                 ax.bar(x + i*width - width*2.5, sizes, width, label=method)
        
#         ax.set_xlabel('Model')
#         ax.set_ylabel('Size (MB)')
#         ax.set_title('Model Size vs Quantization Method')
#         ax.set_xticks(x)
#         ax.set_xticklabels(models, rotation=45)
#         ax.axhline(y=1.0, color='r', linestyle='--', label='1MB Target')
#         ax.legend(loc='upper left', fontsize=8)
#         ax.set_yscale('log')
#         ax.grid(True, alpha=0.3)
        
#         # 2. MAE Comparison
#         ax = axes[0, 1]
        
#         for i, method in enumerate(methods):
#             maes = []
#             for model in models:
#                 if method in self.results[model] and 'mae' in self.results[model][method]:
#                     maes.append(self.results[model][method]['mae'])
#                 else:
#                     maes.append(0)
            
#             if any(maes):
#                 ax.bar(x + i*width - width*2.5, maes, width, label=method)
        
#         ax.set_xlabel('Model')
#         ax.set_ylabel('MAE')
#         ax.set_title('MAE vs Quantization Method')
#         ax.set_xticks(x)
#         ax.set_xticklabels(models, rotation=45)
#         ax.legend(loc='upper left', fontsize=8)
#         ax.grid(True, alpha=0.3)
        
#         # 3. Size vs Accuracy Trade-off
#         ax = axes[1, 0]
        
#         colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
#         markers = ['o', 's', '^', 'D', 'v', 'p']
        
#         for i, model in enumerate(models):
#             sizes = []
#             maes = []
#             labels = []
            
#             for j, method in enumerate(methods):
#                 if method in self.results[model] and 'size_mb' in self.results[model][method]:
#                     sizes.append(self.results[model][method]['size_mb'])
#                     maes.append(self.results[model][method]['mae'])
#                     labels.append(method)
            
#             if sizes and maes:
#                 ax.scatter(sizes, maes, c=[colors[i]]*len(sizes), 
#                           marker=markers[i % len(markers)], s=100, label=model)
                
#                 # Add method labels
#                 for s, m, l in zip(sizes, maes, labels):
#                     if l in ['float32', 'int8_full']: 
#                         ax.annotate(l, (s, m), fontsize=6, 
#                                    xytext=(2, 2), textcoords='offset points')
        
#         ax.set_xlabel('Model Size (MB)')
#         ax.set_ylabel('MAE')
#         ax.set_title('Size-Accuracy Trade-off')
#         ax.set_xscale('log')
#         ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='1MB Target')
#         ax.legend(loc='upper right', fontsize=8)
#         ax.grid(True, alpha=0.3)
        
#         # 4. Inference Time Comparison
#         ax = axes[1, 1]
        
#         bar_width = 0.15
#         x = np.arange(len(methods))
        
#         for i, model in enumerate(models):
#             times = []
#             for method in methods:
#                 if method in self.results[model] and 'inference_ms' in self.results[model][method]:
#                     times.append(self.results[model][method]['inference_ms'])
#                 else:
#                     times.append(0)
            
#             if any(times):
#                 ax.bar(x + i*bar_width - bar_width*(len(models)-1)/2, times, 
#                       bar_width, label=model)
        
#         ax.set_xlabel('Quantization Method')
#         ax.set_ylabel('Inference Time (ms)')
#         ax.set_title('Inference Speed Comparison')
#         ax.set_xticks(x)
#         ax.set_xticklabels(methods, rotation=45)
#         ax.legend(fontsize=8)
#         ax.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         output_path = self.output_dir / 'quantization_analysis.png'
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         plt.show()
        
#         print(f"Visualization saved to: {output_path}")
        
#         # Print summary table
#         self.print_summary_table()
    
#     def print_summary_table(self):
#         """Print formatted summary table of results"""
        
#         print("\n" + "="*100)
#         print("QUANTIZATION RESULTS SUMMARY")
#         print("="*100)
        
#         for model_name, model_results in self.results.items():
#             print(f"\n{model_name}:")
#             print("-"*80)
#             print(f"{'Method':<15} {'Size (MB)':<12} {'MAE':<10} {'MSE':<10} {'Inference (ms)':<15} {'Compression':<12}")
#             print("-"*80)
            
#             baseline_size = model_results.get('float32', {}).get('size_mb', 1)
            
#             for method in ['float32', 'dynamic_range', 'float16', 'int8_full', 'int8_hybrid', 'int8_fallback']:
#                 if method in model_results and 'error' not in model_results[method]:
#                     r = model_results[method]
#                     compression = baseline_size / r['size_mb'] if r.get('size_mb', 0) > 0 else 0
                    
#                     print(f"{method:<15} {r.get('size_mb', 0):<12.2f} {r.get('mae', 0):<10.2f} "
#                           f"{r.get('mse', 0):<10.2f} {r.get('inference_ms', 0):<15.2f} {compression:<12.2f}x")
#                 elif method in model_results:
#                     print(f"{method:<15} {'FAILED':<12} {'-':<10} {'-':<10} {'-':<15} {'-':<12}")
        
#         print("="*100)
        
#         print("\nBEST CONFIGURATIONS:")
#         print("-"*40)
        
#         best_under_1mb = None
#         best_mae = float('inf')
        
#         for model_name, model_results in self.results.items():
#             for method, results in model_results.items():
#                 if 'size_mb' in results and results['size_mb'] < 1.0:
#                     if results.get('mae', float('inf')) < best_mae:
#                         best_mae = results['mae']
#                         best_under_1mb = (model_name, method, results)
        
#         if best_under_1mb:
#             model, method, results = best_under_1mb
#             print(f"Best under 1MB: {model} with {method}")
#             print(f"  Size: {results['size_mb']:.2f} MB")
#             print(f"  MAE: {results['mae']:.2f}")
#             print(f"  Inference: {results['inference_ms']:.2f} ms")
#         else:
#             print("No model achieved <1MB size")


# def main():
#     """Main execution function"""
    
#     # Define models to quantize
#     model_paths = {
#         # 'baseline': r'C:\Users\ss2658\OneDrive - University of Sussex\Desktop\Dissertation\One_SaCNN\saved_models\single_scale_20250824_160038\best_model.h5',
#         # 'pruned_30pct':r'C:\Users\ss2658\OneDrive - University of Sussex\Desktop\Dissertation\One_SaCNN\pruned_models\Exp_256_2_20250823_070552\structured_pruned_30pct.h5',
#         # 'pruned_50pct': r'C:\Users\ss2658\OneDrive - University of Sussex\Desktop\Dissertation\One_SaCNN\pruned_models\Exp_256_2_20250823_070552\structured_pruned_50pct.h5',
#         # 'pruned_70pct': r'C:\Users\ss2658\OneDrive - University of Sussex\Desktop\Dissertation\One_SaCNN\pruned_models\Exp_256_2_20250823_070552\structured_pruned_70pct.h5',
#         'pruned_90pct': r'C:\Users\ss2658\OneDrive - University of Sussex\Desktop\Dissertation\One_SaCNN\saved_models\single_scale_20250824_160038\best_model.h5'
#     }
 
    
#     # Initialize quantizer
#     quantizer = PostTrainingQuantizer()
    
#     # Run quantization pipeline
#     results = quantizer.run_quantization_pipeline(model_paths, part='B')
    
#     print("\nQuantization pipeline complete!")
#     print(f"Results saved in: {quantizer.output_dir}")

# if __name__ == "__main__":
#     main()




"""
Post-Training Quantization Pipeline for Crowd Counting Models

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import sys
from datetime import datetime
from tqdm import tqdm  

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

from config.single_scale_config import CONFIG
from models.single_scale_vgg import SingleScaleSACNN, create_single_scale_model
from models.losses import euclidean_loss, relative_count_loss, mae_count, mse_count, rmse_count
from data.simple_loader import SimpleDataLoader, create_train_val_datasets

class PostTrainingQuantizer:
    """Post-training quantization pipeline """
    
    def __init__(self, experiment_name=None):
        self.experiment_name = experiment_name or f"quantization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path("quantization_results") / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom objects for model loading
        self.custom_objects = {
            'SingleScaleSACNN': SingleScaleSACNN,
            'euclidean_loss': euclidean_loss,
            'relative_count_loss': relative_count_loss,
            'mae_count': mae_count,
            'mse_count': mse_count,
            'rmse_count': rmse_count
        }
        
        self.results = {}
        
    def load_model(self, model_path):
        """Load model """
        try:
            model = tf.keras.models.load_model(
                model_path, 
                custom_objects=self.custom_objects,
                compile=False
            )
            print(f"Loaded model: {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load as saved model: {e}")
            print("Attempting to load weights into architecture...")
            model = create_single_scale_model(input_shape=(256, 256, 3))
            model.load_weights(model_path)
            return model
    

    def get_calibration_dataset(self, part='B', num_samples=100):
        """Get representative calibration data"""
        loader = SimpleDataLoader()
        train_dataset = loader.create_dataset(part, 'train', batch_size=1, shuffle=True)
       
        def representative_dataset():
            """Generator with better coverage"""
            count = 0
            for dataset in [train_dataset]:
                for images, _ in dataset:
                    if count >= num_samples:
                        break
                    # Ensure proper normalization
                    img = images.numpy()
                    # Verify the range is [0, 1]
                    img = np.clip(img, 0.0, 1.0)
                    yield [img.astype(np.float32)]
                    count += 1
                    if count >= num_samples:
                        break
    
        return representative_dataset, train_dataset
    
    def quantize_model(self, model, method, calibration_fn=None):
        """Apply specific quantization method"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if method == 'float32':
            pass
            
        elif method == 'dynamic_range':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif method == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        if method == 'int8_full':
            if calibration_fn is None:
                raise ValueError("INT8 quantization requires calibration data")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = calibration_fn
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
        elif method == 'int8_hybrid':
            if calibration_fn is None:
                raise ValueError("INT8 quantization requires calibration data")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = calibration_fn
            # Float I/O for better accuracy
            
        elif method == 'int8_fallback':
            if calibration_fn is None:
                raise ValueError("INT8 quantization requires calibration data")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = calibration_fn
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS  # Allow TF ops as fallback
            ]
            converter.allow_custom_ops = True
            
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        return converter.convert()
    

    def evaluate_tflite_model(self, tflite_model, test_dataset, num_samples=50, model_name="", method=""):
        """Evaluate quantized model accuracy and speed"""
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        predictions = []
        ground_truths = []
        inference_times = []
        
        # Create progress bar with descriptive name
        desc = f"Eval {model_name}/{method}" if model_name else "Evaluating"
        
      
        sample_count = 0
        skipped_samples = 0
        
        for images, targets in tqdm(test_dataset.take(num_samples), 
                                total=num_samples, 
                                desc=desc,
                                leave=False):
            try:
                # Prepare input
                input_data = images.numpy()
                
                # Handle INT8 input if needed
                if input_details[0]['dtype'] == np.int8:
                    input_scale = input_details[0]['quantization'][0]
                    input_zero_point = input_details[0]['quantization'][1]
                    input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
                
                # Run inference
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                inference_time = (time.time() - start_time) * 1000  # ms
                
               
                count_value = None
                
                if len(output_details) == 2:
                    # Model has density_map and count outputs
                    count_output = interpreter.get_tensor(output_details[1]['index'])
                else:
                    density_output = interpreter.get_tensor(output_details[0]['index'])
                    count_output = np.sum(density_output)
                
                # Handle different output shapes and types
                if isinstance(count_output, np.ndarray):
                    # Flatten the array first
                    count_flat = count_output.flatten()
                    
                    if count_flat.size == 1:
                        count_value = float(count_flat[0])
                    elif count_flat.size == 0:
                        # Empty output
                        skipped_samples += 1
                        continue
                    else:
                        # Multiple values 
                        count_value = float(np.sum(count_flat))
                elif np.isscalar(count_output):
                    # Already a scalar
                    count_value = float(count_output)
                else:
                    # Unknown type 
                    try:
                        count_value = float(count_output)
                    except:
                        skipped_samples += 1
                        continue
                
                # Get ground truth
                gt_value = float(targets['count'].numpy()[0])
                
                predictions.append(count_value)
                ground_truths.append(gt_value)
                inference_times.append(inference_time)
                sample_count += 1
                
            except Exception as e:
                # Log the error but continue
                if sample_count == 0:  # Only print first error to avoid spam
                    print(f"    Warning: Error processing sample: {str(e)[:100]}")
                skipped_samples += 1
                continue
        
        # Check if to see any valid predictions
        if len(predictions) == 0:
            print(f"    WARNING: All {num_samples} samples failed for {model_name}/{method}")
            return {
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'inference_ms': 0,
                'predictions': [],
                'ground_truths': []
            }
        
        if skipped_samples > 0:
            print(f"     Skipped {skipped_samples}/{num_samples} samples due to errors")
        
        # Calculate metrics
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        
        mae = np.mean(np.abs(predictions - ground_truths))
        mse = np.mean((predictions - ground_truths) ** 2)
        rmse = np.sqrt(mse)
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'inference_ms': avg_inference_time,
            'predictions': predictions.tolist(),
            'ground_truths': ground_truths.tolist()
        }
        
    def run_quantization_pipeline(self, model_paths, part='B'):
        """Run complete quantization pipeline on multiple models"""
        
        print("="*60)
        print("Post-Training Quantization Pipeline")
        print("="*60)
        
        # Get calibration and test data
        print("\nLoading calibration dataset...")
        calibration_fn, _ = self.get_calibration_dataset(part=part, num_samples=100)
        
        print("Loading test dataset...")
        _, _, test_dataset = create_train_val_datasets(part=part, batch_size=1, val_split=0)
        
        # Quantization methods to test
        methods = [
            # 'float32',
            # 'dynamic_range', 
            # 'float16',
            'int8_full'
            # 'int8_hybrid',
            # 'int8_fallback'
        ]
        
        # Calculate total operations for main progress bar
        total_operations = len(model_paths) * len(methods)
        
        # Main progress bar
        with tqdm(total=total_operations, desc="Overall Progress") as pbar:
            for model_name, model_path in model_paths.items():
                print(f"\n{'='*40}")
                print(f"Processing: {model_name}")
                print(f"{'='*40}")
                
                # Load model
                model = self.load_model(model_path)
                self.results[model_name] = {}
                
                # Progress bar for methods within each model
                for method in tqdm(methods, desc=f"{model_name} methods", leave=False):
                    print(f"\nApplying {method} quantization...")
                    
                    try:
                        # Quantize model
                        print(f"  Converting to {method}...")
                        tflite_model = self.quantize_model(
                            model, 
                            method,
                            calibration_fn if 'int8' in method else None
                        )
                        
                        # Save quantized model
                        output_path = self.output_dir / f"{model_name}_{method}.tflite"
                        with open(output_path, 'wb') as f:
                            f.write(tflite_model)
                        
                        size_mb = len(tflite_model) / (1024 * 1024)
                        print(f"  Size: {size_mb:.2f} MB")
                        
                        # Evaluate model with progress tracking
                        print(f"  Evaluating {model_name}/{method}...")
                        metrics = self.evaluate_tflite_model(
                            tflite_model, 
                            test_dataset,
                            num_samples=10,
                            model_name=model_name,
                            method=method
                        )
                        
                        # Store results
                        self.results[model_name][method] = {
                            'size_mb': size_mb,
                            'mae': metrics['mae'],
                            'mse': metrics['mse'],
                            'rmse': metrics['rmse'],
                            'inference_ms': metrics['inference_ms'],
                            'path': str(output_path)
                        }
                        
                        print(f"  MAE: {metrics['mae']:.2f}")
                        print(f"  Inference: {metrics['inference_ms']:.2f} ms")
                        
                    except Exception as e:
                        print(f"  Failed: {e}")
                        self.results[model_name][method] = {'error': str(e)}
                    
                    # Update main progress bar
                    pbar.update(1)
        
        # Save results
        results_file = self.output_dir / "quantization_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualize_results()
        
        return self.results

    
    def visualize_results(self):
        """Create comprehensive visualization of quantization results"""
        
        if not self.results:
            print("No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        models = list(self.results.keys())
        methods = ['float32', 'dynamic_range', 'float16', 'int8_full', 'int8_hybrid', 'int8_fallback']
        
        # 1. Size Comparison
        ax = axes[0, 0]
        x = np.arange(len(models))
        width = 0.13
        
        for i, method in enumerate(methods):
            sizes = []
            for model in models:
                if method in self.results[model] and 'size_mb' in self.results[model][method]:
                    sizes.append(self.results[model][method]['size_mb'])
                else:
                    sizes.append(0)
            
            if any(sizes):
                ax.bar(x + i*width - width*2.5, sizes, width, label=method)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Size (MB)')
        ax.set_title('Model Size vs Quantization Method')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.axhline(y=1.0, color='r', linestyle='--', label='1MB Target')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 2. MAE Comparison
        ax = axes[0, 1]
        
        for i, method in enumerate(methods):
            maes = []
            for model in models:
                if method in self.results[model] and 'mae' in self.results[model][method]:
                    maes.append(self.results[model][method]['mae'])
                else:
                    maes.append(0)
            
            if any(maes):
                ax.bar(x + i*width - width*2.5, maes, width, label=method)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('MAE')
        ax.set_title('MAE vs Quantization Method')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. Size vs Accuracy Trade-off
        ax = axes[1, 0]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        markers = ['o', 's', '^', 'D', 'v', 'p']
        
        for i, model in enumerate(models):
            sizes = []
            maes = []
            labels = []
            
            for j, method in enumerate(methods):
                if method in self.results[model] and 'size_mb' in self.results[model][method]:
                    sizes.append(self.results[model][method]['size_mb'])
                    maes.append(self.results[model][method]['mae'])
                    labels.append(method)
            
            if sizes and maes:
                ax.scatter(sizes, maes, c=[colors[i]]*len(sizes), 
                          marker=markers[i % len(markers)], s=100, label=model)
                
                # Add method labels
                for s, m, l in zip(sizes, maes, labels):
                    if l in ['float32', 'int8_full']: 
                        ax.annotate(l, (s, m), fontsize=6, 
                                   xytext=(2, 2), textcoords='offset points')
        
        ax.set_xlabel('Model Size (MB)')
        ax.set_ylabel('MAE')
        ax.set_title('Size-Accuracy Trade-off')
        ax.set_xscale('log')
        ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='1MB Target')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 4. Inference Time Comparison
        ax = axes[1, 1]
        
        bar_width = 0.15
        x = np.arange(len(methods))
        
        for i, model in enumerate(models):
            times = []
            for method in methods:
                if method in self.results[model] and 'inference_ms' in self.results[model][method]:
                    times.append(self.results[model][method]['inference_ms'])
                else:
                    times.append(0)
            
            if any(times):
                ax.bar(x + i*bar_width - bar_width*(len(models)-1)/2, times, 
                      bar_width, label=model)
        
        ax.set_xlabel('Quantization Method')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Speed Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'quantization_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {output_path}")
        
        # Print summary table
        self.print_summary_table()
    
    def print_summary_table(self):
        """Print formatted summary table of results"""
        
        print("\n" + "="*100)
        print("QUANTIZATION RESULTS SUMMARY")
        print("="*100)
        
        for model_name, model_results in self.results.items():
            print(f"\n{model_name}:")
            print("-"*80)
            print(f"{'Method':<15} {'Size (MB)':<12} {'MAE':<10} {'MSE':<10} {'Inference (ms)':<15} {'Compression':<12}")
            print("-"*80)
            
            baseline_size = model_results.get('float32', {}).get('size_mb', 1)
            
            for method in ['float32', 'dynamic_range', 'float16', 'int8_full', 'int8_hybrid', 'int8_fallback']:
                if method in model_results and 'error' not in model_results[method]:
                    r = model_results[method]
                    compression = baseline_size / r['size_mb'] if r.get('size_mb', 0) > 0 else 0
                    
                    print(f"{method:<15} {r.get('size_mb', 0):<12.2f} {r.get('mae', 0):<10.2f} "
                          f"{r.get('mse', 0):<10.2f} {r.get('inference_ms', 0):<15.2f} {compression:<12.2f}x")
                elif method in model_results:
                    print(f"{method:<15} {'FAILED':<12} {'-':<10} {'-':<10} {'-':<15} {'-':<12}")
        
        print("="*100)
        
        print("\nBEST CONFIGURATIONS:")
        print("-"*40)
        
        best_under_1mb = None
        best_mae = float('inf')
        
        for model_name, model_results in self.results.items():
            for method, results in model_results.items():
                if 'size_mb' in results and results['size_mb'] < 1.0:
                    if results.get('mae', float('inf')) < best_mae:
                        best_mae = results['mae']
                        best_under_1mb = (model_name, method, results)
        
        if best_under_1mb:
            model, method, results = best_under_1mb
            print(f"Best under 1MB: {model} with {method}")
            print(f"  Size: {results['size_mb']:.2f} MB")
            print(f"  MAE: {results['mae']:.2f}")
            print(f"  Inference: {results['inference_ms']:.2f} ms")
        else:
            print("No model achieved <1MB size")


def main():
    """Main execution function"""
    
    # Define models to quantize
    SAVED_MODELS_DIR = Path("./saved_models/single_scale_20250824_160038")
    PRUNED_MODELS_DIR = Path("./pruned_models/Exp_256_2_20250823_070552")

    model_paths = {
        'baseline':     str(SAVED_MODELS_DIR / "best_model.h5"),
        'pruned_30pct': str(PRUNED_MODELS_DIR / "structured_pruned_30pct.h5"),
        'pruned_50pct': str(PRUNED_MODELS_DIR / "structured_pruned_50pct.h5"),
        'pruned_70pct': str(PRUNED_MODELS_DIR / "structured_pruned_70pct.h5"),
        'pruned_90pct': str(PRUNED_MODELS_DIR / "structured_pruned_90pct.h5"),
    }
 
    
    # Initialize quantizer
    quantizer = PostTrainingQuantizer()
    
    # Run quantization pipeline
    results = quantizer.run_quantization_pipeline(model_paths, part='B')
    
    print("\nQuantization pipeline complete!")
    print(f"Results saved in: {quantizer.output_dir}")

if __name__ == "__main__":
    main()