"""
Evaluation Script
"""

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class QuantizedModelEvaluator:
    def __init__(self, models_dir, test_data_file):
        """
        Initialize evaluator with pre-computed test data
        
        Args:
            models_dir: Directory containing .tflite files
            test_data_file: Path to .npz file with test data
        """
        self.models_dir = Path(models_dir)
        self.test_data_file = Path(test_data_file)
        self.results_dir = self.models_dir / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load test data from npz file
        self.load_test_data()
        
    def load_test_data(self):
        """Load pre-computed test data from .npz file"""
        print(f"\nLoading test data from: {self.test_data_file}")
        
        if not self.test_data_file.exists():
            raise FileNotFoundError(f"Test data file not found: {self.test_data_file}")
        
        # Load the npz file
        data = np.load(self.test_data_file)
        self.test_images = data['images']
        self.ground_truth_counts = data['counts']
        
        # Reshape if needed 
        if len(self.test_images.shape) == 5:  
            self.test_images = self.test_images.squeeze(1)
        if len(self.ground_truth_counts.shape) == 3: 
            self.ground_truth_counts = self.ground_truth_counts.squeeze()
        
        print(f"Loaded {len(self.test_images)} test samples")
        print(f"Image shape: {self.test_images[0].shape}")
        print(f"Count range: {self.ground_truth_counts.min():.0f} - {self.ground_truth_counts.max():.0f}")
        print(f"Average count: {self.ground_truth_counts.mean():.1f}")
        
    def find_tflite_models(self):
        """Find all .tflite files in the models directory AND subdirectories"""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")
        
        # Look in subdirectories recursively
        tflite_files = list(self.models_dir.rglob("*.tflite"))
        
        # Also check for alternative naming patterns
        # Check for models with different percentage formats
        print(f"\nSearching for models in: {self.models_dir}")
        print(f"Looking for patterns including: 30pct, 30_pct")
        
        if not tflite_files:
            raise FileNotFoundError(f"No .tflite files found in {self.models_dir}")
        
        print(f"\nFound {len(tflite_files)} .tflite models:")
        
        # Group by directory for better display
        by_directory = {}
        for f in sorted(tflite_files):
            dir_name = f.parent.name
            if dir_name not in by_directory:
                by_directory[dir_name] = []
            by_directory[dir_name].append(f)
        
        # Display all found models clearly
        for dir_name, files in sorted(by_directory.items()):
            print(f"\n{dir_name}:")
            for f in sorted(files):
                size_mb = f.stat().st_size / (1024 * 1024)
                # Check if it's a 30% model
                if any(x in f.name.lower() for x in ['30pct', '30_pct']):
                    print(f"  {f.name} ({size_mb:.2f} MB) - 30% PRUNED MODEL FOUND")
                else:
                    print(f"  {f.name} ({size_mb:.2f} MB)")
        
        return sorted(tflite_files)
    
    def evaluate_single_model(self, model_path):
        """Evaluate a single TFLite model"""
        model_name = Path(model_path).name
        parent_dir = Path(model_path).parent.name
        full_model_name = f"{parent_dir}/{model_name}"
        
        print(f"Evaluating: {full_model_name}")
        
        # Load interpreter
        try:
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_detail = input_details[0]
        output_detail = output_details[0]
        
        print(f"Input shape: {input_detail['shape']}, dtype: {input_detail['dtype']}")
        print(f"Output shape: {output_detail['shape']}, dtype: {output_detail['dtype']}")
        
        # Run inference on all test samples
        predictions = []
        inference_times = []
        errors = []
        
        for i in tqdm(range(len(self.test_images)), desc="Running inference"):
            input_data = self.test_images[i:i+1].astype(np.float32)
            

            if input_detail['dtype'] == np.int8:
                if input_detail.get('quantization') and len(input_detail['quantization']) >= 2:
                    scale, zero_point = input_detail['quantization'][:2]
                    if scale != 0:
                        input_data = input_data / scale + zero_point
                        input_data = np.clip(input_data, -128, 127).astype(np.int8)
                else:
                    input_data = (input_data * 127).astype(np.int8)
            elif input_detail['dtype'] == np.uint8:

                if input_detail.get('quantization') and len(input_detail['quantization']) >= 2:
                    scale, zero_point = input_detail['quantization'][:2]
                    if scale != 0:
                        input_data = input_data / scale + zero_point
                        input_data = np.clip(input_data, 0, 255).astype(np.uint8)
                else:
                    # Default UINT8 quantization (0-255 range)
                    input_data = np.clip(input_data * 255, 0, 255).astype(np.uint8)
            
            # Set input tensor
            interpreter.set_tensor(input_detail['index'], input_data)
            
            # Run inference
            start_time = time.time()
            interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get output
            output_data = interpreter.get_tensor(output_detail['index'])
            
            if output_detail['dtype'] == np.int8:
                if output_detail.get('quantization') and len(output_detail['quantization']) >= 2:
                    scale, zero_point = output_detail['quantization'][:2]
                    if scale != 0:
                        output_data = (output_data.astype(np.float32) - zero_point) * scale
                else:
                    output_data = output_data.astype(np.float32)
            
            # Calculate predicted count
            if len(output_data.shape) == 4:  # Density map output
                pred_count = float(np.sum(output_data))
            else:  # Direct count output
                pred_count = float(output_data.flatten()[0])
            
            true_count = float(self.ground_truth_counts[i])
            error = pred_count - true_count
            
            predictions.append(pred_count)
            inference_times.append(inference_time)
            errors.append(error)
        
        # Convert to arrays
        predictions = np.array(predictions)
        errors = np.array(errors)
        
        # Calculate metrics
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        
        # Additional metrics
        avg_inference_time = np.mean(inference_times)
        model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        
        print(f"\nResults for {full_model_name}:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  MSE:  {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  Mean inference time: {avg_inference_time:.2f} ms")
        print(f"  Model size: {model_size_mb:.2f} MB")
        
        return {
            'model_name': full_model_name,
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'inference_time_ms': float(avg_inference_time),
            'model_size_mb': float(model_size_mb),
            'predictions': predictions.tolist(),
            'errors': errors.tolist()
        }
    
    def evaluate_all_models(self):
        """Evaluate all TFLite models"""
        # Find all models
        model_files = self.find_tflite_models()
        
        # Evaluate each model
        all_results = {}
        for model_path in model_files:
            result = self.evaluate_single_model(model_path)
            if result:
                all_results[result['model_name']] = result
        
        # Save results
        self.save_results(all_results)
        
        # Generate visualizations
        self.generate_clean_visualizations(all_results)
        
        return all_results
    
    def save_results(self, results):
        """Save evaluation results to JSON"""
        summary = {}
        for name, data in results.items():
            summary[name] = {
                'mae': data['mae'],
                'mse': data['mse'],
                'rmse': data['rmse'],
                'inference_time_ms': data['inference_time_ms'],
                'model_size_mb': data['model_size_mb']
            }
        
        output_file = self.results_dir / 'quantization_evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def generate_clean_visualizations(self, results):
        """Generate cleaner, more focused visualizations"""
        print("\nGenerating clean visualizations...")
        
        # Separate INT8 results for separate analysis
        int8_results = {k: v for k, v in results.items() if 'int8' in k.lower()}
        working_results = {k: v for k, v in results.items() if 'int8' not in k.lower()}
        
        # Organize by pruning level and quantization type
        organized_data = self.organize_results(working_results)
        
      
        self.create_main_comparison_plots(organized_data, int8_results)
        self.create_detailed_analysis_plots(working_results)
        
    def organize_results(self, results):
        """Organize results by pruning level and quantization type"""
        organized = {
            'baseline': {},
            'pruned_30pct': {},  
            'pruned_50pct': {},
            'pruned_70pct': {},
            'pruned_90pct': {}
        }
        
        for name, data in results.items():
            # Determine pruning level 
            name_lower = name.lower()
            if 'pruned_90pct' in name_lower or 'pruned_90_pct' in name_lower:
                pruning_level = 'pruned_90pct'
            elif 'pruned_70pct' in name_lower or 'pruned_70_pct' in name_lower:
                pruning_level = 'pruned_70pct'
            elif 'pruned_50pct' in name_lower or 'pruned_50_pct' in name_lower:
                pruning_level = 'pruned_50pct'
            elif 'pruned_30pct' in name_lower or 'pruned_30_pct' in name_lower:
                pruning_level = 'pruned_30pct'
            else:
                pruning_level = 'baseline'
            
            # Determine quantization type
            if 'float16' in name_lower:
                quant_type = 'float16'
            elif 'dynamic' in name_lower:
                quant_type = 'dynamic_range'
            else:
                quant_type = 'baseline'
            
            organized[pruning_level][quant_type] = data
        
        return organized
    
   
    def create_main_comparison_plots(self, organized_data, int8_results):
        """Create main comparison plots with INT8 included"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: MAE comparison with INT8 and clearer values
        ax = axes[0, 0]
        pruning_labels = ['Baseline', '30% Pruned', '50% Pruned', '70% Pruned', '90% Pruned']
        pruning_keys = ['baseline', 'pruned_30pct', 'pruned_50pct', 'pruned_70pct', 'pruned_90pct']
        
        # Include INT8_full
        quant_methods = ['baseline', 'float16', 'dynamic_range', 'int8_full']
        quant_labels = ['Float32', 'Float16', 'Dynamic', 'INT8 Full']
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        x = np.arange(len(pruning_labels))
        width = 0.2  # Narrower bars for 4 types
        
        for i, (method, label, color) in enumerate(zip(quant_methods, quant_labels, colors)):
            mae_values = []
            for key in pruning_keys:
                if method == 'int8_full':
                    # Look for INT8 full models
                    found = False
                    for model_name, model_data in int8_results.items():
                        if key in model_name.lower() and 'int8_full' in model_name.lower():
                            mae_values.append(model_data['mae'])
                            found = True
                            break
                    if not found:
                        mae_values.append(0)
                elif key in organized_data and method in organized_data[key]:
                    mae_values.append(organized_data[key][method]['mae'])
                else:
                    mae_values.append(0)
            
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, mae_values, width, label=label, color=color)
            
            # Add clear value labels
            for bar, val in zip(bars, mae_values):
                if val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{val:.1f}', ha='center', va='bottom', 
                        fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Pruning Level', fontsize=11)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title('Mean Absolute Error: Pruning × Quantization (All Types)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pruning_labels)
        ax.axhline(y=38.59, color='red', linestyle='--', alpha=0.5, label='Baseline MAE')
        ax.axhline(y=33.2, color='green', linestyle='--', alpha=0.5, label='Target MAE')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 160)  # Extended for INT8 values
        
        # Plot 2: Size heatmap WITH INT8
        ax = axes[0, 1]
        size_matrix = np.zeros((5, 4))  # 5 pruning levels, 4 quant methods
        
        for i, key in enumerate(pruning_keys):
            # Float32, Float16, Dynamic
            for j, method in enumerate(['baseline', 'float16', 'dynamic_range']):
                if key in organized_data and method in organized_data[key]:
                    size_matrix[i, j] = organized_data[key][method]['model_size_mb']
            
            # INT8 Full
            for model_name, model_data in int8_results.items():
                if key in model_name.lower() and 'int8_full' in model_name.lower():
                    size_matrix[i, 3] = model_data['model_size_mb']
                    break
        
        im = ax.imshow(size_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=35)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Float32', 'Float16', 'Dynamic', 'INT8 Full'])
        ax.set_yticks(range(5))
        ax.set_yticklabels(pruning_labels)
        ax.set_title('Model Size Heatmap (MB) - Including INT8', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(5):
            for j in range(4):
                if size_matrix[i, j] > 0:
                    ax.text(j, i, f'{size_matrix[i, j]:.1f}',
                        ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax)
        
        # Plot 3: Size-Accuracy Pareto Frontier
        ax = axes[1, 0]
        
        for pruning_level, quant_data in organized_data.items():
            for quant_type, data in quant_data.items():
                if pruning_level == 'baseline':
                    color = 'blue'
                    marker = 'o'
                elif pruning_level == 'pruned_30pct':
                    color = 'purple'
                    marker = 'p'
                elif pruning_level == 'pruned_50pct':
                    color = 'green'
                    marker = 's'
                elif pruning_level == 'pruned_70pct':
                    color = 'orange'
                    marker = '^'
                else:
                    color = 'red'
                    marker = 'd'
                
                ax.scatter(data['model_size_mb'], data['mae'], 
                        s=100, color=color, marker=marker, alpha=0.7,
                        edgecolors='black', linewidth=1)
        
        ax.set_xlabel('Model Size (MB)', fontsize=11)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title('Size-Accuracy Trade-off (Pareto Frontier)', fontsize=12, fontweight='bold')
        ax.axhline(y=33.2, color='green', linestyle='--', alpha=0.5, label='Target MAE')
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='1MB Target')
        ax.set_xlim(-1, 40)
        ax.set_ylim(20, 50)
        ax.grid(True, alpha=0.3)
        
        legend_elements = [
            mpatches.Patch(color='blue', label='Baseline'),
            mpatches.Patch(color='purple', label='30% Pruned'),
            mpatches.Patch(color='green', label='50% Pruned'),
            mpatches.Patch(color='orange', label='70% Pruned'),
            mpatches.Patch(color='red', label='90% Pruned')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Plot 4: Best Models Table 
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        best_models = []
        for name, data in sorted(organized_data.items()):
            for quant, qdata in data.items():
                if qdata['model_size_mb'] < 10:
                    best_models.append((f"{name}/{quant}", qdata))
        
        best_models = sorted(best_models, key=lambda x: x[1]['mae'])[:10]
        
        table_data = [['Model Configuration', 'MAE', 'Size (MB)', 'Compression']]
        for name, data in best_models:
            short_name = name.replace('structured_pruned_', '').replace('pruned_', '').replace('baseline/', '')
            compression = 34.0 / data['model_size_mb']
            table_data.append([
                short_name, 
                f"{data['mae']:.1f}", 
                f"{data['model_size_mb']:.2f}",
                f"{compression:.1f}×"
            ])
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        for i in range(4):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Top Compressed Models Performance', fontsize=12, fontweight='bold')
        
        plt.suptitle('Complete Quantization + Pruning Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'complete_quantization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create INT8 full analysis
        if int8_results:
            self.create_int8_failure_analysis(int8_results)

        
    def create_int8_failure_analysis(self, int8_results):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter for INT8_full models ONLY
        int8_full_results = {}
        for model_name, model_data in int8_results.items():
            if 'int8_full' in model_name.lower():
                int8_full_results[model_name] = model_data
        
        if not int8_full_results:
            print("No INT8 Full models found")
            return
        
        # Organize by pruning level
        pruning_data = {
            'Baseline': None,
            '30% Pruned': None,
            '50% Pruned': None,
            '70% Pruned': None,
            '90% Pruned': None
        }
        
        for model_name, model_data in int8_full_results.items():
            model_lower = model_name.lower()
            if 'baseline' in model_lower and not any(x in model_lower for x in ['30pct', '50pct', '70pct', '90pct']):
                pruning_data['Baseline'] = model_data
            elif '30pct' in model_lower or '30_pct' in model_lower:
                pruning_data['30% Pruned'] = model_data
            elif '50pct' in model_lower or '50_pct' in model_lower:
                pruning_data['50% Pruned'] = model_data
            elif '70pct' in model_lower or '70_pct' in model_lower:
                pruning_data['70% Pruned'] = model_data
            elif '90pct' in model_lower or '90_pct' in model_lower:
                pruning_data['90% Pruned'] = model_data
        
        # Plot 1: INT8 Full MAE comparison
        ax = axes[0]
        labels = []
        mae_values = []
        
        for level, data in pruning_data.items():
            if data is not None:
                labels.append(level)
                mae_values.append(data['mae'])
        
        if mae_values:
            bars = ax.bar(range(len(labels)), mae_values, color='#e74c3c', alpha=0.8, edgecolor='darkred', linewidth=2)
            
            # Add value labels above bars
            for bar, val in zip(bars, mae_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val:.1f}', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Pruning Level', fontsize=11)
            ax.set_ylabel('MAE', fontsize=11)
            ax.set_title('INT8 Full Quantization Only', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=10)
            ax.axhline(y=38.59, color='blue', linestyle='--', alpha=0.5, label='Baseline MAE')
            ax.axhline(y=33.2, color='green', linestyle='--', alpha=0.5, label='Target MAE')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(mae_values) * 1.15 if mae_values else 150)
        
        # Plot 2: Analysis
        ax = axes[1]
        ax.axis('off')
        
        text = "INT8 Full Quantization Analysis:\n\n"
        text += "Results by Pruning Level:\n"
        
        for level, data in pruning_data.items():
            if data is not None:
                text += f"  • {level}: MAE = {data['mae']:.1f}\n"
        
        text += "\n\nKey Findings:\n"
        text += "• INT8 Full shows severe accuracy degradation\n"
        text += "• Performance does not improve with pruning\n"
        text += "• Requires quantization-aware training\n\n"
        text += "Recommendation:\n"
        text += "Use Float16 or Dynamic Range for deployment"
        
        ax.text(0.05, 0.5, text, fontsize=10, verticalalignment='center', fontfamily='monospace')
        
        plt.suptitle('INT8 Full Quantization Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'int8_full_only_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_detailed_analysis_plots(self, results):
        """Create detailed analysis INCLUDING INT8 models"""
        # Include ALL results
        all_results = results.copy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # First plot: Working models 
        working_results = {k: v for k, v in all_results.items() if 'int8' not in k.lower()}
        sorted_working = sorted(working_results.items(), key=lambda x: x[1]['mae'])
        
        names_working = []
        maes_working = []
        sizes_working = []
        
        for name, data in sorted_working:
            short_name = self._get_short_name(name)
            names_working.append(short_name)
            maes_working.append(data['mae'])
            sizes_working.append(data['model_size_mb'])
        
        bars = ax1.bar(range(len(names_working)), maes_working, color='skyblue', edgecolor='navy')
        
        if sizes_working:
            norm = plt.Normalize(min(sizes_working), max(sizes_working))
            sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
            sm.set_array([])
            
            for bar, size in zip(bars, sizes_working):
                bar.set_facecolor(sm.to_rgba(size))
            
            cbar = plt.colorbar(sm, ax=ax1)
            cbar.set_label('Model Size (MB)', rotation=270, labelpad=15)
        
        ax1.set_xlabel('Model Configuration', fontsize=11)
        ax1.set_ylabel('MAE', fontsize=11)
        ax1.set_title('Working Models Performance (Float32, Float16, Dynamic Range)', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(names_working)))
        ax1.set_xticklabels(names_working, rotation=45, ha='right', fontsize=9)
        ax1.axhline(y=38.59, color='red', linestyle='--', alpha=0.5, label='Baseline MAE')
        ax1.axhline(y=33.2, color='green', linestyle='--', alpha=0.5, label='Target MAE')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()
        
        # Second plot
        all_sorted = sorted(all_results.items(), key=lambda x: x[1]['mae'])
        
        names_all = []
        maes_all = []
        colors_all = []
        
        for name, data in all_sorted:
            short_name = self._get_short_name(name)
            names_all.append(short_name)
            maes_all.append(data['mae'])
            
            name_lower = name.lower()
            if 'int8_full' in name_lower:
                colors_all.append('#e74c3c')
            elif 'int8_hybrid' in name_lower:
                colors_all.append('#c0392b')
            elif 'int8_fallback' in name_lower:
                colors_all.append('#a93226')
            elif 'int8' in name_lower:
                colors_all.append('#922b21')
            elif 'float16' in name_lower:
                colors_all.append('#2ecc71')
            elif 'dynamic' in name_lower:
                colors_all.append('#f39c12')
            else:
                colors_all.append('#3498db')
        
        bars2 = ax2.bar(range(len(names_all)), maes_all, color=colors_all, alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Model Configuration', fontsize=11)
        ax2.set_ylabel('MAE', fontsize=11)
        ax2.set_title('All Models Comparison ', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(names_all)))
        ax2.set_xticklabels(names_all, rotation=45, ha='right', fontsize=8)
        ax2.axhline(y=38.59, color='red', linestyle='--', alpha=0.5, label='Baseline MAE')
        ax2.axhline(y=33.2, color='green', linestyle='--', alpha=0.5, label='Target MAE')
        ax2.grid(True, alpha=0.3, axis='y')
        
        legend_elements = [
            Patch(facecolor='#3498db', label='Float32', alpha=0.7),
            Patch(facecolor='#2ecc71', label='Float16', alpha=0.7),
            Patch(facecolor='#f39c12', label='Dynamic Range', alpha=0.7),
            Patch(facecolor='#e74c3c', label='INT8 Full', alpha=0.7),
            Patch(facecolor='#c0392b', label='INT8 Hybrid', alpha=0.7),
            Patch(facecolor='#a93226', label='INT8 Fallback', alpha=0.7)
        ]
        ax2.legend(handles=legend_elements, loc='upper left', ncol=2)
        
        # Adjust y-axis to show INT8 values
        max_mae = max(maes_all) if maes_all else 150
        ax2.set_ylim(0, max_mae * 1.1)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'all_models_complete_with_int8.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _get_short_name(self, name):
        """Helper function to create short, clear model names"""
        name_lower = name.lower()
        
        # Determine pruning level
        if 'baseline' in name_lower or (not any(x in name_lower for x in ['30pct', '50pct', '70pct', '90pct'])):
            pruning = 'Base'
        elif '30pct' in name_lower or '30_pct' in name_lower:
            pruning = '30%'
        elif '50pct' in name_lower or '50_pct' in name_lower:
            pruning = '50%'
        elif '70pct' in name_lower or '70_pct' in name_lower:
            pruning = '70%'
        elif '90pct' in name_lower or '90_pct' in name_lower:
            pruning = '90%'
        else:
            pruning = 'Unk'
        
        # Determine quantization type
        if 'int8' in name_lower:
            quant = 'INT8'
        elif 'float16' in name_lower:
            quant = 'F16'
        elif 'dynamic' in name_lower:
            quant = 'Dyn'
        else:
            quant = 'F32'
        
        return f'{pruning}-{quant}'
    
    def print_summary(self, results):
        """Print summary table"""
        print("\n" + "="*80)
        print("QUANTIZATION EVALUATION SUMMARY")
        print("="*80)
        
        # Check for 30% pruned models specifically
        pruned_30_models = [(k, v) for k, v in results.items() 
                            if any(x in k.lower() for x in ['30pct', '30_pct'])]
        
        # Separate INT8 and working models
        int8_models = [(k, v) for k, v in results.items() if 'int8' in k.lower()]
        working_models = [(k, v) for k, v in results.items() if 'int8' not in k.lower()]
        
        # Sort by MAE
        working_models = sorted(working_models, key=lambda x: x[1]['mae'])
        
        print("\nWorking Models (Float32, Float16, Dynamic Range):")
        print(f"{'Model':<55} {'MAE':<10} {'Size (MB)':<12} {'Compression':<10}")
        print("-"*90)
        
        for model_name, data in working_models:
            short_name = model_name.replace('structured_pruned_', '').replace('model_', '')
            compression = 34.0 / data['model_size_mb']
            
            # Highlight 30% pruned models
            if any(x in model_name.lower() for x in ['30pct', '30_pct']):
                print(f"→ {short_name:<53} {data['mae']:<10.2f} {data['model_size_mb']:<12.2f} {compression:<10.1f}×")
            else:
                print(f"{short_name:<55} {data['mae']:<10.2f} {data['model_size_mb']:<12.2f} {compression:<10.1f}×")
        
        if pruned_30_models:
            print(f"\nFound {len(pruned_30_models)} models with 30% pruning")
        else:
            print("\n⚠ No 30% pruned models found - check naming convention")
        
        if int8_models:
            print("\nINT8 Models (Failed):")
            print(f"{'Model':<55} {'MAE':<10}")
            print("-"*65)
            for model_name, data in int8_models:
                short_name = model_name.replace('structured_pruned_', '').replace('model_', '')
                print(f"{short_name:<55} {data['mae']:<10.2f} (FAILED)")
        
        # Best model
        best_model = working_models[0]
        print(f"\nBest Model: {best_model[0]}")
        print(f"  MAE: {best_model[1]['mae']:.2f}")
        print(f"  Size: {best_model[1]['model_size_mb']:.2f} MB")
        print(f"  Compression: {34.0/best_model[1]['model_size_mb']:.1f}×")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate quantized TFLite models with enhanced visualization")
    parser.add_argument('--models_dir', type=str, required=True,
                       help='Directory containing .tflite models')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to .npz file with test data')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = QuantizedModelEvaluator(args.models_dir, args.test_data)
    
    # Run evaluation
    results = evaluator.evaluate_all_models()
    
    # Print summary
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()