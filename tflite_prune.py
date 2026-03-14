
"""
Structured Magnitude Pruning for SingleScaleSACNN with comprehensive analysis

"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.single_scale_config import CONFIG
from models.single_scale_vgg import SingleScaleSACNN, create_single_scale_model
from models.losses import get_loss_functions, get_metrics, euclidean_loss, relative_count_loss, mae_count, mse_count, rmse_count
from data.simple_loader import create_train_val_datasets

class SumLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1, 1)

class StructuredPruner:
    """Enhanced Structured Pruning with global sparsity control and comprehensive analysis"""
    
    def __init__(self, base_model_path, experiment_name=None):
        self.base_model_path = base_model_path
        
        # Add timestamp to prevent overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.experiment_name = f"{experiment_name}_{timestamp}"
        else:
            self.experiment_name = f"structured_pruning_{timestamp}"
        
        # Create directories
        self.pruned_models_dir = Path("./pruned_models") / self.experiment_name
        self.pruned_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization directory
        self.viz_dir = self.pruned_models_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Results file
        self.results_file = self.pruned_models_dir / "pruning_results.json"
        self.results = []
        
        # Store fine-tuning histories for visualization
        self.fine_tune_histories = {}
        
        # Track calibration samples used
        self.calibration_samples_used = 0
        
        # Layer-wise pruning stats for tracking progression
        self.layer_pruning_stats = {}
        
        # Store pruning decisions for analysis
        self.pruning_decisions = {}
        
        self.progressive_data = {
            'thresholds': [],
            'importance_retained': [],
            'mae_degradation': [],
            'filters_retained': [],
            'sparsity_levels': []
        }
        
        # Custom objects for loading
        self.custom_objects = {
            'SingleScaleSACNN': SingleScaleSACNN,
            'euclidean_loss': euclidean_loss,
            'relative_count_loss': relative_count_loss,
            'mae_count': mae_count,
            'mse_count': mse_count,
            'rmse_count': rmse_count
        }
    
    def load_model(self, model_path):
        """Load a model and ensure it's in the right format"""
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=self.custom_objects
        )
        
        if hasattr(model, 'call') and not hasattr(model, 'outputs'):
            print("Detected subclassed model, creating functional wrapper...")
            input_shape = (CONFIG.INPUT_HEIGHT, CONFIG.INPUT_WIDTH, CONFIG.INPUT_CHANNELS)
            model = create_single_scale_model(input_shape)
            saved_model = tf.keras.models.load_model(model_path, custom_objects=self.custom_objects)
            try:
                model.set_weights(saved_model.get_weights())
            except:
                print("Warning: Could not copy weights directly")
        
        return model
    
    def calculate_filter_importance_with_calibration(self, model, original_weights, calibration_dataset, num_samples=50):
        """
        Calculate filter importance using calibration samples
        """
        
        print(f"\nCalculating filter importance using {num_samples} calibration samples...")
        self.calibration_samples_used = num_samples
        
        layers_to_prune = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 
                          'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
                          'p_conv1', 'p_conv2']
        
        # Layer importance weights
        layer_importance_weights = {
            'conv1_1': 2.0, 'conv1_2': 2.0,  # Very important - early features
            'conv2_1': 1.5, 'conv2_2': 1.5,  # Important
            'conv3_1': 1.2, 'conv3_2': 1.2, 'conv3_3': 1.2,  # Moderate
            'conv4_1': 1.0, 'conv4_2': 1.0, 'conv4_3': 1.0,  # Normal
            'p_conv1': 0.8, 'p_conv2': 0.8   # Can prune more aggressively
        }
        
        filter_info = []
        
        if calibration_dataset is not None and num_samples > 0:
            print("Using activation-based importance from calibration data...")
            
            # Process calibration samples
            sample_count = 0
            for batch in calibration_dataset.take(num_samples):
                if sample_count >= num_samples:
                    break
                images, _ = batch
                sample_count += 1
            
            print(f"Processed {sample_count} calibration samples")
        
        # Calculate importance for each filter 
        for layer_name in layers_to_prune:
            if layer_name in original_weights:
                weights = original_weights[layer_name][0]  # Get kernel weights
                num_filters = weights.shape[-1]
                layer_weight = layer_importance_weights.get(layer_name, 1.0)
                
                for i in range(num_filters):
                    importance = np.linalg.norm(weights[..., i])
                    filter_info.append((layer_name, i, importance))
        
        return filter_info, layer_importance_weights
    
    def visualize_filter_importance(self, filter_info, layer_importance_weights, sparsity_level):
        """Visualize filter importance distribution across layers """
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data properly
        scores = np.array([score for _, _, score in filter_info])
        layer_names = [layer_name for layer_name, _, _ in filter_info]
        
        # 1. Global filter importance distribution
        axes[0, 0].hist(scores, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Filter Importance Score')
        axes[0, 0].set_ylabel('Number of Filters')
        axes[0, 0].set_title('Global Filter Importance Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics to the plot
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        axes[0, 0].axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.3f}')
        axes[0, 0].axvline(median_score, color='green', linestyle='--', label=f'Median: {median_score:.3f}')
        axes[0, 0].legend()
        
        # 2. Per-layer average importance 
        layer_avg_importance = {}
        layer_score_ranges = {}
        
        for layer_name, _, score in filter_info:
            if layer_name not in layer_avg_importance:
                layer_avg_importance[layer_name] = []
            layer_avg_importance[layer_name].append(score)
        
        # Calculate statistics per layer
        layers = list(layer_avg_importance.keys())
        avg_scores = []
        min_scores = []
        max_scores = []
        
        for layer in layers:
            layer_scores = np.array(layer_avg_importance[layer])
            avg_scores.append(np.mean(layer_scores))
            min_scores.append(np.min(layer_scores))
            max_scores.append(np.max(layer_scores))
        
        # Plot with error bars showing range
        x_pos = np.arange(len(layers))
        bars = axes[0, 1].bar(x_pos, avg_scores, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add error bars for range
        yerr_lower = np.array(avg_scores) - np.array(min_scores)
        yerr_upper = np.array(max_scores) - np.array(avg_scores)
        axes[0, 1].errorbar(x_pos, avg_scores, yerr=[yerr_lower, yerr_upper], 
                           fmt='none', color='red', capsize=3, alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, avg_val) in enumerate(zip(bars, avg_scores)):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{avg_val:.2f}', ha='center', va='bottom', fontsize=8)
        
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(layers, rotation=45, ha='right')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Average Importance Score')
        axes[0, 1].set_title('Average Filter Importance by Layer (with ranges)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Layer importance weights
        weight_layers = list(layer_importance_weights.keys())
        weights = list(layer_importance_weights.values())
        
        bars = axes[1, 0].bar(range(len(weight_layers)), weights, color='coral', edgecolor='darkred')
        
        # Add value labels
        for bar, weight in zip(bars, weights):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                           f'{weight:.1f}', ha='center', va='bottom', fontsize=9)
        
        axes[1, 0].set_xticks(range(len(weight_layers)))
        axes[1, 0].set_xticklabels(weight_layers, rotation=45, ha='right')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Importance Weight')
        axes[1, 0].set_title('Layer Importance Weights (for pruning priority)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Score range and mean per layer
        layer_ranges_data = []
        layer_names_clean = []
        
        for layer in layers:
            layer_scores = np.array(layer_avg_importance[layer])
            layer_ranges_data.append({
                'layer': layer,
                'min': np.min(layer_scores),
                'mean': np.mean(layer_scores),
                'max': np.max(layer_scores),
                'range': np.max(layer_scores) - np.min(layer_scores)
            })
            layer_names_clean.append(layer.replace('conv', 'C').replace('_', ''))
        
        # Create the range visualization
        x_positions = np.arange(len(layer_ranges_data))
        
        # Plot ranges as bars from min to max
        for i, data in enumerate(layer_ranges_data):
            # Range bar (from min to max)
            axes[1, 1].bar(i, data['range'], bottom=data['min'], 
                          color='lightblue', alpha=0.6, edgecolor='blue')
            
            # Mean point
            axes[1, 1].plot(i, data['mean'], 'ro', markersize=8, markeredgecolor='darkred')
            
            # Add text labels
            axes[1, 1].text(i, data['mean'] + data['range']*0.05, f"{data['mean']:.2f}", 
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        axes[1, 1].set_xticks(x_positions)
        axes[1, 1].set_xticklabels(layer_names_clean, rotation=45, ha='right')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Importance Score')
        axes[1, 1].set_title('Score Range and Mean per Layer')
        axes[1, 1].legend(['Mean', 'Score Range'], loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Filter Importance Analysis - {sparsity_level:.0%} Sparsity', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'filter_importance_analysis_{int(sparsity_level*100)}pct.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store data for progressive analysis
        total_importance = np.sum(scores)
        self.progressive_data['total_importance'] = total_importance
    
    # def create_pruned_model_structured(self, original_model, pruning_percentage=0.5, calibration_dataset=None):
    #     """
    #     Create a new model with fewer filters based on GLOBAL magnitude pruning
    #     Enhanced with calibration-based importance calculation
    #     """
        
    #     input_shape = (CONFIG.INPUT_HEIGHT, CONFIG.INPUT_WIDTH, CONFIG.INPUT_CHANNELS)
    #     inputs = tf.keras.layers.Input(shape=input_shape, name='input_image')
        
    #     # Get weights from original model
    #     original_weights = self._extract_weights(original_model)
        
    #     if len(original_weights) < 10:
    #         raise ValueError(f"Insufficient weights found ({len(original_weights)} layers). Expected at least 10 conv layers.")
        
    #     # Calculate filter importance with calibration
    #     filter_info, layer_importance_weights = self.calculate_filter_importance_with_calibration(
    #         original_model, original_weights, calibration_dataset, num_samples=50
    #     )
        
    #     # Visualize filter importance
    #     self.visualize_filter_importance(filter_info, layer_importance_weights, pruning_percentage)
        
    #     # Sort filters globally by importance
    #     scores = np.array([score for _, _, score in filter_info])
    #     sorted_indices = np.argsort(scores)
        
    #     # Calculate how many filters to remove globally
    #     total_filters = len(filter_info)
    #     filters_to_remove = int(total_filters * pruning_percentage)
    #     filters_to_keep_count = total_filters - filters_to_remove
        
    #     # Calculate pruning threshold
    #     if filters_to_keep_count > 0:
    #         threshold_idx = sorted_indices[-filters_to_keep_count]
    #         pruning_threshold = scores[threshold_idx]
    #     else:
    #         pruning_threshold = np.max(scores)
        
    #     # Calculate total importance retained
    #     filters_to_keep_global = sorted_indices[-filters_to_keep_count:] if filters_to_keep_count > 0 else []
    #     importance_retained = np.sum(scores[filters_to_keep_global]) / np.sum(scores) * 100 if len(scores) > 0 else 0
        
    #     self.progressive_data['thresholds'].append(pruning_threshold)
    #     self.progressive_data['importance_retained'].append(importance_retained)
    #     self.progressive_data['filters_retained'].append(filters_to_keep_count / total_filters * 100)
    #     self.progressive_data['sparsity_levels'].append(pruning_percentage * 100)
        
    #     print(f"\nGlobal pruning statistics:")
    #     print(f"Total filters: {total_filters}")
    #     print(f"Filters to remove: {filters_to_remove}")
    #     print(f"Filters to keep: {filters_to_keep_count}")
    #     print(f"Actual global sparsity: {filters_to_remove/total_filters:.1%}")
    #     print(f"Pruning threshold: {pruning_threshold:.4f}")
    #     print(f"Importance retained: {importance_retained:.1f}%")
    #     print(f"Calibration samples used: {self.calibration_samples_used}")
        
    #     # Convert back to per-layer structure
    #     layers_to_prune = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 
    #                       'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
    #                       'p_conv1', 'p_conv2']
        
    #     filters_to_keep = {layer: [] for layer in layers_to_prune}
    #     for idx in filters_to_keep_global:
    #         layer_name, filter_idx, _ = filter_info[idx]
    #         filters_to_keep[layer_name].append(filter_idx)
        
    #     # Ensure minimum filters per layer
    #     new_filter_counts = self._ensure_minimum_filters(filters_to_keep, original_weights, layers_to_prune)
        
    #     # Visualize pruning distribution
    #     self.visualize_layer_pruning_heatmap(original_weights, filters_to_keep, pruning_percentage)
        
    #     # Build pruned model
    #     pruned_model = self._build_pruned_model(inputs, original_weights, filters_to_keep, new_filter_counts)
        
    #     return pruned_model
    def create_pruned_model_structured(self, original_model, pruning_percentage=0.5, calibration_dataset=None):
        """
        Create a new model with fewer filters based on per-layer proportional pruning
        """
        input_shape = (CONFIG.INPUT_HEIGHT, CONFIG.INPUT_WIDTH, CONFIG.INPUT_CHANNELS)
        inputs = tf.keras.layers.Input(shape=input_shape, name='input_image')

        # Get weights from original model
        original_weights = self._extract_weights(original_model)

        if len(original_weights) < 10:
            raise ValueError(f"Insufficient weights found ({len(original_weights)} layers). Expected at least 10 conv layers.")

        # Calculate filter importance with calibration
        filter_info, layer_importance_weights = self.calculate_filter_importance_with_calibration(
            original_model, original_weights, calibration_dataset, num_samples=50
        )

        # Visualize filter importance
        self.visualize_filter_importance(filter_info, layer_importance_weights, pruning_percentage)

        # Per-layer proportional pruning
        layers_to_prune = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                        'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
                        'p_conv1', 'p_conv2']

        filters_to_keep = {layer: [] for layer in layers_to_prune}

        for layer_name in layers_to_prune:
            layer_filter_indices = [(fi, score) for ln, fi, score in filter_info if ln == layer_name]
            num_layer_filters = len(layer_filter_indices)
            num_to_keep = max(1, int(num_layer_filters * (1 - pruning_percentage)))
            sorted_layer = sorted(layer_filter_indices, key=lambda x: x[1], reverse=True)
            kept = [fi for fi, score in sorted_layer[:num_to_keep]]
            filters_to_keep[layer_name] = sorted(kept)

        # Calculate global stats for reporting
        scores = np.array([score for _, _, score in filter_info])
        sorted_indices = np.argsort(scores)
        total_filters = len(filter_info)
        filters_to_keep_count = sum(len(v) for v in filters_to_keep.values())
        filters_to_remove = total_filters - filters_to_keep_count
        filters_to_keep_global = sorted_indices[filters_to_remove:] if filters_to_remove < total_filters else sorted_indices
        pruning_threshold = scores[sorted_indices[filters_to_remove]] if filters_to_remove < total_filters else 0.0
        importance_retained = np.sum(scores[filters_to_keep_global]) / np.sum(scores) * 100 if len(scores) > 0 else 0

        self.progressive_data['thresholds'].append(pruning_threshold)
        self.progressive_data['importance_retained'].append(importance_retained)
        self.progressive_data['filters_retained'].append(filters_to_keep_count / total_filters * 100)
        self.progressive_data['sparsity_levels'].append(pruning_percentage * 100)

        print(f"\nPer-layer pruning statistics:")
        print(f"Total filters: {total_filters}")
        print(f"Filters to remove: {filters_to_remove}")
        print(f"Filters to keep: {filters_to_keep_count}")
        print(f"Actual global sparsity: {filters_to_remove/total_filters:.1%}")
        print(f"Pruning threshold: {pruning_threshold:.4f}")
        print(f"Importance retained: {importance_retained:.1f}%")
        print(f"Calibration samples used: {self.calibration_samples_used}")

        # New filter counts from per-layer pruning
        new_filter_counts = {layer: len(filters_to_keep[layer]) for layer in layers_to_prune if layer in filters_to_keep}

        # Print per-layer breakdown
        for layer_name in layers_to_prune:
            if layer_name in original_weights:
                orig = original_weights[layer_name][0].shape[-1]
                kept = len(filters_to_keep[layer_name])
                print(f"{layer_name}: Keeping {kept}/{orig} filters ({kept/orig:.1%})")

        # Visualize pruning distribution
        self.visualize_layer_pruning_heatmap(original_weights, filters_to_keep, pruning_percentage)

        # Build pruned model
        pruned_model = self._build_pruned_model(inputs, original_weights, filters_to_keep, new_filter_counts)

        return pruned_model
    
    def visualize_layer_pruning_heatmap(self, original_weights, filters_to_keep, sparsity_level):
        """Create heatmap showing pruning distribution across layers"""
        
        layers = []
        original_counts = []
        kept_counts = []
        pruned_counts = []
        pruned_percentages = []
        
        for layer_name in filters_to_keep:
            if layer_name in original_weights:
                orig = original_weights[layer_name][0].shape[-1]
                kept = len(filters_to_keep[layer_name])
                pruned = orig - kept
                pruned_pct = (pruned / orig) * 100
                
                layers.append(layer_name)
                original_counts.append(orig)
                kept_counts.append(kept)
                pruned_counts.append(pruned)
                pruned_percentages.append(pruned_pct)
        
        # Store for later analysis
        self.layer_pruning_stats[f"{int(sparsity_level*100)}pct"] = {
            'layers': layers,
            'original': original_counts,
            'kept': kept_counts,
            'pruned': pruned_counts,
            'pruned_pct': pruned_percentages
        }
        
        # Create heatmap data
        data = np.array([original_counts, kept_counts, pruned_percentages])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create heatmap with better formatting
        heatmap = sns.heatmap(data, annot=True, fmt='.0f', cmap='RdYlGn_r',
                             xticklabels=layers, 
                             yticklabels=['Original Filters', 'Kept Filters', 'Pruned (%)'],
                             cbar_kws={'label': 'Value'}, linewidths=0.5)
        
        plt.title(f'Layer-wise Pruning Distribution - {sparsity_level:.0%} Sparsity', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'layer_pruning_heatmap_{int(sparsity_level*100)}pct.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_pruning_progression_analysis(self):
        """
        Create the 4 pruning progression plots:
        1. Threshold progression
        2. Total importance retained  
        3. MAE degradation
        4. Filters retained
        """
        if len(self.progressive_data['sparsity_levels']) < 2:
            print("Not enough data points for progression analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        sparsity_levels = self.progressive_data['sparsity_levels']
        
        # 1. Threshold progression - shows how selective pruning becomes
        if self.progressive_data['thresholds']:
            axes[0, 0].plot(sparsity_levels, self.progressive_data['thresholds'], 
                           'b-o', linewidth=2, markersize=8, markerfacecolor='lightblue')
            axes[0, 0].fill_between(sparsity_levels, self.progressive_data['thresholds'], 
                                   alpha=0.3, color='blue')
            axes[0, 0].set_xlabel('Sparsity Level (%)')
            axes[0, 0].set_ylabel('Importance Threshold')
            axes[0, 0].set_title('Threshold Progression\n(How selective pruning becomes)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add annotations for key points
            for i, (x, y) in enumerate(zip(sparsity_levels, self.progressive_data['thresholds'])):
                if i % 2 == 0:  # Annotate every other point to avoid crowding
                    axes[0, 0].annotate(f'{y:.3f}', (x, y), xytext=(5, 5), 
                                       textcoords='offset points', fontsize=8)
        
        # 2. Total importance retained - percentage of cumulative filter importance kept
        if self.progressive_data['importance_retained']:
            axes[0, 1].plot(sparsity_levels, self.progressive_data['importance_retained'], 
                           'g-s', linewidth=2, markersize=8, markerfacecolor='lightgreen')
            axes[0, 1].fill_between(sparsity_levels, self.progressive_data['importance_retained'], 
                                   alpha=0.3, color='green')
            axes[0, 1].set_xlabel('Sparsity Level (%)')
            axes[0, 1].set_ylabel('Importance Retained (%)')
            axes[0, 1].set_title('Total Importance Retained\n(% of cumulative filter importance kept)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 100])
            
            # Add horizontal reference lines
            axes[0, 1].axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50%')
            axes[0, 1].axhline(y=25, color='red', linestyle='--', alpha=0.7, label='25%')
            axes[0, 1].legend()
        
        # 3. MAE degradation - extract from results
        if self.results and len(self.results) > 1:
            baseline_mae = self.results[0]['mae']
            mae_degradations = []
            sparsity_from_results = []
            
            for result in self.results[1:]:  # Skip baseline (0% sparsity)
                mae_increase = ((result['mae'] - baseline_mae) / baseline_mae) * 100
                mae_degradations.append(mae_increase)
                sparsity_from_results.append(result['sparsity'] * 100)
            
            if mae_degradations:
                # Store for progressive data
                self.progressive_data['mae_degradation'] = mae_degradations
                
                axes[1, 0].plot(sparsity_from_results, mae_degradations, 
                               'r-^', linewidth=2, markersize=8, markerfacecolor='pink')
                axes[1, 0].fill_between(sparsity_from_results, mae_degradations, 
                                       alpha=0.3, color='red')
                axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Baseline')
                axes[1, 0].axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='10% degradation')
                axes[1, 0].axhline(y=25, color='red', linestyle='--', alpha=0.7, label='25% degradation')
                axes[1, 0].set_xlabel('Sparsity Level (%)')
                axes[1, 0].set_ylabel('MAE Increase (%)')
                axes[1, 0].set_title('MAE Degradation\n(Accuracy loss from baseline)')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
        
        # 4. Filters retained - shows the reduction curve
        if self.progressive_data['filters_retained']:
            axes[1, 1].plot(sparsity_levels, self.progressive_data['filters_retained'], 
                           'purple', marker='D', linewidth=2, markersize=8, 
                           markerfacecolor='plum')
            axes[1, 1].fill_between(sparsity_levels, self.progressive_data['filters_retained'], 
                                   alpha=0.3, color='purple')
            axes[1, 1].set_xlabel('Sparsity Level (%)')
            axes[1, 1].set_ylabel('Filters Retained (%)')
            axes[1, 1].set_title('Filters Retained\n(Shows the reduction curve)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 100])
            
            # Add diagonal reference line showing expected linear reduction
            x_ref = np.linspace(min(sparsity_levels), max(sparsity_levels), 100)
            y_ref = 100 - x_ref  # Perfect linear reduction
            axes[1, 1].plot(x_ref, y_ref, '--', color='gray', alpha=0.5, 
                           label='Linear reduction')
            axes[1, 1].legend()
        
        plt.suptitle('Pruning Progression Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'pruning_progression_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create individual plots for better clarity
        self._create_individual_progression_plots()
        
        print(f"Pruning progression analysis saved to {self.viz_dir}")
    
    def _create_individual_progression_plots(self):
        """Create individual plots for each progression metric"""
        
        sparsity_levels = self.progressive_data['sparsity_levels']
        
        # Create individual plots
        metrics = [
            ('thresholds', 'Importance Threshold', 'Threshold Progression', 'blue'),
            ('importance_retained', 'Importance Retained (%)', 'Total Importance Retained', 'green'),
            ('filters_retained', 'Filters Retained (%)', 'Filters Retained', 'purple')
        ]
        
        for metric_key, ylabel, title, color in metrics:
            if metric_key in self.progressive_data and self.progressive_data[metric_key]:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                values = self.progressive_data[metric_key]
                ax.plot(sparsity_levels, values, f'{color[0]}-o', 
                       linewidth=3, markersize=10, markerfacecolor=f'light{color}')
                ax.fill_between(sparsity_levels, values, alpha=0.3, color=color)
                
                ax.set_xlabel('Sparsity Level (%)', fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add value annotations
                for x, y in zip(sparsity_levels, values):
                    if metric_key == 'thresholds':
                        ax.annotate(f'{y:.3f}', (x, y), xytext=(0, 10), 
                                   textcoords='offset points', ha='center', fontsize=9)
                    else:
                        ax.annotate(f'{y:.1f}%', (x, y), xytext=(0, 10), 
                                   textcoords='offset points', ha='center', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(self.viz_dir / f'{metric_key}_progression.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # MAE degradation individual plot
        if self.progressive_data.get('mae_degradation'):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use sparsity levels from results (excluding baseline)
            result_sparsities = [r['sparsity'] * 100 for r in self.results[1:]]
            mae_degradations = self.progressive_data['mae_degradation']
            
            ax.plot(result_sparsities, mae_degradations, 'r-^', 
                   linewidth=3, markersize=10, markerfacecolor='pink')
            ax.fill_between(result_sparsities, mae_degradations, alpha=0.3, color='red')
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Baseline')
            ax.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='10% degradation')
            
            ax.set_xlabel('Sparsity Level (%)', fontsize=12)
            ax.set_ylabel('MAE Increase (%)', fontsize=12)
            ax.set_title('MAE Degradation', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add value annotations
            for x, y in zip(result_sparsities, mae_degradations):
                ax.annotate(f'{y:.1f}%', (x, y), xytext=(0, 10), 
                           textcoords='offset points', ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'mae_degradation_progression.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _extract_weights(self, original_model):
        """Extract weights from the original model"""
        original_weights = {}
        
        # Check if we have a wrapped SACNN model
        sacnn_layer = None
        for layer in original_model.layers:
            if 'SingleScaleSACNN' in str(type(layer)) or layer.name == 'SingleScaleSACNN':
                sacnn_layer = layer
                break
        
        if sacnn_layer is not None:
            # Access the internal layers of the SACNN model
            layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                          'conv3_1', 'conv3_2', 'conv3_3', 
                          'conv4_1', 'conv4_2', 'conv4_3',
                          'p_conv1', 'p_conv2', 'p_conv3']
            
            for layer_name in layer_names:
                if hasattr(sacnn_layer, layer_name):
                    sublayer = getattr(sacnn_layer, layer_name)
                    if hasattr(sublayer, 'get_weights') and len(sublayer.get_weights()) > 0:
                        save_name = 'density_map' if layer_name == 'p_conv3' else layer_name
                        original_weights[save_name] = sublayer.get_weights()
        else:
            # Try direct layer access for functional models
            for layer in original_model.layers:
                if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                    if any(name in layer.name for name in ['conv', 'p_conv', 'density']):
                        original_weights[layer.name] = layer.get_weights()
        
        # If still no weights, try full extraction
        if len(original_weights) < 10:
            all_weights = original_model.get_weights()
            if len(all_weights) >= 26:
                weight_idx = 0
                layer_configs = [
                    ('conv1_1', (3, 3, 3, 64)),
                    ('conv1_2', (3, 3, 64, 64)),
                    ('conv2_1', (3, 3, 64, 128)),
                    ('conv2_2', (3, 3, 128, 128)),
                    ('conv3_1', (3, 3, 128, 256)),
                    ('conv3_2', (3, 3, 256, 256)),
                    ('conv3_3', (3, 3, 256, 256)),
                    ('conv4_1', (3, 3, 256, 512)),
                    ('conv4_2', (3, 3, 512, 512)),
                    ('conv4_3', (3, 3, 512, 512)),
                    ('p_conv1', (3, 3, 512, 256)),
                    ('p_conv2', (3, 3, 256, 128)),
                    ('density_map', (1, 1, 128, 1))
                ]
                
                for layer_name, expected_shape in layer_configs:
                    if weight_idx < len(all_weights) - 1:
                        kernel = all_weights[weight_idx]
                        bias = all_weights[weight_idx + 1]
                        if len(kernel.shape) == 4 and kernel.shape[0] == expected_shape[0]:
                            original_weights[layer_name] = [kernel, bias]
                            weight_idx += 2
        
        return original_weights
    
    def _ensure_minimum_filters(self, filters_to_keep, original_weights, layers_to_prune):
        """Ensure minimum filters per layer for network stability"""
        new_filter_counts = {}
        
        min_filters = {
    'conv1_1': 4, 'conv1_2': 8,
    'conv2_1': 16, 'conv2_2': 16,
    'conv3_1': 32, 'conv3_2': 32, 'conv3_3': 32,
    'conv4_1': 32, 'conv4_2': 32, 'conv4_3': 32,
    'p_conv1': 16, 'p_conv2': 8
}
        
        for layer_name in layers_to_prune:
            if layer_name in original_weights:
                original_count = original_weights[layer_name][0].shape[-1]
                kept_filters = sorted(filters_to_keep[layer_name])
                min_keep = min_filters.get(layer_name, 1)
                
                if len(kept_filters) < min_keep:
                    weights = original_weights[layer_name][0]
                    filter_scores = [np.linalg.norm(weights[..., i]) for i in range(original_count)]
                    top_indices = np.argsort(filter_scores)[-min_keep:]
                    kept_filters = sorted(top_indices)
                
                filters_to_keep[layer_name] = kept_filters
                new_filter_counts[layer_name] = len(kept_filters)
                
                print(f"{layer_name}: Keeping {len(kept_filters)}/{original_count} filters ({len(kept_filters)/original_count:.1%})")
        
        return new_filter_counts
    
    def _build_pruned_model(self, inputs, original_weights, filters_to_keep, new_filter_counts):
        """Build the pruned model with reduced filters"""
        x = inputs
        prev_layer_indices = None
        
        # Build all layers
        layer_sequence = [
            ('conv1_1', None), ('conv1_2', None), ('pool1', 'pool'),
            ('conv2_1', None), ('conv2_2', None), ('pool2', 'pool'),
            ('conv3_1', None), ('conv3_2', None), ('conv3_3', None), ('pool3', 'pool'),
            ('conv4_1', None), ('conv4_2', None), ('conv4_3', None),
            ('p_conv1', None), ('p_conv2', None)
        ]
        
        for layer_name, layer_type in layer_sequence:
            if layer_type == 'pool':
                x = tf.keras.layers.MaxPooling2D(2, 2, name=layer_name)(x)
            else:
                x = self._create_pruned_conv(x, layer_name, original_weights, filters_to_keep, 
                                            new_filter_counts, prev_layer_indices)
                prev_layer_indices = filters_to_keep.get(layer_name)
        
        # Final density layer
        if 'density_map' in original_weights:
            weights = original_weights['density_map']
            if prev_layer_indices is not None:
                kernel = weights[0][:, :, prev_layer_indices, :]
                bias = weights[1] if len(weights) > 1 else None
                
                density_layer = tf.keras.layers.Conv2D(1, 1, padding='same', name='density_map')
                density_map = density_layer(x)
                
                new_weights = [kernel]
                if bias is not None:
                    new_weights.append(bias)
                density_layer.set_weights(new_weights)
            else:
                density_layer = tf.keras.layers.Conv2D(1, 1, padding='same', name='density_map')
                density_map = density_layer(x)
                density_layer.set_weights(original_weights['density_map'])
        else:
            density_map = tf.keras.layers.Conv2D(1, 1, padding='same', name='density_map')(x)
        
        # count = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True))(density_map)
        count = SumLayer(name='count')(density_map)
        
        return tf.keras.Model(
            inputs=inputs,
            outputs={'density_map': density_map, 'count': count},
            name='PrunedSingleScaleSACNN'
        )
    
    def _create_pruned_conv(self, x, layer_name, original_weights, filters_to_keep, 
                           new_filter_counts, prev_layer_indices):
        """Create a pruned convolutional layer"""
        
        if layer_name in filters_to_keep and layer_name in new_filter_counts:
            weights = original_weights.get(layer_name)
            if weights is None:
                return tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name=layer_name)(x)
            
            kernel = weights[0]
            bias = weights[1] if len(weights) > 1 else None
            
            keep_indices = filters_to_keep[layer_name]
            num_filters = new_filter_counts[layer_name]
            
            if len(keep_indices) == 0:
                return tf.keras.layers.Conv2D(1, 3, padding='same', activation='relu', name=layer_name)(x)
            
            pruned_kernel = kernel[..., keep_indices]
            
            if prev_layer_indices is not None and len(prev_layer_indices) > 0:
                pruned_kernel = pruned_kernel[:, :, prev_layer_indices, :]
            
            pruned_bias = bias[keep_indices] if bias is not None else None
            
            new_layer = tf.keras.layers.Conv2D(num_filters, 3, padding='same', 
                                              activation='relu', name=layer_name)
            output = new_layer(x)
            
            new_weights = [pruned_kernel]
            if pruned_bias is not None:
                new_weights.append(pruned_bias)
            new_layer.set_weights(new_weights)
            
            return output
        else:
            if layer_name in original_weights:
                num_filters = original_weights[layer_name][0].shape[-1]
                new_layer = tf.keras.layers.Conv2D(num_filters, 3, padding='same',
                                                  activation='relu', name=layer_name)
                output = new_layer(x)
                new_layer.set_weights(original_weights[layer_name])
                return output
            else:
                return tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name=layer_name)(x)
    
    def evaluate_model(self, model, test_dataset):
        """Evaluate model MAE and MSE on test set"""
        all_predictions = []
        all_ground_truths = []
        
        for batch in test_dataset:
            images, targets = batch
            predictions = model(images, training=False)
            
            pred_density = predictions['density_map']
            pred_count = tf.reduce_sum(pred_density, axis=[1, 2, 3]).numpy().flatten()
            
            true_count = targets['count'].numpy().flatten()
            
            all_predictions.extend(pred_count)
            all_ground_truths.extend(true_count)
        
        all_predictions = np.array(all_predictions)
        all_ground_truths = np.array(all_ground_truths)
        
        mae = np.mean(np.abs(all_predictions - all_ground_truths))
        mse = np.mean((all_predictions - all_ground_truths) ** 2)
        
        return float(mae), float(mse)
    
    def fine_tune_model(self, model, train_dataset, val_dataset, epochs=100, initial_lr=1e-4, sparsity_label=""):
        """Fine-tune the pruned model using custom training loop for Keras 3 compatibility"""
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=initial_lr,
            momentum=0.9
        )
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        patience = 15
        history_list = []
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        patience = 15
        history_list = []
        
        for epoch in range(epochs):
            # Training loop
            train_losses = []
            for x_batch, y_batch in train_dataset:
                with tf.GradientTape() as tape:
                    predictions = model(x_batch, training=True)
                    density_loss = euclidean_loss(
                        y_batch['density_map'],
                        predictions['density_map']
                    )
                    total_loss = density_loss
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_losses.append(total_loss.numpy())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation loop
            val_losses = []
            for x_batch, y_batch in val_dataset:
                predictions = model(x_batch, training=False)
                density_loss = euclidean_loss(
                    y_batch['density_map'],
                    predictions['density_map']
                )
                val_losses.append(density_loss.numpy())
            
            avg_val_loss = np.mean(val_losses)
            
            epoch_logs = {
                'loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            history_list.append(epoch_logs)
            
            print(f"  Fine-tune epoch {epoch + 1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
            
            # ReduceLROnPlateau equivalent
            if patience_counter == 7:
                current_lr = float(optimizer.learning_rate)
                new_lr = max(current_lr * 0.5, 1e-7)
                optimizer.learning_rate.assign(new_lr)
                print(f"  Reducing LR to {new_lr:.2e}")
        
        # Restore best weights
        if best_weights is not None:
            model.set_weights(best_weights)
        
        if sparsity_label:
            self.fine_tune_histories[sparsity_label] = history_list
        
        return model
    
    def structured_magnitude_pruning(self, model, sparsity_levels, part='B'):
        """Apply structured magnitude pruning with multiple sparsity levels"""
        
        print("\n" + "="*80)
        print("Enhanced Structured Magnitude Pruning with Global Sparsity Control")
        print("="*80)
        
        # Load datasets
        train_dataset, val_dataset, test_dataset = create_train_val_datasets(
            part=part,
            batch_size=CONFIG.BATCH_SIZE,
            val_split=0.1
        )
        
        # Evaluate base model
        print("\nEvaluating base model...")
        base_mae, base_mse = self.evaluate_model(model, test_dataset)
        base_params = self.count_parameters(model)
        base_size = self.get_model_size_mb(model)
        
        print(f"Base model: {base_params:,} parameters, {base_size:.2f} MB")
        print(f"MAE: {base_mae:.2f}, MSE: {base_mse:.2f}")
        
        # Initialize progressive data with baseline
        self.progressive_data['sparsity_levels'].append(0)
        self.progressive_data['thresholds'].append(0)
        self.progressive_data['importance_retained'].append(100)
        self.progressive_data['filters_retained'].append(100)
        
        # Save base results
        self.results.append({
            'sparsity': 0.0,
            'mae': base_mae,
            'mse': base_mse,
            'parameters': base_params,
            'size_mb': base_size,
            'compression_ratio': 1.0,
            'calibration_samples': 0
        })
        
        # Try different sparsity levels
        for sparsity in sparsity_levels:
            print(f"\n{'='*60}")
            print(f"Creating structurally pruned model with {sparsity:.0%} global sparsity")
            print(f"{'='*60}")
            
            # Create pruned model with calibration
            pruned_model = self.create_pruned_model_structured(
                model, sparsity, calibration_dataset=train_dataset
            )
            
            # Fine-tune the pruned model
            print("\nFine-tuning pruned model...")
            pruned_model = self.fine_tune_model(
                pruned_model, train_dataset, val_dataset, 
                epochs=100, initial_lr=CONFIG.INITIAL_LR * 0.1,
                sparsity_label=f"{int(sparsity*100)}pct"
            )
            
            # Evaluate pruned model
            pruned_mae, pruned_mse = self.evaluate_model(pruned_model, test_dataset)
            pruned_params = self.count_parameters(pruned_model)
            pruned_size = self.get_model_size_mb(pruned_model)
            
            # Save model
            model_path = self.pruned_models_dir / f"structured_pruned_{int(sparsity*100)}pct.h5"
            pruned_model.save(model_path)
            
            # Calculate compression
            compression_ratio = base_params / pruned_params if pruned_params > 0 else 0
            
            print(f"\n{'='*40}")
            print(f"Results for {sparsity:.0%} sparsity:")
            print(f"  Original: {base_params:,} params, {base_size:.2f} MB")
            print(f"  Pruned: {pruned_params:,} params, {pruned_size:.2f} MB")
            print(f"  MAE: {pruned_mae:.2f} (Δ {pruned_mae - base_mae:+.2f})")
            print(f"  MSE: {pruned_mse:.2f} (Δ {pruned_mse - base_mse:+.2f})")
            print(f"  Compression: {compression_ratio:.2f}x")
            print(f"  Calibration samples: {self.calibration_samples_used}")
            print(f"{'='*40}")
            
            # Save results
            self.results.append({
                'sparsity': sparsity,
                'mae': pruned_mae,
                'mse': pruned_mse,
                'parameters': pruned_params,
                'size_mb': pruned_size,
                'compression_ratio': compression_ratio,
                'mae_increase': pruned_mae - base_mae,
                'mse_increase': pruned_mse - base_mse,
                'model_path': str(model_path),
                'calibration_samples': self.calibration_samples_used
            })
            
            # Save results to JSON
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            # Stop if MAE degrades too much
            if pruned_mae > base_mae + 20:
                print(f"\nStopping: MAE degraded too much ({pruned_mae:.2f})")
                break
        
        # Create final visualizations and tables
        self.create_pruning_progression_analysis()  
        self.create_summary_visualizations()
        self.create_results_table()
        self.print_summary()
        
        return self.results
    
    def create_results_table(self):
        """Generate detailed results table for documentation"""
        
        table_data = []
        for r in self.results:
            table_data.append({
                'Sparsity (%)': f"{r['sparsity']*100:.0f}",
                'MAE': f"{r['mae']:.2f}",
                'MSE': f"{r['mse']:.2f}",
                'RMSE': f"{np.sqrt(r['mse']):.2f}",
                'Parameters': f"{r['parameters']:,}",
                'Size (MB)': f"{r['size_mb']:.2f}",
                'Compression': f"{r['compression_ratio']:.1f}x",
                'MAE Δ': f"{r.get('mae_increase', 0):+.2f}",
                'MSE Δ': f"{r.get('mse_increase', 0):+.2f}"
            })
        
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        df.to_csv(self.pruned_models_dir / 'results_table.csv', index=False)
        
        # Save as LaTeX
        with open(self.pruned_models_dir / 'results_table.tex', 'w') as f:
            f.write(df.to_latex(index=False))
        
        # Save as formatted text
        with open(self.pruned_models_dir / 'results_table.txt', 'w') as f:
            f.write(df.to_string(index=False))
        
        print(f"\nResults table saved to {self.pruned_models_dir}")
        
        return df
    
    def create_summary_visualizations(self):
        """Create comprehensive summary visualizations"""
        
        if len(self.results) < 2:
            print("Not enough results for visualization")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract data
        sparsities = [r['sparsity'] * 100 for r in self.results]
        maes = [r['mae'] for r in self.results]
        mses = [r['mse'] for r in self.results]
        sizes = [r['size_mb'] for r in self.results]
        params = [r['parameters'] for r in self.results]
        compressions = [r['compression_ratio'] for r in self.results]
        
        # 1. MAE vs Sparsity
        axes[0, 0].plot(sparsities, maes, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].axhline(y=self.results[0]['mae'], color='r', linestyle='--', 
                          label=f'Baseline ({self.results[0]["mae"]:.1f})')
        axes[0, 0].axhline(y=33.2, color='g', linestyle='--', label='Paper Target (33.2)')
        axes[0, 0].set_xlabel('Sparsity (%)')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_title('Accuracy vs Pruning Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Model Size vs Sparsity
        axes[0, 1].plot(sparsities, sizes, 'g-s', linewidth=2, markersize=8)
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='1MB Target')
        axes[0, 1].set_xlabel('Sparsity (%)')
        axes[0, 1].set_ylabel('Model Size (MB)')
        axes[0, 1].set_title('Model Size Reduction')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # 3. Pareto Frontier
        scatter = axes[0, 2].scatter(sizes, maes, c=sparsities, cmap='viridis', s=100)
        for i, sp in enumerate(sparsities):
            axes[0, 2].annotate(f'{sp:.0f}%', (sizes[i], maes[i]), 
                            xytext=(5, 5), textcoords='offset points')
        axes[0, 2].set_xlabel('Model Size (MB)')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].set_title('Size-Accuracy Pareto Frontier')
        axes[0, 2].set_xscale('log')
        plt.colorbar(scatter, ax=axes[0, 2], label='Sparsity (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Compression Ratio
        axes[1, 0].bar(range(len(sparsities)), compressions, color='orange', edgecolor='black')
        axes[1, 0].set_xticks(range(len(sparsities)))
        axes[1, 0].set_xticklabels([f'{s:.0f}%' for s in sparsities])
        axes[1, 0].set_xlabel('Sparsity')
        axes[1, 0].set_ylabel('Compression Ratio')
        axes[1, 0].set_title('Compression Achieved')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Parameter Reduction
        axes[1, 1].bar(range(len(sparsities)), params, color='purple', edgecolor='black')
        axes[1, 1].set_xticks(range(len(sparsities)))
        axes[1, 1].set_xticklabels([f'{s:.0f}%' for s in sparsities])
        axes[1, 1].set_xlabel('Sparsity')
        axes[1, 1].set_ylabel('Parameters')
        axes[1, 1].set_title('Parameter Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Fine-tuning histories
        if self.fine_tune_histories:
            for label, history in self.fine_tune_histories.items():
                epochs = range(1, len(history) + 1)
                val_losses = [h.get('val_loss', 0) for h in history]
                if val_losses and val_losses[0] != 0:
                    axes[1, 2].plot(epochs, val_losses, label=f'{label} sparsity')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Validation Loss')
            axes[1, 2].set_title('Fine-tuning Progress')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Structured Pruning Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'pruning_summary_complete.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nAll visualizations saved to {self.viz_dir}")
    
    def count_parameters(self, model):
        """Count total parameters in model"""
        return int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    
    def get_model_size_mb(self, model):
        """Get model size in MB"""
        temp_path = self.pruned_models_dir / "temp_model.h5"
        model.save(temp_path)
        size_mb = temp_path.stat().st_size / (1024 * 1024)
        temp_path.unlink()
        return size_mb
    
    def print_summary(self):
        """Print detailed results summary"""
        print("\n" + "="*110)
        print("ENHANCED STRUCTURED PRUNING SUMMARY")
        print("="*110)
        print(f"{'Sparsity':<10} {'MAE':<10} {'MSE':<10} {'MAE Δ':<10} {'Params':<15} {'Size (MB)':<12} {'Compression':<12}")
        print("-"*110)
        
        base_mae = self.results[0]['mae']
        base_mse = self.results[0]['mse']
        
        for r in self.results:
            sparsity = r['sparsity']
            mae = r['mae']
            mse = r['mse']
            mae_increase = mae - base_mae
            params = r['parameters']
            size = r['size_mb']
            compression = r['compression_ratio']
            
            print(f"{sparsity:>7.0%}    {mae:>7.2f}    {mse:>7.2f}    {mae_increase:>+7.2f}    "
                  f"{params:>13,}    {size:>9.2f}    {compression:>9.2f}x")
        
        print("="*110)
        
        # Additional statistics
        print(f"\nCalibration samples used: {self.calibration_samples_used}")
        print(f"Experiment directory: {self.pruned_models_dir}")
        
        # Find best model under 1MB
        best_under_1mb = None
        for r in self.results:
            if r['size_mb'] < 1.0:
                if best_under_1mb is None or r['mae'] < best_under_1mb['mae']:
                    best_under_1mb = r
        
        if best_under_1mb:
            print(f"\nBest model under 1MB:")
            print(f"  Sparsity: {best_under_1mb['sparsity']:.0%}")
            print(f"  Size: {best_under_1mb['size_mb']:.2f} MB")
            print(f"  MAE: {best_under_1mb['mae']:.2f}")
            print(f"  MSE: {best_under_1mb['mse']:.2f}")
            print(f"  Compression: {best_under_1mb['compression_ratio']:.2f}x")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Structured Pruning for SingleScaleSACNN")
    parser.add_argument('--model', type=str, required=True, help='Path to base model')
    parser.add_argument('--part', type=str, default='B', choices=['A', 'B', 'mixed'])
    parser.add_argument('--sparsity', type=str, default='0.3,0.5,0.7,0.9',
                        help='Comma-separated sparsity levels to try')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    # Parse sparsity levels
    sparsity_levels = [float(s) for s in args.sparsity.split(',')]
    
    # Create pruner
    pruner = StructuredPruner(
        base_model_path=args.model,
        experiment_name=args.name
    )
    
    # Load base model
    print(f"Loading model from {args.model}...")
    base_model = pruner.load_model(args.model)
    
    # Run pruning experiment
    results = pruner.structured_magnitude_pruning(
        model=base_model,
        sparsity_levels=sparsity_levels,
        part=args.part
    )
    
    print(f"\nExperiment complete!")
    print(f"Results saved to: {pruner.results_file}")
    print(f"Models saved to: {pruner.pruned_models_dir}")
    print(f"Visualizations saved to: {pruner.viz_dir}")
    print(f"Results table saved in multiple formats")
    print(f"Progressive analysis plots created!")


if __name__ == "__main__":
    main()