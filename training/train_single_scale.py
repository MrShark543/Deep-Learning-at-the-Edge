# """
# Training script for Single-Scale SACNN
# """

# import tensorflow as tf
# import numpy as np
# from pathlib import Path
# from datetime import datetime
# import json
# import sys
# import os

# # Suppress TF warnings about CUDA
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Add project root to path
# sys.path.append(str(Path(__file__).parent.parent))

# from config.single_scale_config import CONFIG
# from models.single_scale_vgg import create_single_scale_model, count_parameters
# from data.simple_loader import SimpleDataLoader, create_train_val_datasets
# from models.losses import get_loss_functions, get_metrics
# from utils.cuda_setup import auto_setup, estimate_training_time

# class SingleScaleTrainer:
#     """Trainer for Single-Scale SACNN"""
    
#     def __init__(self, experiment_name=None):
#         """
#         Initialize trainer
        
#         Args:
#             experiment_name: Name for this experiment
#         """
        
#         # Setup GPU/CPU
#         self.gpu_available = auto_setup()
        
#         # Create experiment directory
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         if experiment_name:
#             self.exp_name = f"{experiment_name}_{timestamp}"
#         else:
#             self.exp_name = f"single_scale_{timestamp}"
        
#         self.exp_dir = CONFIG.SAVED_MODELS_DIR / self.exp_name
#         self.exp_dir.mkdir(parents=True, exist_ok=True)
        
#         # Setup logging
#         self.log_file = self.exp_dir / "training.log"
        
#         print(f"Experiment: {self.exp_name}")
#         print(f"Directory: {self.exp_dir}")
#         print(f"Device: {'GPU' if self.gpu_available else 'CPU'}")
    
#     def log(self, message):
#         """Log message to file and console"""
#         print(message)
#         with open(self.log_file, 'a') as f:
#             f.write(f"{datetime.now()}: {message}\n")
    
  
  


#     def create_callbacks(self, part='A'):
#         """
#         Create training callbacks
        
#         Args:
#             part: Dataset part ('A', 'B', or 'mixed')  
        
#         Returns:
#             List of callbacks
#         """
        
#         callbacks = []
        
#         # Model checkpoint
#         checkpoint_path = self.exp_dir / "best_model.h5"
#         callbacks.append(tf.keras.callbacks.ModelCheckpoint(
#             filepath=str(checkpoint_path),
#             monitor='val_loss',
#             save_best_only=True,
#             save_weights_only=False,
#             verbose=1
#         ))
#         callbacks.append(tf.keras.callbacks.EarlyStopping(
#         monitor='val_loss',
#         patience=50,
#         restore_best_weights=True,
#         verbose=1
#     ))
    
        
#         # Learning rate scheduler 
#         def lr_schedule(epoch):
#             lr = CONFIG.INITIAL_LR
#             for threshold in CONFIG.LR_DECAY_EPOCHS:
#                 if epoch >= threshold:
#                     lr *= CONFIG.LR_DECAY_FACTOR
#             return max(lr, CONFIG.FINAL_LR)
        
#         callbacks.append(tf.keras.callbacks.LearningRateScheduler(
#             lr_schedule, verbose=1
#         ))
        
#         # TensorBoard
#         tb_dir = self.exp_dir / "tensorboard"
#         callbacks.append(tf.keras.callbacks.TensorBoard(
#             log_dir=str(tb_dir),
#             histogram_freq=10,
#             write_graph=True
#         ))
        
#         # CSV logger
#         callbacks.append(tf.keras.callbacks.CSVLogger(
#             str(self.exp_dir / "history.csv")
#         ))
        
#         # Custom callback to add count loss after certain epochs
#         class AddCountLossCallback(tf.keras.callbacks.Callback):
#             def __init__(self, add_epoch=CONFIG.ADD_COUNT_LOSS_EPOCH):
#                 self.add_epoch = add_epoch
#                 self.added = False
                
#             def on_epoch_begin(self, epoch, logs=None):
#                 if epoch == self.add_epoch and not self.added:
#                     print(f"\nAdding count loss at epoch {epoch} <<<\n")
#                     # Update loss weights
#                     self.model.loss_weights = {
#                         'density_map': CONFIG.DENSITY_LOSS_WEIGHT,
#                         'count': CONFIG.COUNT_LOSS_WEIGHT
#                     }
#                     self.added = True
        
#         callbacks.append(AddCountLossCallback())
        
#         # Progress tracking 
#         class ProgressCallback(tf.keras.callbacks.Callback):
#             def __init__(self, target_mae):
#                 self.target_mae = target_mae
                
#             def on_epoch_end(self, epoch, logs=None):
#                 if logs and (epoch + 1) % 10 == 0:
#                     # Find the MAE metric in logs
#                     val_mae = None
#                     for key in logs.keys():
#                         if 'val' in key and 'mae' in key.lower() and 'count' in key.lower():
#                             val_mae = logs[key]
#                             break
#                     if val_mae is None and 'val_count_mae' in logs:
#                         val_mae = logs['val_count_mae']
#                     if val_mae is None:
#                         val_mae = 0
                    
#                     train_loss = logs.get('loss', 0)
#                     print(f"\n[Epoch {epoch+1}] "
#                         f"Train Loss: {train_loss:.4f}, "
#                         f"Val MAE: {val_mae:.2f} "
#                         f"(Target: {self.target_mae:.1f})")
        
    
#         if part == 'mixed':
#             # Calculate weighted average target for mixed dataset
#             # Part A: 300 train, Part B: 400 train
#             target_mae = (CONFIG.PAPER_TARGETS['A']['MAE'] * 300 + 
#                         CONFIG.PAPER_TARGETS['B']['MAE'] * 400) / 700
#         else:
#             target_mae = CONFIG.PAPER_TARGETS[part]['MAE']
        
#         callbacks.append(ProgressCallback(target_mae))
        
#         return callbacks
        
#     def train(self, part='A', epochs=None, batch_size=None, quick_test=False):
#         """
#         Train the model
        
#         Args:
#             part: Dataset part ('A' or 'B')
#             epochs: Number of epochs (None uses config)
#             batch_size: Batch size (None uses config)
#             quick_test: If True, uses fewer epochs for testing
        
#         Returns:
#             Trained model and history
#         """
        
#         # Set parameters
#         if quick_test:
#             epochs = CONFIG.QUICK_TEST_EPOCHS
#             self.log(f"Quick test mode: {epochs} epochs")
#         else:
#             epochs = epochs or CONFIG.EPOCHS
        
#         batch_size = batch_size or CONFIG.BATCH_SIZE
        
#         self.log(f"Training configuration:")
#         self.log(f"  Part: {part}")
#         self.log(f"  Epochs: {epochs}")
#         self.log(f"  Batch size: {batch_size}")
#         self.log(f"  Initial LR: {CONFIG.INITIAL_LR}")
#         self.log(f"  Device: {'GPU' if self.gpu_available else 'CPU'}")
        
#         # Estimate training time
#         num_samples = 400 
#         # est_time = estimate_training_time(num_samples, batch_size, epochs, self.gpu_available)
#         # self.log(f"  Estimated time: {est_time}")
        
#         # Load data
#         self.log("Loading datasets...")
#         train_dataset, val_dataset, test_dataset = create_train_val_datasets(
#             part=part,
#             batch_size=batch_size,
#             val_split=0.1  
#         )
        
#         # If no validation split, use test as validation
#         if val_dataset is None:
#             val_dataset = test_dataset
#             self.log("Using test set for validation")
        
#         # Create model
#         self.log("Creating model...")
#         if CONFIG.EDGE_DEPLOYMENT:
#             from models.single_scale_edge import create_single_scale_model as create_edge_model
#             model = create_edge_model(
#                 input_shape=(256, 256, 3),
#                 width_multiplier=CONFIG.WIDTH_MULTIPLIER,
#                 use_depthwise=CONFIG.USE_DEPTHWISE,
#                 dropout_rate=CONFIG.DROPOUT_RATE
#             )
#         else:
#             model = create_single_scale_model(input_shape=(256, 256, 3))
        
#         # Count parameters
#         param_info = count_parameters(model)
#         self.log(f"Model parameters: {param_info['total_params']:,}")
#         self.log(f"Model size (Float32): {param_info['float32_size_mb']:.2f} MB")
        
#         # Compile model
#         self.log("Compiling model...")
        
#         # Start with only density loss
#         initial_loss_weights = {
#             'density_map': CONFIG.DENSITY_LOSS_WEIGHT,
#             'count': 0.0  # Start with 0
#         }
        
#         model.compile(
#             optimizer=tf.keras.optimizers.SGD(
#                 learning_rate=CONFIG.INITIAL_LR,
#                 momentum=CONFIG.MOMENTUM
#             ),
#             loss=get_loss_functions(),
#             loss_weights=initial_loss_weights,
#             metrics=get_metrics()
#         )
        
#         # Save model architecture
#         with open(self.exp_dir / "model_summary.txt", 'w') as f:
#             model.summary(print_fn=lambda x: f.write(x + '\n'))
        
#         # Create callbacks
#         callbacks = self.create_callbacks(part)
        
#         # Train model
#         self.log("Starting training...")
#         self.log(f"Target MAE: {CONFIG.PAPER_TARGETS[part]['MAE']}")
        
#         history = model.fit(
#             train_dataset,
#             validation_data=val_dataset,
#             epochs=epochs,
#             callbacks=callbacks,
#             verbose=1
#         )
        
#         # Save final model
#         final_model_path = self.exp_dir / "final_model.h5"
#         model.save(str(final_model_path))
#         self.log(f"Model saved to {final_model_path}")
        
#         # Evaluate on test set
#         self.log("\nEvaluating on test set...")
#         test_results = model.evaluate(test_dataset, verbose=1)
        
#         # Save results
#         results = {
#             'experiment': self.exp_name,
#             'part': part,
#             'epochs': epochs,
#             'final_test_results': dict(zip(model.metrics_names, test_results)),
#             'target_mae': CONFIG.PAPER_TARGETS[part]['MAE'],
        
#         }
        
#         with open(self.exp_dir / "results.json", 'w') as f:
#             json.dump(results, f, indent=2)
        
 
#         mae_value = None
#         for i, name in enumerate(model.metrics_names):
#             if 'count' in name and 'mae' in name.lower():
#                 mae_value = test_results[i]
#                 break
        
#         if mae_value is not None:
#             self.log(f"\nFinal test MAE: {mae_value:.2f}")
#         else:
#             self.log(f"\nTest results: {dict(zip(model.metrics_names, test_results))}")
        
#         self.log(f"Target MAE: {CONFIG.PAPER_TARGETS[part]['MAE']:.1f}")
        
#         return model, history
    
    


# def run_training(part='B', quick_test=False):
#     """
#     Main training function
    
#     Args:
#         part: Dataset part ('A' or 'B')
#         quick_test: If True, run quick test
    
#     Returns:
#         Trainer instance
#     """
    
#     print("\n" + "="*60)
#     print("Single-Scale SACNN Training")
#     print("="*60)
    
#     # Print configuration
#     CONFIG.print_config()
    
#     # Create trainer
#     trainer = SingleScaleTrainer(experiment_name=f"part_{part}")
    
#     # Train model
#     model, history = trainer.train(
#         part=part,
#         quick_test=quick_test
#     )
    
    
#     print(f"\nTraining complete! Results saved to {trainer.exp_dir}")
    
#     return trainer


# if __name__ == "__main__":
#     # Run quick test
#     run_training(part='A', quick_test=True)


"""
Training script for Single-Scale SACNN
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys
import os

# Suppress TF warnings about CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.single_scale_config import CONFIG
from models.single_scale_vgg import create_single_scale_model, count_parameters
from data.simple_loader import SimpleDataLoader, create_train_val_datasets
from models.losses import get_loss_functions, get_metrics
from utils.cuda_setup import auto_setup, estimate_training_time

class SingleScaleTrainer:
    """Trainer for Single-Scale SACNN"""
    
    def __init__(self, experiment_name=None):
        """
        Initialize trainer
        
        Args:
            experiment_name: Name for this experiment
        """
        
        # Setup GPU/CPU
        self.gpu_available = auto_setup()
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.exp_name = f"{experiment_name}_{timestamp}"
        else:
            self.exp_name = f"single_scale_{timestamp}"
        
        self.exp_dir = CONFIG.SAVED_MODELS_DIR / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_file = self.exp_dir / "training.log"
        
        print(f"Experiment: {self.exp_name}")
        print(f"Directory: {self.exp_dir}")
        print(f"Device: {'GPU' if self.gpu_available else 'CPU'}")
    
    def log(self, message):
        """Log message to file and console"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()}: {message}\n")
    
  
  


    def create_callbacks(self, part='A'):
        """
        Create training callbacks
        
        Args:
            part: Dataset part ('A', 'B', or 'mixed')  
        
        Returns:
            List of callbacks
        """
        
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.exp_dir / "best_model.h5"
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))
        callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=75,
        restore_best_weights=True,
        verbose=1
    ))
    
        
        # Learning rate scheduler 
        def lr_schedule(epoch):
            lr = CONFIG.INITIAL_LR
            for threshold in CONFIG.LR_DECAY_EPOCHS:
                if epoch >= threshold:
                    lr *= CONFIG.LR_DECAY_FACTOR
            return max(lr, CONFIG.FINAL_LR)
        
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(
            lr_schedule, verbose=1
        ))
        
        # TensorBoard
        tb_dir = self.exp_dir / "tensorboard"
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=str(tb_dir),
            histogram_freq=10,
            write_graph=True
        ))
        
        # CSV logger
        callbacks.append(tf.keras.callbacks.CSVLogger(
            str(self.exp_dir / "history.csv")
        ))
        
        # Custom callback to add count loss after certain epochs
        class AddCountLossCallback(tf.keras.callbacks.Callback):
            def __init__(self, add_epoch=CONFIG.ADD_COUNT_LOSS_EPOCH):
                self.add_epoch = add_epoch
                self.added = False
                
            def on_epoch_begin(self, epoch, logs=None):
                if epoch == self.add_epoch and not self.added:
                    print(f"\nAdding count loss at epoch {epoch} <<<\n")
                    # Update loss weights
                    self.model.loss_weights = {
                        'density_map': CONFIG.DENSITY_LOSS_WEIGHT,
                        'count': CONFIG.COUNT_LOSS_WEIGHT
                    }
                    self.added = True
        
        callbacks.append(AddCountLossCallback())
        
        # Progress tracking 
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, target_mae):
                self.target_mae = target_mae
                
            def on_epoch_end(self, epoch, logs=None):
                if logs and (epoch + 1) % 10 == 0:
                    # Find the MAE metric in logs
                    val_mae = None
                    for key in logs.keys():
                        if 'val' in key and 'mae' in key.lower() and 'count' in key.lower():
                            val_mae = logs[key]
                            break
                    if val_mae is None and 'val_count_mae' in logs:
                        val_mae = logs['val_count_mae']
                    if val_mae is None:
                        val_mae = 0
                    
                    train_loss = logs.get('loss', 0)
                    print(f"\n[Epoch {epoch+1}] "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val MAE: {val_mae:.2f} "
                        f"(Target: {self.target_mae:.1f})")
        
    
        if part == 'mixed':
            # Calculate weighted average target for mixed dataset
            # Part A: 300 train, Part B: 400 train
            target_mae = (CONFIG.PAPER_TARGETS['A']['MAE'] * 300 + 
                        CONFIG.PAPER_TARGETS['B']['MAE'] * 400) / 700
        else:
            target_mae = CONFIG.PAPER_TARGETS[part]['MAE']
        
        callbacks.append(ProgressCallback(target_mae))
        
        return callbacks
        
    def train(self, part='B', epochs=None, batch_size=None, quick_test=False):
        """
        Train the model
        
        Args:
            part: Dataset part ('B')
            epochs: Number of epochs (None uses config)
            batch_size: Batch size (None uses config)
            quick_test: If True, uses fewer epochs for testing
        
        Returns:
            Trained model and history
        """
        
        # Set parameters
        if quick_test:
            epochs = CONFIG.QUICK_TEST_EPOCHS
            self.log(f"Quick test mode: {epochs} epochs")
        else:
            epochs = epochs or CONFIG.EPOCHS
        
        batch_size = batch_size or CONFIG.BATCH_SIZE
        
        self.log(f"Training configuration:")
        self.log(f"  Part: {part}")
        self.log(f"  Epochs: {epochs}")
        self.log(f"  Batch size: {batch_size}")
        self.log(f"  Initial LR: {CONFIG.INITIAL_LR}")
        self.log(f"  Device: {'GPU' if self.gpu_available else 'CPU'}")
        
        # Estimate training time
        num_samples = 400  # Part B training set size
        # est_time = estimate_training_time(num_samples, batch_size, epochs, self.gpu_available)
        # self.log(f"  Estimated time: {est_time}")
        
        # Load data
        self.log("Loading datasets...")
        train_dataset, val_dataset, test_dataset = create_train_val_datasets(
            part=part,
            batch_size=batch_size,
            val_split=0.1  
        )
        
        # If no validation split, use test as validation
        if val_dataset is None:
            val_dataset = test_dataset
            self.log("Using test set for validation")
        
        # Create model
        self.log("Creating model...")
        if CONFIG.EDGE_DEPLOYMENT:
            from models.single_scale_edge import create_single_scale_model as create_edge_model
            model = create_edge_model(
                input_shape=(256, 256, 3),
                width_multiplier=CONFIG.WIDTH_MULTIPLIER,
                use_depthwise=CONFIG.USE_DEPTHWISE,
                dropout_rate=CONFIG.DROPOUT_RATE
            )
        else:
            model = create_single_scale_model(input_shape=(256, 256, 3))
        
        # Count parameters
        param_info = count_parameters(model)
        self.log(f"Model parameters: {param_info['total_params']:,}")
        self.log(f"Model size (Float32): {param_info['float32_size_mb']:.2f} MB")
        
        # Compile model
        self.log("Compiling model...")
        
        # Start with only density loss
        initial_loss_weights = {
            'density_map': CONFIG.DENSITY_LOSS_WEIGHT,
            'count': 0.0  # Start with 0
        }
        
        # model.compile(
        #     optimizer=tf.keras.optimizers.SGD(
        #         learning_rate=CONFIG.INITIAL_LR,
        #         momentum=CONFIG.MOMENTUM
        #     ),
        #     loss=get_loss_functions(),
        #     loss_weights=initial_loss_weights,
        #     metrics=get_metrics()
        # )

        #for kaggle
        model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=CONFIG.INITIAL_LR,
        momentum=CONFIG.MOMENTUM
    ),
    loss={
        'density_map': get_loss_functions()['density_map'],
        'count': get_loss_functions()['count']
    },
    loss_weights={
        'density_map': CONFIG.DENSITY_LOSS_WEIGHT,
        'count': 0.0
    }
)
        
        # Save model architecture
        with open(self.exp_dir / "model_summary.txt", 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Create callbacks
        callbacks = self.create_callbacks(part)
        
        # Train model
        self.log("Starting training...")
        target_mae = CONFIG.PAPER_TARGETS.get(part, CONFIG.PAPER_TARGETS['B'])['MAE']
        self.log(f"Target MAE: {target_mae}")
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = self.exp_dir / "final_model.h5"
        model.save(str(final_model_path))
        self.log(f"Model saved to {final_model_path}")
        
        # Evaluate on test set
        self.log("\nEvaluating on test set...")
        test_results = model.evaluate(test_dataset, verbose=1)
        
        # Save results
        results = {
            'experiment': self.exp_name,
            'part': part,
            'epochs': epochs,
            'final_test_results': dict(zip(model.metrics_names, test_results)),
            'target_mae': CONFIG.PAPER_TARGETS.get(part, CONFIG.PAPER_TARGETS['B'])['MAE'],
        }
        
        with open(self.exp_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
 
        mae_value = None
        for i, name in enumerate(model.metrics_names):
            if 'count' in name and 'mae' in name.lower():
                mae_value = test_results[i]
                break
        
        if mae_value is not None:
            self.log(f"\nFinal test MAE: {mae_value:.2f}")
        else:
            self.log(f"\nTest results: {dict(zip(model.metrics_names, test_results))}")
        
        self.log(f"Target MAE: {CONFIG.PAPER_TARGETS.get(part, CONFIG.PAPER_TARGETS['B'])['MAE']:.1f}")
        
        return model, history
    
    


def run_training(part='B', quick_test=False):
    """
    Main training function
    
    Args:
        part: Dataset part ('B')
        quick_test: If True, run quick test
    
    Returns:
        Trainer instance
    """
    
    print("\n" + "="*60)
    print("Single-Scale SACNN Training")
    print("="*60)
    
    # Print configuration
    CONFIG.print_config()
    
    # Create trainer
    trainer = SingleScaleTrainer(experiment_name=f"part_{part}")
    
    # Train model
    model, history = trainer.train(
        part=part,
        quick_test=quick_test
    )
    
    
    print(f"\nTraining complete! Results saved to {trainer.exp_dir}")
    
    return trainer


if __name__ == "__main__":
    run_training(part='B', quick_test=True)