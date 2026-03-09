"""
Main script
"""

import argparse
import sys
from pathlib import Path


project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.single_scale_config import CONFIG
from training.train_single_scale import run_training, SingleScaleTrainer
if CONFIG.EDGE_DEPLOYMENT:
    from models.single_scale_edge import create_single_scale_model
else:
    from models.single_scale_vgg import create_single_scale_model
from data.simple_loader import SimpleDataLoader

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Single Scale SACNN for Crowd Counting "
    )
    
    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--part', 
        type=str, 
        default='A', 
        choices=['A', 'B', 'mixed'],  
        help='Dataset part to use (A, B, or mixed for combined A+B)'
    )
    train_parser.add_argument(
        '--epochs', 
        type=int, 
        default=None,
        help='Number of epochs'
    )
    train_parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick test with few epochs'
    )
    train_parser.add_argument(
        '--name', 
        type=str, 
        default=None,
        help='Experiment name'
    )
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to model file'
    )
    test_parser.add_argument(
        '--part', 
        type=str, 
        default='A',
        choices=['A', 'B', 'mixed'],  
        help='Dataset part to test on'
    )
    

    # Info command
    info_parser = subparsers.add_parser('info', help='Show dataset/model info')
    info_parser.add_argument(
        '--dataset', 
        action='store_true',
        help='Show dataset information'
    )
    info_parser.add_argument(
        '--model', 
        action='store_true',
        help='Show model information'
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'train':
        print("\n" + "="*60)
        print("Starting Single-Scale SACNN Training")
        
        # Display dataset choice
        if args.part == 'mixed':
            print("Dataset: Mixed (Part A + Part B combined)")
        else:
            print(f"Dataset: Part {args.part}")
        print("="*60)
        
        # Create trainer 
        trainer = SingleScaleTrainer(experiment_name=args.name)
        
        # Running training
        model, history = trainer.train(
            part=args.part,  
            epochs=args.epochs,
            batch_size=CONFIG.BATCH_SIZE,
            quick_test=args.quick
        )
        
        print("\nTraining complete!")
        print(f"Results saved to: {trainer.exp_dir}")
        
    elif args.command == 'test':
        print("\nTesting model...")
        
        # Display dataset choice
        if args.part == 'mixed':
            print("Testing on: Mixed dataset (Part A + Part B)")
        else:
            print(f"Testing on: Part {args.part}")
        
        # Load model 
        import tensorflow as tf
        from models.losses import get_loss_functions, get_metrics
        
        model = tf.keras.models.load_model(
            args.model,
            custom_objects={
                'euclidean_loss': get_loss_functions()['density_map'],
                'relative_count_loss': get_loss_functions()['count'],
                'mae_count': get_metrics()['count'][0],
                'mse_count': get_metrics()['count'][1],
                'rmse_count': get_metrics()['count'][2]
            }
        )
        
        # Load test data 
        loader = SimpleDataLoader()
        test_dataset = loader.create_dataset(
            part=args.part,  
            split='test',
            batch_size=1,
            shuffle=False
        )
        
        # Evaluate
        results = model.evaluate(test_dataset, verbose=1)
        
        # Print results
        print("\nTest Results:")
        for name, value in zip(model.metrics_names, results):
            print(f"  {name}: {value:.4f}")
        
        # Display MAE
        mae_value = None
        for i, name in enumerate(model.metrics_names):
            if 'count' in name and 'mae' in name.lower():
                mae_value = results[i]
                break
        
        # Compare with paper targets
        if args.part == 'A':
            target_mae = CONFIG.PAPER_TARGETS['A']['MAE']
        elif args.part == 'B':
            target_mae = CONFIG.PAPER_TARGETS['B']['MAE']
        else:  # mixed
            # Average of A and B targets as estimate
            target_mae = (CONFIG.PAPER_TARGETS['A']['MAE'] + CONFIG.PAPER_TARGETS['B']['MAE']) / 2
            
        if mae_value is not None:
            print(f"\nMAE Comparison:")
            print(f"  Your model: {mae_value:.2f}")
            if args.part == 'mixed':
                print(f"  Estimated target: ~{target_mae:.1f}")
                print(f"  (Part A target: {CONFIG.PAPER_TARGETS['A']['MAE']}, Part B target: {CONFIG.PAPER_TARGETS['B']['MAE']})")
            else:
                print(f"  Paper target: {target_mae:.1f}")
        
    

    elif args.command == 'info':
        if args.dataset:
            print("\nDataset Information")
            print("="*50)
            
            loader = SimpleDataLoader()
            
            
            for part in ['A', 'B', 'mixed']:
                try:
                    info = loader.get_dataset_info(part)
                    if part == 'mixed':
                        print(f"\nPart Mixed (A+B combined):")
                    else:
                        print(f"\nPart {part}:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                except Exception as e:
                    if part == 'mixed':
                        print(f"\nPart Mixed: Not available ({e})")
                    else:
                        print(f"\nPart {part}: Not available ({e})")
        
        if args.model:
            print("\nModel Information")
            print("="*50)
            
            #ORIGINAL model
            model = create_single_scale_model(input_shape=(256, 256, 3))
            model.summary()
            
            from models.single_scale_edge import count_parameters
            param_info = count_parameters(model)
            
            print("\nModel Statistics:")
            print(f"  Total parameters: {param_info['total_params']:,}")
            print(f"  Trainable parameters: {param_info['trainable_params']:,}")
            print(f"  Float32 size: {param_info['float32_size_mb']:.2f} MB")

            

    
    else:
     
        parser.print_help()

        
   


if __name__ == "__main__":
    main()

