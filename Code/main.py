import os
import sys
import warnings

import torch

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import create_train_val_test_loaders
from model import (create_model, ModelTrainer, ModelEvaluator, compare_models, count_parameters)

def main():
    """Main function để chạy training và evaluation"""
    print("="*80)
    print("TRUTH/LIE CLASSIFICATION PROJECT")
    print("="*80)
    
    # Configuration - Tối ưu cho RTX 3050 Ti với image size 560x560
    CONFIG = {
        'root_dir': '.',
        'batch_size': 16,  # Sẽ được điều chỉnh tự động
        'num_epochs': 30,  # Giảm epochs do training chậm hơn với image size lớn
        'learning_rate': 0.0005,  # Giảm learning rate cho stability
        'weight_decay': 1e-4,
        'num_workers': 2,  # Change from 0 to 2 for better performance
        'cache_size': 500,  # Sẽ được điều chỉnh tự động
        'test_size': 0.2,
        'random_state': 42,
        'save_dir': 'results'
    }
    
    # Create results directory
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # Check device và tối ưu hóa cho RTX 3050 Ti
    print("\n=== DEVICE DETECTION ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            
            print(f"GPU: {gpu_props.name}")
            print(f"VRAM: {gpu_memory_gb:.1f}GB")
            print(f"Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            
            # Test GPU accessibility
            test_tensor = torch.randn(1).cuda()
            print("GPU test successful - using GPU")
            device = torch.device('cuda')
            
            # Tối ưu hóa đặc biệt cho RTX 3050 Ti với image size 560x560
            if gpu_memory_gb <= 4.5:
                print("\n=== RTX 3050 Ti OPTIMIZATION (560x560 images) ===")
                CONFIG['batch_size'] = 6  # Batch size nhỏ do image size lớn
                CONFIG['cache_size'] = 200  # Giảm cache size
                CONFIG['num_epochs'] = 10  # Điều chỉnh epochs
                CONFIG['num_workers'] = min(2, os.cpu_count() // 2)  # Optimize num_workers
                print(f"Optimized batch_size: {CONFIG['batch_size']}")
                print(f"Optimized cache_size: {CONFIG['cache_size']}")
                print(f"Optimized epochs: {CONFIG['num_epochs']}")
                print(f"Optimized num_workers: {CONFIG['num_workers']}")
        
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction để tránh OOM
            torch.cuda.set_per_process_memory_fraction(0.85)  # Giảm từ 0.9 xuống 0.85
            print("GPU optimizations enabled")
            
        except Exception as e:
            print(f"GPU test failed: {e}")
            print("Falling back to CPU")
            device = torch.device('cpu')
    else:
        print("CUDA not available - using CPU")
        device = torch.device('cpu')
        # Điều chỉnh config cho CPU
        CONFIG['batch_size'] = min(CONFIG['batch_size'], 8)
        CONFIG['num_workers'] = min(2, os.cpu_count() // 2)
    
    print(f"Final device: {device}")
    
    try:
        # Step 1: Load and prepare data
        print("\n" + "="*50)
        print("STEP 1: LOADING AND PREPARING DATA")
        print("="*50)
        
        train_loader, val_loader, test_loader, dataset, split_info = create_train_val_test_loaders(
            root_dir=CONFIG['root_dir'],
            train_ratio=0.6,    # 60% train
            val_ratio=0.2,      # 20% validation
            test_ratio=0.2,     # 20% test
            batch_size=CONFIG['batch_size'],
            num_workers=CONFIG['num_workers'],
            cache_size=CONFIG['cache_size'],
            random_state=CONFIG['random_state']
        )
        
        print(f"Dataset split successfully!")
        print(f"Train: {split_info['train_size']} samples ({len(train_loader)} batches)")
        print(f"Validation: {split_info['val_size']} samples ({len(val_loader)} batches)")
        print(f"Test: {split_info['test_size']} samples ({len(test_loader)} batches)")
        
        # Get dataset info
        num_classes = dataset.get_num_classes()
        class_names = dataset.get_class_names()
        
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")
        
        # Step 2: Create models với cấu hình tối ưu cho RTX 3050 Ti
        print("\n" + "="*50)
        print("STEP 2: CREATING MODELS")
        print("="*50)
        
        models_config = [
            {
                'name': 'SE-ResNet_CNN_Lite',
                'type': 'custom_cnn',
                'params': {
                    'dropout_rate': 0.3,
                    'use_se': True
                }
            },
            {
                'name': 'EfficientNet_B0_Optimized',  # B0 cho RTX 3050 Ti
                'type': 'efficientnet',
                'params': {
                    'efficientnet_version': 'b0',
                    'dropout_rate': 0.3,
                    'pretrained': True
                }
            }
        ]
        
        models = {}
        trainers = {}
        
        for model_config in models_config:
            print(f"\nCreating {model_config['name']}...")
            
            model = create_model(
                model_type=model_config['type'],
                num_classes=num_classes,  # Chỉ truyền num_classes
                **model_config['params']
            )
            
            model = model.to(device)
            
            # Count parameters
            params = count_parameters(model)
            print(f"  Total parameters: {params['total_parameters']:,}")
            print(f"  Trainable parameters: {params['trainable_parameters']:,}")
            
            models[model_config['name']] = model
            trainers[model_config['name']] = ModelTrainer(model, device, CONFIG['save_dir'])
        
        # Step 3: Train models
        print("\n" + "="*50)
        print("STEP 3: TRAINING MODELS")
        print("="*50)
        
        training_histories = {}
        
        for model_name, trainer in trainers.items():
            print(f"\n{'='*30}")
            print(f"Training {model_name}")
            print(f"{'='*30}")
            
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=CONFIG['num_epochs'],
                learning_rate=CONFIG['learning_rate'],
                weight_decay=CONFIG['weight_decay'],
                step_size=15,
                gamma=0.1
            )
            
            training_histories[model_name] = history
            print(f"Completed training {model_name}")
        
        # Step 4: Evaluate models
        print("\n" + "="*50)
        print("STEP 4: EVALUATING MODELS")
        print("="*50)
        
        results = {}
        evaluators = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            evaluator = ModelEvaluator(model, device, class_names, CONFIG['save_dir'])
            result = evaluator.evaluate(test_loader)
            
            results[model_name] = result
            evaluators[model_name] = evaluator
            
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Precision (macro): {result['classification_report']['macro avg']['precision']:.4f}")
            print(f"  Recall (macro): {result['classification_report']['macro avg']['recall']:.4f}")
            print(f"  F1-score (macro): {result['classification_report']['macro avg']['f1-score']:.4f}")
        
        # Step 5: Create visualizations
        print("\n" + "="*50)
        print("STEP 5: CREATING VISUALIZATIONS")
        print("="*50)
        
        for model_name, evaluator in evaluators.items():
            print(f"\nCreating visualizations for {model_name}...")
            
            # Training history plots
            evaluator.plot_training_history(training_histories[model_name], model_name)
            
            # Confusion matrix
            evaluator.plot_confusion_matrix(results[model_name]['confusion_matrix'], model_name)
            
            # Classification report
            evaluator.plot_classification_report(results[model_name]['classification_report'], model_name)
            
            # ROC curve (for binary classification)
            if len(class_names) == 2:
                evaluator.plot_roc_curve(
                    results[model_name]['targets'],
                    results[model_name]['probabilities'],
                    model_name
                )
        
        # Step 6: Compare models
        print("\n" + "="*50)
        print("STEP 6: COMPARING MODELS")
        print("="*50)
        
        model_names = list(models.keys())
        if len(model_names) >= 2:
            comparison_df = compare_models(
                results[model_names[0]], 
                results[model_names[1]],
                model_names[0], 
                model_names[1],
                CONFIG['save_dir']
            )
        
        # Step 7: Save detailed results
        print("\n" + "="*50)
        print("STEP 7: SAVING DETAILED RESULTS")
        print("="*50)
        
        import json
        
        # Save configuration
        with open(os.path.join(CONFIG['save_dir'], 'config.json'), 'w') as f:
            json.dump(CONFIG, f, indent=4)
        
        # Save results summary
        results_summary = {}
        for model_name, result in results.items():
            results_summary[model_name] = {
                'accuracy': float(result['accuracy']),
                'precision_macro': float(result['classification_report']['macro avg']['precision']),
                'recall_macro': float(result['classification_report']['macro avg']['recall']),
                'f1_macro': float(result['classification_report']['macro avg']['f1-score']),
                'parameters': count_parameters(models[model_name])
            }
        
        with open(os.path.join(CONFIG['save_dir'], 'results_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        print("All results saved successfully!")
        print(f"Results directory: {CONFIG['save_dir']}")
        
        # Final summary
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        
        print("\nModel Performance Summary:")
        for model_name, summary in results_summary.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {summary['accuracy']:.4f}")
            print(f"  F1-Score (Macro): {summary['f1_macro']:.4f}")
            print(f"  Parameters: {summary['parameters']['trainable_parameters']:,}")
        
        # Determine best model
        best_model = max(results_summary.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest performing model: {best_model[0]}")
        print(f"Best accuracy: {best_model[1]['accuracy']:.4f}")
        
        print("\n" + "="*80)
        print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # GPU cleanup mạnh mẽ hơn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"GPU memory cleared and synchronized")
        
        # Cleanup
        if 'dataset' in locals():
            dataset.clear_cache()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("\nCleanup completed.")

if __name__ == "__main__":
    main()