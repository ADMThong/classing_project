import gc
import os
import pickle
import threading
from typing import Dict, List, Tuple

import psutil
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


class TruthLieDataset(Dataset):
    """Dataset phân loại Truth/Lie đơn giản"""
    
    def __init__(self, root_dir: str = ".",
                 transform=None, 
                 cache_size: int = 1000,
                 load_all_data: bool = True):
        """
        Args:
            root_dir: Thư mục gốc của project
            transform: Transforms cho ảnh
            cache_size: Số lượng ảnh cache trong memory
            load_all_data: Có load tất cả data từ cả Train và Test folders không
        """
        self.root_dir = root_dir
        self.transform = transform
        self.cache_size = cache_size
        
        # Cache cho images
        self._image_cache = {}
        self._access_count = {}
        self._cache_lock = None
        
        # Label encoder cho Truth/Lie
        self.label_encoder = LabelEncoder()
        
        # Tìm data directories
        self.data_dirs = []
        if load_all_data:
            for folder_name in ['Data\\Train\\Train', 'Data\\Test\\Test']:
                data_dir = os.path.join(root_dir, folder_name)
                if os.path.exists(data_dir):
                    self.data_dirs.append(data_dir)
                    print(f"Found data directory: {data_dir}")
        
        # Cache directory
        self.cache_dir = os.path.join(root_dir, ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Tạo dataset index
        self.dataset_index = self._create_dataset_index()
        
        # Fit label encoder
        self._fit_label_encoder()
        
        print(f"Initialized Truth/Lie dataset with {len(self.dataset_index)} samples")
        self._print_dataset_stats()
        self._print_memory_usage()
    
    def _get_cache_lock(self):
        """Lazy initialization của cache lock"""
        if self._cache_lock is None:
            self._cache_lock = threading.Lock()
        return self._cache_lock
    
    def _create_dataset_index(self) -> List[Dict]:
        """Tạo index cho Truth/Lie dataset"""
        cache_file = os.path.join(self.cache_dir, "truth_lie_dataset_index.pkl")
        
        # Kiểm tra cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    print("Loading Truth/Lie dataset index from cache...")
                    index = pickle.load(f)
                    if len(index) > 0:
                        # Validate cache
                        if self._validate_cache(index):
                            return index
                        else:
                            print("Cache invalid, recreating...")
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        print("Creating Truth/Lie dataset index...")
        dataset_index = []
        
        # Process data directories
        for data_dir in self.data_dirs:
            print(f"Processing: {data_dir}")
            self._scan_directory(data_dir, dataset_index)
        
        print(f"Created index with {len(dataset_index)} samples")
        
        # Save cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(dataset_index, f)
            print("Saved dataset index cache")
        except Exception as e:
            print(f"Error saving cache: {e}")
        
        return dataset_index
    
    def _validate_cache(self, index: List[Dict]) -> bool:
        """Validate cache"""
        if len(index) == 0:
            return False
        
        # Check random samples
        import random
        sample_size = min(10, len(index))
        samples = random.sample(index, sample_size)
        
        for item in samples:
            if 'label' not in item or not os.path.exists(item['path']):
                return False
        return True
    
    def _scan_directory(self, data_dir: str, dataset_index: List[Dict]):
        """Scan directory for Truth/Lie images"""
        # Check for Truth/Lie subdirectories
        truth_dir = os.path.join(data_dir, 'Truth')
        lie_dir = os.path.join(data_dir, 'Lie')
        
        if os.path.exists(truth_dir):
            print(f"  Scanning Truth: {truth_dir}")
            self._scan_label_directory(truth_dir, 'Truth', dataset_index)
        
        if os.path.exists(lie_dir):
            print(f"  Scanning Lie: {lie_dir}")
            self._scan_label_directory(lie_dir, 'Lie', dataset_index)
    
    def _scan_label_directory(self, label_dir: str, label: str, dataset_index: List[Dict]):
        """Scan label directory for images"""
        valid_count = 0
        invalid_count = 0
        
        for root, dirs, files in os.walk(label_dir):
            for file in files:
                if self._is_image_file(file):
                    image_path = os.path.join(root, file)
                    
                    if self._add_image_to_index(image_path, label, dataset_index):
                        valid_count += 1
                    else:
                        invalid_count += 1
        
        print(f"    {label}: {valid_count} valid, {invalid_count} invalid images")
    
    def _add_image_to_index(self, image_path: str, label: str, dataset_index: List[Dict]) -> bool:
        """Add image to index"""
        try:
            # Test if image can be opened
            with Image.open(image_path):
                pass
            
            dataset_index.append({
                'path': os.path.normpath(image_path),
                'label': label,
                'filename': os.path.basename(image_path),
                'size': os.path.getsize(image_path)
            })
            return True
            
        except Exception as e:
            print(f"      Skipping invalid image {image_path}: {e}")
            return False
    
    @staticmethod
    def _is_image_file(filename: str) -> bool:
        """Check if file is image"""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    
    def _fit_label_encoder(self):
        """Fit label encoder"""
        if len(self.dataset_index) == 0:
            return
        
        labels = [item['label'] for item in self.dataset_index]
        self.label_encoder.fit(labels)
        
        print(f"Label encoder fitted:")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
        print(f"  Encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        if len(self.dataset_index) == 0:
            print("No data found!")
            return
        
        print(f"\nTruth/Lie Dataset Statistics:")
        print(f"Total samples: {len(self.dataset_index)}")
        
        # Count by label
        label_counts = {}
        for item in self.dataset_index:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(self.dataset_index)) * 100
            print(f"  {label}: {count} samples ({percentage:.1f}%)")
        
        # Check balance
        if len(label_counts) == 2:
            counts = list(label_counts.values())
            ratio = max(counts) / min(counts)
            if ratio > 1.5:
                print(f"  ⚠️  Dataset imbalanced (ratio: {ratio:.2f})")
            else:
                print(f"  ✅ Dataset balanced (ratio: {ratio:.2f})")
    
    def _load_image_lazy(self, image_path: str) -> Image.Image:
        """Lazy loading with caching"""
        cache_lock = self._get_cache_lock()
        
        # Check cache first
        with cache_lock:
            if image_path in self._image_cache:
                self._access_count[image_path] = self._access_count.get(image_path, 0) + 1
                return self._image_cache[image_path].copy()
        
        # Load image
        try:
            img = Image.open(image_path).convert('RGB')
            
            # Add to cache
            with cache_lock:
                if len(self._image_cache) < self.cache_size:
                    self._image_cache[image_path] = img.copy()
                    self._access_count[image_path] = 1
                else:
                    # LRU eviction
                    if self._access_count:
                        least_used = min(self._access_count.items(), key=lambda x: x[1])
                        del self._image_cache[least_used[0]]
                        del self._access_count[least_used[0]]
                    
                    self._image_cache[image_path] = img.copy()
                    self._access_count[image_path] = 1
            
            return img
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            return Image.new('RGB', (560, 560), color='black')
    
    def _print_memory_usage(self):
        """Print memory usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage: {memory_mb:.2f} MB")
            print(f"Cache size: {len(self._image_cache)} images")
        except Exception:
            pass
    
    def clear_cache(self):
        """Clear cache"""
        cache_lock = self._get_cache_lock()
        with cache_lock:
            self._image_cache.clear()
            self._access_count.clear()
        gc.collect()
        print("Cache cleared")
    
    def __len__(self) -> int:
        return len(self.dataset_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item with Truth/Lie label"""
        if idx >= len(self.dataset_index):
            raise IndexError(f"Index {idx} out of range")
        
        item = self.dataset_index[idx]
        
        # Load image
        image = self._load_image_lazy(item['path'])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = transforms.Compose([
                transforms.Resize((560, 560)),
                transforms.ToTensor()
            ])(image)
        
        # Get encoded label
        label = self.label_encoder.transform([item['label']])[0]
        
        return image, label
    
    def get_item_with_metadata(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get item with metadata"""
        if idx >= len(self.dataset_index):
            raise IndexError(f"Index {idx} out of range")
        
        item = self.dataset_index[idx]
        image = self._load_image_lazy(item['path'])
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose([
                transforms.Resize((560, 560)),
                transforms.ToTensor()
            ])(image)
        
        metadata = {
            'label': item['label'],
            'label_encoded': self.label_encoder.transform([item['label']])[0],
            'path': item['path'],
            'filename': item['filename'],
            'size': item['size']
        }
        
        return image, metadata
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return len(self.label_encoder.classes_)
    
    def get_class_names(self) -> List[str]:
        """Get class names"""
        return list(self.label_encoder.classes_)
    
    def get_dataset_info(self) -> Dict:
        """Get dataset info"""
        if len(self.dataset_index) == 0:
            return {
                'total_samples': 0,
                'num_classes': 0,
                'class_names': [],
                'memory_usage_mb': 0
            }
        
        total_size = sum(item['size'] for item in self.dataset_index)
        
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        except:
            memory_usage = 0
        
        return {
            'total_samples': len(self.dataset_index),
            'total_size_mb': total_size / (1024 * 1024),
            'num_classes': self.get_num_classes(),
            'class_names': self.get_class_names(),
            'cache_size': len(self._image_cache),
            'memory_usage_mb': memory_usage
        }

def get_transforms(augment: bool = True):
    """Get transforms for Truth/Lie dataset"""
    if augment:
        return transforms.Compose([
            transforms.Resize((560, 560)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((560, 560)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(root_dir: str = ".",
                       test_size: float = 0.2,
                       batch_size: int = 16,
                       num_workers: int = 0,
                       cache_size: int = 500,
                       random_state: int = 42):
    """Create train/test data loaders for Truth/Lie classification"""
    
    print("Creating Truth/Lie dataset...")
    
    # GPU optimization for RTX 3050 Ti
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {gpu_memory:.1f}GB")
        
        if gpu_memory <= 4.5:  # RTX 3050 Ti optimization
            optimal_batch_size = min(batch_size, 8)
            optimal_num_workers = min(4, os.cpu_count() // 2)
            cache_size = min(cache_size, 200)
            print(f"RTX 3050 Ti optimization:")
            print(f"  Batch size: {batch_size} -> {optimal_batch_size}")
            print(f"  Num workers: {num_workers} -> {optimal_num_workers}")
            batch_size = optimal_batch_size
            num_workers = optimal_num_workers
    
    # Create datasets
    train_dataset = TruthLieDataset(
        root_dir=root_dir,
        transform=get_transforms(augment=True),
        cache_size=cache_size
    )
    
    test_dataset = TruthLieDataset(
        root_dir=root_dir,
        transform=get_transforms(augment=False),
        cache_size=cache_size // 2
    )
    
    if len(train_dataset) == 0:
        raise ValueError("No data found in dataset!")
    
    # Split indices
    indices = list(range(len(train_dataset)))
    labels = [train_dataset.dataset_index[i]['label'] for i in indices]
    
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Maintain label balance
    )
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")
    print(f"Class names: {train_dataset.get_class_names()}")
    
    return train_loader, test_loader, train_dataset, (train_indices, test_indices)

def create_train_val_test_loaders(root_dir: str = ".",
                                  train_ratio: float = 0.6,
                                  val_ratio: float = 0.2,
                                  test_ratio: float = 0.2,
                                  batch_size: int = 16,
                                  num_workers: int = 0,
                                  cache_size: int = 500,
                                  random_state: int = 42):
    """Create train/validation/test data loaders"""
    
    # Check ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Ratios must sum to 1.0")
    
    print(f"Creating train/val/test split: {train_ratio}/{val_ratio}/{test_ratio}")
    
    # Create dataset
    dataset = TruthLieDataset(
        root_dir=root_dir,
        transform=get_transforms(augment=True),
        cache_size=cache_size
    )
    
    if len(dataset) == 0:
        raise ValueError("No data found!")
    
    # Get labels for stratification
    labels = [dataset.dataset_index[i]['label'] for i in range(len(dataset))]
    
    # Split train and temp (val + test)
    train_indices, temp_indices = train_test_split(
        range(len(dataset)),
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=labels
    )
    
    # Split temp into val and test
    temp_labels = [labels[i] for i in temp_indices]
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_state,
        stratify=temp_labels
    )
    
    print(f"Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Create subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    split_info = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices)
    }
    
    return train_loader, val_loader, test_loader, dataset, split_info

# Test script
if __name__ == "__main__":
    print("Testing Truth/Lie dataset...")
    
    try:
        # Create dataset
        dataset = TruthLieDataset(cache_size=10)
        
        info = dataset.get_dataset_info()
        print(f"\nDataset info: {info}")
        
        if info['total_samples'] > 0:
            # Add visualization for class distribution
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get label counts for visualization
            label_counts = {}
            for item in dataset.dataset_index:
                label = item['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # Create visualization
            if len(label_counts) > 0:
                labels = list(label_counts.keys())
                counts = list(label_counts.values())
                percentages = [(count / sum(counts)) * 100 for count in counts]
                
                # Create figure with subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Pie chart
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                wedges, texts, autotexts = ax1.pie(counts, labels=labels, autopct='%1.1f%%', 
                                                  startangle=90, colors=colors[:len(labels)])
                ax1.set_title('Class Distribution - Pie Chart\n(Truth vs Lie)', fontsize=14, fontweight='bold')
                
                # Make percentage text bold and larger
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(12)
                
                # Bar chart
                bars = ax2.bar(labels, counts, color=colors[:len(labels)], alpha=0.8, edgecolor='black', linewidth=1.5)
                ax2.set_title('Class Distribution - Bar Chart\n(Sample Counts)', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Classes', fontsize=12)
                ax2.set_ylabel('Number of Samples', fontsize=12)
                
                # Add value labels on bars
                for i, (bar, count, percentage) in enumerate(zip(bars, counts, percentages)):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                            f'{count}\n({percentage:.1f}%)', 
                            ha='center', va='bottom', fontweight='bold', fontsize=11)
                
                # Improve bar chart appearance
                ax2.set_ylim(0, max(counts) * 1.15)
                ax2.grid(axis='y', alpha=0.3, linestyle='--')
                ax2.set_axisbelow(True)
                
                # Add overall statistics text
                total_samples = sum(counts)
                stats_text = f"Total Samples: {total_samples}\n"
                for label, count, percentage in zip(labels, counts, percentages):
                    stats_text += f"{label}: {count} ({percentage:.1f}%)\n"
                
                # Add balance analysis
                if len(counts) == 2:
                    ratio = max(counts) / min(counts)
                    balance_status = "Balanced ✅" if ratio <= 1.5 else f"Imbalanced ⚠️ (ratio: {ratio:.2f})"
                    stats_text += f"\nBalance: {balance_status}"
                
                fig.suptitle(f'Truth/Lie Dataset Analysis\n{stats_text}', 
                           fontsize=16, fontweight='bold', y=0.95)
                
                plt.tight_layout()
                
                # Save the plot
                plot_path = os.path.join(dataset.root_dir, 'dataset_distribution.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"\n📊 Class distribution chart saved to: {plot_path}")
                
                # Show the plot
                try:
                    plt.show()
                except:
                    print("Display not available, chart saved to file only.")
                
                plt.close()
                
                # Print detailed statistics
                print(f"\n{'='*50}")
                print(f"DETAILED CLASS DISTRIBUTION ANALYSIS")
                print(f"{'='*50}")
                print(f"Total Samples: {total_samples}")
                print(f"Number of Classes: {len(labels)}")
                print("-" * 30)
                
                for i, (label, count, percentage) in enumerate(zip(labels, counts, percentages)):
                    print(f"{i+1}. {label}:")
                    print(f"   Samples: {count}")
                    print(f"   Percentage: {percentage:.2f}%")
                    print(f"   Bar: {'█' * int(percentage/2)}")
                    print()
                
                if len(counts) == 2:
                    ratio = max(counts) / min(counts)
                    majority_class = labels[counts.index(max(counts))]
                    minority_class = labels[counts.index(min(counts))]
                    
                    print(f"Balance Analysis:")
                    print(f"  Majority class: {majority_class} ({max(counts)} samples)")
                    print(f"  Minority class: {minority_class} ({min(counts)} samples)")
                    print(f"  Imbalance ratio: {ratio:.3f}")
                    
                    if ratio <= 1.2:
                        print(f"  Status: Excellent balance ✅")
                    elif ratio <= 1.5:
                        print(f"  Status: Good balance ✅")
                    elif ratio <= 2.0:
                        print(f"  Status: Moderate imbalance ⚠️")
                    else:
                        print(f"  Status: High imbalance ❌")
                        print(f"  Recommendation: Consider data augmentation or resampling")
                
                print(f"{'='*50}")
            
            # Test sample
            image, label = dataset[0]
            print(f"Sample - Image: {image.shape}, Label: {label}")
            
            # Test with metadata
            image_meta, metadata = dataset.get_item_with_metadata(0)
            print(f"Metadata: {metadata}")
            
            # Test data loaders
            train_loader, test_loader, _, _ = create_data_loaders(
                batch_size=4,
                cache_size=10,
                test_size=0.2,
                num_workers=0
            )
            
            print(f"Train batches: {len(train_loader)}")
            print(f"Test batches: {len(test_loader)}")
            
            # Test batch
            for batch_images, batch_labels in train_loader:
                print(f"Batch - Images: {batch_images.shape}, Labels: {batch_labels.shape}")
                print(f"Label values: {batch_labels.numpy()}")
                break
        
        dataset.clear_cache()
        print("✅ Test completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()