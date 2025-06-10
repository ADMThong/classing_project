import os
import time
from typing import Dict, List  # Thêm tất cả typing imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, \
    roc_auc_score  # Thêm sklearn metrics


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block để tăng cường feature representation"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)

class ResidualBlock(nn.Module):
    """Residual Block với SE attention"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE Block
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        out += residual
        out = F.relu(out)
        
        return out

class MicroExpressionCNN(nn.Module):
    """Custom CNN cho Truth/Lie Classification với SE-ResNet architecture"""
    
    def __init__(self, 
                 num_classes: int = 2,  # Simplified to single int for binary classification
                 dropout_rate: float = 0.5,
                 use_se: bool = True,
                 lite_mode: bool = False):
        """
        Args:
            num_classes: Số lượng classes (mặc định 2 cho Truth/Lie)
            dropout_rate: Dropout rate
            use_se: Có sử dụng SE blocks không
            lite_mode: Chế độ tối ưu cho RTX 3050 Ti
        """
        super(MicroExpressionCNN, self).__init__()
        
        self.num_classes = num_classes
        self.use_se = use_se
        self.lite_mode = lite_mode
        
        # Initial conv layer - giảm channels nếu lite mode
        initial_channels = 32 if lite_mode else 64
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers - điều chỉnh cho lite mode
        if lite_mode:
            self.layer1 = self._make_layer(initial_channels, 64, 2, 1)
            self.layer2 = self._make_layer(64, 128, 2, 2)
            self.layer3 = self._make_layer(128, 256, 2, 2)
            self.layer4 = self._make_layer(256, 512, 2, 2)
            final_features = 512
        else:
            self.layer1 = self._make_layer(initial_channels, 128, 3, 1)
            self.layer2 = self._make_layer(128, 256, 4, 2)
            self.layer3 = self._make_layer(256, 512, 6, 2)
            self.layer4 = self._make_layer(512, 1024, 3, 2)
            final_features = 1024
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Single classification head for Truth/Lie
        self.fc = nn.Linear(final_features, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, self.use_se))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, self.use_se))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        # Classification for Truth/Lie
        return self.fc(x)

class EfficientNetMicroExpression(nn.Module):
    """EfficientNet-based model cho Truth/Lie Classification"""
    
    def __init__(self, 
                 num_classes: int = 2,  # Simplified to single int
                 efficientnet_version: str = "b3",
                 dropout_rate: float = 0.5,
                 pretrained: bool = True,
                 lite_mode: bool = False):
        """
        Args:
            num_classes: Số lượng classes (mặc định 2 cho Truth/Lie)
            efficientnet_version: "b0", "b1", "b2", "b3", "b4"
            dropout_rate: Dropout rate
            pretrained: Sử dụng pretrained weights
            lite_mode: Chế độ tối ưu cho RTX 3050 Ti
        """
        super(EfficientNetMicroExpression, self).__init__()
        
        self.num_classes = num_classes
        
        # Tự động chọn version phù hợp cho RTX 3050 Ti
        if lite_mode and efficientnet_version in ["b3", "b4"]:
            efficientnet_version = "b0"
            print(f"RTX 3050 Ti optimization: Using EfficientNet-{efficientnet_version}")
        
        # Load EfficientNet backbone
        if efficientnet_version == "b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        elif efficientnet_version == "b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            feature_dim = 1280
        elif efficientnet_version == "b2":
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
            feature_dim = 1408
        elif efficientnet_version == "b3":
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = 1536
        elif efficientnet_version == "b4":
            self.backbone = models.efficientnet_b4(pretrained=pretrained)
            feature_dim = 1792
        else:
            raise ValueError(f"Unsupported EfficientNet version: {efficientnet_version}")
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add simplified attention mechanism cho lite mode
        if lite_mode:
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 8),  # Giảm complexity
                nn.ReLU(),
                nn.Linear(feature_dim // 8, feature_dim),
                nn.Sigmoid()
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, feature_dim),
                nn.Sigmoid()
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Single classification head for Truth/Lie
        self.fc = nn.Linear(feature_dim, num_classes)
        
        # Initialize new layers
        self._initialize_new_layers()
    
    def _initialize_new_layers(self):
        for module in [self.attention, self.fc]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, 0, 0.01)
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        
        # Attention mechanism
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        features = self.dropout(features)
        
        # Classification for Truth/Lie
        return self.fc(features)

class ModelTrainer:
    """Trainer class cho việc huấn luyện và đánh giá models"""
    
    def __init__(self, model: nn.Module, device: torch.device, save_dir: str = "results"):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Khởi tạo mặc định cho gradient accumulation
        self.use_gradient_accumulation = False
        self.accumulation_steps = 1
        
        # Tối ưu hóa cho RTX 3050 Ti
        if torch.cuda.is_available() and device.type == 'cuda':
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_memory_gb = gpu_props.total_memory / (1024**3)
                print(f"GPU detected: {gpu_props.name}")
                print(f"VRAM: {gpu_memory_gb:.1f}GB")
                
                # Enable optimizations cho RTX 3050 Ti
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Cài đặt gradient accumulation cho RTX 3050 Ti
                if gpu_memory_gb <= 4.5:
                    self.use_gradient_accumulation = True
                    self.accumulation_steps = 4  # Tích lũy gradient qua 4 steps
                    print(f"Enabled gradient accumulation: {self.accumulation_steps} steps")
                else:
                    self.use_gradient_accumulation = False
                    self.accumulation_steps = 1
                    
            except Exception as e:
                print(f"Warning: GPU optimization setup failed: {e}")
                self.use_gradient_accumulation = False
                self.accumulation_steps = 1
        else:
            print("Using CPU training mode")
            self.use_gradient_accumulation = False
            self.accumulation_steps = 1
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch: int):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Initialize gradient scaler cho mixed precision (chỉ khi có GPU)
        if self.device.type == 'cuda':
            if not hasattr(self, 'scaler'):
                self.scaler = torch.cuda.amp.GradScaler()
        
        optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Transfer to device
            inputs = inputs.to(self.device, non_blocking=True if self.device.type == 'cuda' else False)
            targets = targets.to(self.device, non_blocking=True if self.device.type == 'cuda' else False)
            
            # Forward pass với hoặc không có mixed precision
            if self.device.type == 'cuda':
                # Mixed precision training cho GPU
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == targets.data)
                    
                    # Scale loss for gradient accumulation
                    if self.use_gradient_accumulation:
                        loss = loss / self.accumulation_steps
                
                # Backward pass với gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                # CPU training (không có mixed precision)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == targets.data)
                
                # Scale loss for gradient accumulation
                if self.use_gradient_accumulation:
                    loss = loss / self.accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            running_loss += loss.item() * inputs.size(0) * self.accumulation_steps
            total_samples += inputs.size(0)
            
            # Memory cleanup và progress reporting
            if batch_idx % 20 == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                    print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, GPU: {gpu_memory:.2f}GB')
                else:
                    print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Handle remaining gradients
        if self.use_gradient_accumulation and (len(train_loader) % self.accumulation_steps != 0):
            if self.device.type == 'cuda':
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def validate_epoch(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True if self.device.type == 'cuda' else False)
                targets = targets.to(self.device, non_blocking=True if self.device.type == 'cuda' else False)
                
                # Forward pass với hoặc không có mixed precision
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        
                        _, preds = torch.max(outputs, 1)
                        running_corrects += torch.sum(preds == targets.data)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == targets.data)
                
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def train(self, train_loader, val_loader, num_epochs: int, learning_rate: float = 0.001,
              weight_decay: float = 1e-4, step_size: int = 10, gamma: float = 0.1):
        
        # Điều chỉnh learning rate cho RTX 3050 Ti với gradient accumulation
        if hasattr(self, 'use_gradient_accumulation') and self.use_gradient_accumulation:
            effective_batch_size = train_loader.batch_size * self.accumulation_steps
            lr_scale = effective_batch_size / 32  # Base batch size = 32
            learning_rate = learning_rate * lr_scale
            print(f"Adjusted learning rate for gradient accumulation: {learning_rate:.6f}")
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        # Setup criterion - single criterion for binary classification
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)
        
        best_acc = 0.0
        best_model_wts = self.model.state_dict().copy()
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Task: Binary Classification (Truth/Lie)")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch+1)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = self.model.state_dict().copy()
                self.save_model(f'best_model_{self.model.__class__.__name__}.pth')
        
        training_time = time.time() - start_time
        print(f'\nTraining complete in {training_time//60:.0f}m {training_time%60:.0f}s')
        print(f'Best Val Acc: {best_acc:.4f}')
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        return self.history
    
    def save_model(self, filename: str):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, filename))
        print(f"Model saved to {os.path.join(self.save_dir, filename)}")

class ModelEvaluator:
    """Evaluator class cho việc đánh giá models và tạo visualizations"""
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str], save_dir: str = "results"):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def evaluate(self, test_loader) -> Dict:
        """Đánh giá model trên test set"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        
        # Classification report
        report = classification_report(all_targets, all_preds, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': np.array(all_probs)
        }
        
        return results
    
    def plot_training_history(self, history: Dict, model_name: str):
        """Vẽ training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} Training History', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', marker='s')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', marker='o')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(history['learning_rates'], marker='o')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss difference plot
        loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[1, 1].plot(loss_diff, marker='o', color='red')
        axes[1, 1].set_title('Overfitting Monitor (Val Loss - Train Loss)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{model_name}_training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Vẽ confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.save_dir, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_report(self, report: Dict, model_name: str):
        """Vẽ classification report"""
        # Exclude 'accuracy', 'macro avg', 'weighted avg' for per-class metrics
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        metrics = ['precision', 'recall', 'f1-score']
        data = []
        
        for cls in classes:
            for metric in metrics:
                data.append({
                    'Class': cls,
                    'Metric': metric,
                    'Value': report[cls][metric]
                })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Class', y='Value', hue='Metric')
        plt.title(f'{model_name} - Classification Metrics by Class')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{model_name}_classification_report.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, targets: np.ndarray, probabilities: np.ndarray, model_name: str):
        """Vẽ ROC curve (for binary classification)"""
        if len(self.class_names) == 2:
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            auc = roc_auc_score(targets, probabilities[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} - ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, f'{model_name}_roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.show()

def compare_models(results1: Dict, results2: Dict, model1_name: str, model2_name: str, save_dir: str = "results"):
    """So sánh hai models"""
    
    # Comparison metrics
    metrics_comparison = {
        'Model': [model1_name, model2_name],
        'Accuracy': [results1['accuracy'], results2['accuracy']],
        'Precision (Macro)': [results1['classification_report']['macro avg']['precision'],
                             results2['classification_report']['macro avg']['precision']],
        'Recall (Macro)': [results1['classification_report']['macro avg']['recall'],
                          results2['classification_report']['macro avg']['recall']],
        'F1-Score (Macro)': [results1['classification_report']['macro avg']['f1-score'],
                            results2['classification_report']['macro avg']['f1-score']]
    }
    
    df_comparison = pd.DataFrame(metrics_comparison)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot comparison
    metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
    x = np.arange(len(metrics))
    width = 0.35
    
    values1 = [df_comparison[metric][0] for metric in metrics]
    values2 = [df_comparison[metric][1] for metric in metrics]
    
    axes[0].bar(x - width/2, values1, width, label=model1_name, alpha=0.8)
    axes[0].bar(x + width/2, values2, width, label=model2_name, alpha=0.8)
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    values1 += values1[:1]
    values2 += values2[:1]
    
    axes[1] = plt.subplot(1, 2, 2, projection='polar')
    axes[1].plot(angles, values1, 'o-', linewidth=2, label=model1_name)
    axes[1].fill(angles, values1, alpha=0.25)
    axes[1].plot(angles, values2, 'o-', linewidth=2, label=model2_name)
    axes[1].fill(angles, values2, alpha=0.25)
    axes[1].set_xticks(angles[:-1])
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Performance Radar Chart')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'models_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save comparison table
    print("\nModel Comparison Summary:")
    print(df_comparison.to_string(index=False))
    df_comparison.to_csv(os.path.join(save_dir, 'models_comparison.csv'), index=False)
    
    return df_comparison

def create_model(model_type: str = "custom_cnn",
                num_classes: int = 2,  # Simplified to single int
                **kwargs) -> nn.Module:
    """
    Factory function để tạo models - tối ưu cho RTX 3050 Ti
    """
    
    # Tự động enable lite mode cho RTX 3050 Ti
    lite_mode = False
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb <= 4.5:  # RTX 3050 Ti
            lite_mode = True
            print("RTX 3050 Ti detected - Enabling lite mode for 560x560 images")
    
    if model_type == "custom_cnn":
        return MicroExpressionCNN(
            num_classes=num_classes,
            dropout_rate=kwargs.get('dropout_rate', 0.3),  # Giảm dropout cho lite mode
            use_se=kwargs.get('use_se', True),
            lite_mode=lite_mode
        )
    
    elif model_type == "efficientnet":
        return EfficientNetMicroExpression(
            num_classes=num_classes,
            efficientnet_version=kwargs.get('efficientnet_version', 'b3'),
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            pretrained=kwargs.get('pretrained', True),
            lite_mode=lite_mode
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Đếm số parameters của model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

# Test models
if __name__ == "__main__":
    print("Testing models for Truth/Lie classification...")
    
    # Test single task models
    print("\n" + "="*50)
    print("Testing Truth/Lie Binary Classification Models")
    print("="*50)
    
    for model_type in ["custom_cnn", "efficientnet"]:
        print(f"\nTesting {model_type}:")
        
        # Create model for binary classification
        model = create_model(
            model_type=model_type,
            num_classes=2  # Truth and Lie
        )
        
        # Test forward pass
        dummy_input = torch.randn(4, 3, 560, 560)
        
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            print("  Using GPU")
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected output shape: [4, 2] for binary classification")
        
        # Count parameters
        params = count_parameters(model)
        print(f"  Parameters: {params['trainable_parameters']:,}")
    
    print("\nModel testing completed!")
    print("✅ Models are ready for Truth/Lie binary classification!")