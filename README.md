# Truth/Lie Classification Project - Micro-Expression Analysis

## 📖 Mô tả dự án
Dự án phân loại hình ảnh sử dụng Deep Learning để phân biệt giữa Truth (Sự thật) và Lie (Dối trá) dựa trên micro-expression (vi biểu cảm khuôn mặt). Dự án sử dụng hai kiến trúc model chính: Custom SE-ResNet CNN và EfficientNet được tối ưu hóa cho RTX 3050 Ti.

## 🗂️ Cấu trúc dự án
```
ClassingProject/
├── Code/
│   ├── dataset.py          # Dataset loader và preprocessing cho Truth/Lie
│   ├── model.py           # Định nghĩa SE-ResNet CNN và EfficientNet models
│   ├── main.py            # Main training và evaluation script
│   └── __pycache__/       # Python cache files
├── Data/
│   ├── Demo.png           # Demo image
│   ├── Directory.docx     # Hướng dẫn cấu trúc thư mục
│   ├── README File.txt    # Thông tin về dataset micro-expression
│   ├── Metadata/          # Metadata files
│   ├── Train/
│   │   ├── Truth/         # Training images - Truth class
│   │   └── Lie/           # Training images - Lie class
│   └── Test/
│       ├── Truth/         # Test images - Truth class
│       └── Lie/           # Test images - Lie class
├── results/               # Training results, models và visualizations
│   ├── best_model_*.pth   # Saved model weights
│   ├── config.json        # Training configuration
│   ├── *_training_history.png
│   ├── *_confusion_matrix.png
│   ├── *_classification_report.png
│   └── models_comparison.png
├── dataset_distribution.png # Class distribution visualization
└── README.md
```

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- CUDA 11.0+ (optional, cho GPU training)
- RTX 3050 Ti hoặc GPU tương đương (khuyến nghị)
- RAM: 8GB+ (16GB khuyến nghị)

### Cài đặt dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow numpy pandas matplotlib seaborn scikit-learn psutil
```

## 🎯 Đặc điểm dự án

### Dataset Micro-Expression
- **Kích thước ảnh**: 560x560px
- **Format**: PNG
- **Nguồn**: Được chụp bằng Nothing Phone (2) với camera 50MP
- **Đặc điểm**: Tập trung vào micro-expression để phân biệt Truth/Lie
- **Cân bằng**: Dataset được thiết kế cân bằng giữa 2 classes

### Model Architectures
1. **SE-ResNet CNN Custom**:
   - Squeeze-and-Excitation blocks
   - Residual connections
   - Tối ưu cho RTX 3050 Ti (lite mode)
   - Gradient accumulation support

2. **EfficientNet-B0 Optimized**:
   - Pre-trained EfficientNet backbone
   - Custom attention mechanism
   - Mixed precision training
   - Memory-efficient cho RTX 3050 Ti

## 🚀 Sử dụng

### 1. Chuẩn bị dữ liệu
Đặt hình ảnh micro-expression vào thư mục tương ứng:
```
Data/
├── Train/
│   ├── Truth/    # Hình ảnh Truth (nói thật) cho training
│   └── Lie/      # Hình ảnh Lie (nói dối) cho training
└── Test/
    ├── Truth/    # Hình ảnh Truth cho training
    └── Lie/      # Hình ảnh Lie cho training
```

### 2. Chạy training và evaluation
```bash
cd Code
python main.py
```

### 3. Tùy chỉnh cấu hình (trong main.py)
```python
CONFIG = {
    'batch_size': 16,        # Tự động điều chỉnh cho RTX 3050 Ti
    'num_epochs': 30,        # Số epochs
    'learning_rate': 0.0005, # Learning rate
    'cache_size': 500,       # Image cache size
    'num_workers': 2,        # DataLoader workers
}
```

## 📊 Kết quả hiện tại
- **Model**: EfficientNet-B0 Optimized
- **Input Size**: 560x560px
- **Optimization**: RTX 3050 Ti compatible
- **Features**: Mixed precision training, gradient accumulation

### Performance Metrics
- **Accuracy**: Được đánh giá trên test set
- **Precision/Recall**: Per-class metrics
- **F1-Score**: Macro average
- **ROC-AUC**: Binary classification curve

## 🔧 Tối ưu hóa RTX 3050 Ti

### Automatic Optimizations
- **Batch size**: Tự động giảm xuống 6 cho images 560x560
- **Gradient accumulation**: 4 steps để mô phỏng batch size lớn
- **Mixed precision**: Sử dụng AMP để tiết kiệm VRAM
- **Memory management**: Tự động cleanup và cache optimization
- **Lite mode**: Giảm model complexity khi cần thiết

### Memory Usage
- **VRAM**: ~3.5GB cho EfficientNet-B0
- **RAM**: ~2GB cho image caching
- **Cache**: Intelligent LRU caching cho images

## 📈 Training Process
1. **Data Loading**: Lazy loading với intelligent caching
2. **Preprocessing**: Resize, normalization, augmentation
3. **Training**: Mixed precision với gradient accumulation
4. **Validation**: Real-time monitoring
5. **Evaluation**: Comprehensive metrics và visualizations
6. **Comparison**: Side-by-side model comparison

## 📊 Visualizations
- **Training History**: Loss/accuracy curves
- **Confusion Matrix**: Classification performance
- **ROC Curves**: Binary classification analysis
- **Class Distribution**: Dataset balance visualization
- **Model Comparison**: Performance comparison charts

## 🔍 Monitoring và Debugging
- **GPU Memory**: Real-time VRAM monitoring
- **Cache Status**: Image cache utilization
- **Training Progress**: Detailed epoch-by-epoch logs
- **Performance Metrics**: Comprehensive evaluation reports

## 🤝 Đóng góp
1. Fork dự án
2. Tạo feature branch (`git checkout -b feature/MicroExpressionFeature`)
3. Commit changes (`git commit -m 'Add micro-expression feature'`)
4. Push to branch (`git push origin feature/MicroExpressionFeature`)
5. Tạo Pull Request

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.

## 👤 Tác giả
- **Machine Learning Student** - Học kỳ 2 Năm 4
- **Project**: Truth/Lie Classification using Micro-Expression Analysis
- **Focus**: Deep Learning, Computer Vision, Micro-Expression Recognition

## 🙏 Acknowledgments
- [PyTorch](https://pytorch.org/) - Deep Learning framework
- [EfficientNet](https://arxiv.org/abs/1905.11946) - Efficient CNN architecture
- [SE-Net](https://arxiv.org/abs/1709.01507) - Squeeze-and-Excitation Networks
- [Nothing Phone (2)](https://nothing.tech/) - Camera hardware for dataset creation
- **Micro-Expression Research Community** - Inspiration and methodology

## 📚 References
- Micro-Expression Recognition in Psychology and Security Applications
- Deep Learning approaches for Facial Expression Analysis
- Truth/Lie Detection using Computer Vision
- Efficient Neural Networks for Mobile Devices

## 🎯 Future Work
- [ ] Extend to multi-class emotion recognition
- [ ] Real-time inference optimization
- [ ] Mobile deployment (ONNX/TensorRT)
- [ ] Data augmentation for micro-expressions
- [ ] Cross-cultural micro-expression analysis